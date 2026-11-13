"""One interface for both firmware generations.

  V8Robot     - Annin firmware 8.1 (JSON protocol) via the official ARRobot API.
                Every move is streamed (spline_start / queued move_l / spline_end),
                which is the only mode in which the firmware accepts a software
                stop mid-move. stop() can be called from another thread.
  LegacyRobot - Annin firmware 6.x text protocol via ar4link.py. No software stop.

Both return poses as dicts: {"x","y","z","rot":[a,b,c],"joints":[...]} where
"rot" is the three orientation values exactly as the firmware reports them.
They are passed back unchanged, so the axis order never has to be guessed.
"""
import glob
import json
import os
import time

MAX_SPEED = 25  # percent, hard cap for this project


class RobotFault(RuntimeError):
    pass


def default_port():
    ports = sorted(glob.glob("/dev/serial/by-id/usb-Teensyduino_USB_Serial_*-if00"))
    if len(ports) > 1:
        raise RuntimeError(f"More than one Teensy connected, pass --port: {ports}")
    if ports:
        return ports[0]
    if os.path.exists("/dev/ttyACM0"):
        return "/dev/ttyACM0"
    raise RuntimeError("No Teensy found - is the USB cable plugged in?")


def hmi_running():
    import subprocess
    out = subprocess.run(["ps", "-axo", "args="], capture_output=True, text=True).stdout
    return any("AR4.py" in line and "python" in line for line in out.splitlines())


def _check_speed(speed):
    if not 1 <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be 1-{MAX_SPEED} percent")


class V8Robot:
    supports_stop = True

    def __init__(self, port=None):
        from ARrobots.ARrobot import ARRobot
        if hmi_running():
            raise RuntimeError("The AR4 HMI is open. Close it first - one program per port.")
        self.port = port or default_port()
        self.r = ARRobot(port=self.port, baudrate=115200, timeout=0.2)
        if not self.r.connect():
            raise RuntimeError(f"Could not open {self.port}")
        time.sleep(1.0)
        self.r.serial_connection.reset_input_buffer()
        self.stopped = False
        hello = self.r.command({"cmd": "hello"}, timeout=5)
        try:
            self.hello = json.loads(hello)
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError(f"Not firmware 8.x (hello reply: {hello!r}). "
                               "Flash 8.1 or run with --firmware 6")
        if not self.hello.get("Configured"):
            raise RuntimeError("Board is in SAFE MODE (robot model not set). Open the 8.1 HMI "
                               f"and set the robot ID (AR4, MK4) first. hello: {self.hello}")

    @staticmethod
    def _parse(reply):
        if isinstance(reply, str):
            try:
                reply = json.loads(reply)
            except json.JSONDecodeError:
                raise RobotFault(f"Unexpected reply: {reply}")
        if not isinstance(reply, dict):
            raise RobotFault(f"No reply from robot: {reply}")
        if reply.get("msg") == "error":
            raise RobotFault(f"Robot error: {reply}")
        if reply.get("msg") != "robot_pos":
            raise RobotFault(f"Unexpected reply: {reply}")
        p = reply["pose"]
        pose = {"x": p[0], "y": p[1], "z": p[2], "rot": list(p[3:6]),
                "joints": reply["j"][:6], "flag": reply.get("flag", "")}
        if pose["flag"]:
            raise RobotFault(f"Robot alarm flag '{pose['flag']}' (E-stop or fault). "
                             f"Pose {pose['x']:.1f},{pose['y']:.1f},{pose['z']:.1f}")
        return pose

    def pose(self):
        return self._parse(self.r.command({"cmd": "get_position"}, timeout=5))

    def move_linear(self, x, y, z, rot, speed=10):
        """Straight-line move, streamed so stop() can interrupt it."""
        _check_speed(speed)
        if self.stopped:  # latched until clear_stop(), so a stop between moves also counts
            raise RobotFault("Stopped by operator")
        reply = self.r.command({"cmd": "spline_start"}, timeout=5)
        if str(reply).strip() != "SL":
            raise RobotFault(f"spline_start refused: {reply}")
        move = {"cmd": "move_l", "pose": [x, y, z] + list(rot), "ext": [0, 0, 0],
                "spd_type": "percent", "spd": speed, "acc": 10, "dec": 10,
                "rounding": 0, "w": "A"}
        ack = self.r.queue_move(move, timeout=10)
        kind = self.r._msg_type(ack)
        if kind == "error":
            self.r.command({"cmd": "spline_end"}, timeout=10)
            raise RobotFault(f"Move rejected (unreachable or outside limits): {ack}")
        if kind == "robot_pos":  # a stop arrived before the ack
            self.stopped = True
            raise RobotFault("Stopped by operator")
        final = self.r.command({"cmd": "spline_end"}, timeout=120)
        if self.stopped:
            raise RobotFault("Stopped by operator")
        return self._parse(final)

    def move_joints(self, angles, speed=10):
        """Move to absolute joint angles J1..J6 (deg)."""
        _check_speed(speed)
        if self.stopped:
            raise RobotFault("Stopped by operator")
        reply = self.r.command({"cmd": "move_joints", "j": list(angles)[:6] + [0, 0, 0],
                                "spd_type": "percent", "spd": speed, "acc": 10, "dec": 10},
                               timeout=120)
        return self._parse(reply)

    def move_joint(self, x, y, z, rot, speed=10):
        """Joint-interpolated move (path not straight). Used for parking."""
        _check_speed(speed)
        if self.stopped:
            raise RobotFault("Stopped by operator")
        reply = self.r.command({"cmd": "move_j", "pose": [x, y, z] + list(rot),
                                "spd_type": "percent", "spd": speed, "acc": 10,
                                "dec": 10, "w": "A"}, timeout=120)
        return self._parse(reply)

    def stop(self):
        self.stopped = True
        self.r.stop()

    def clear_stop(self):
        self.stopped = False

    def close(self):
        self.r.disconnect()


class LegacyRobot:
    """Annin 6.x. Cartesian moves are planned on the laptop (kinematics.py) and sent as
    direct joint moves (RJ): the 6.x firmware's own IK can jump to the mirrored wrist
    (J4/J5/J6 swing ~180 deg) near the wrist singularity. Falls back to the firmware's
    moves if the laptop model does not match this robot (e.g. a tool frame set in the HMI)."""
    supports_stop = False

    def __init__(self, port=None, loop_mode="000000", local_ik=True):
        from ar4link import AR4
        import kinematics as K
        self.K = K
        self.a = AR4(port, loop_mode=loop_mode)
        self.port = self.a.port
        self.local_ik = False
        if local_ik:
            p = self.a.pose()
            if K.model_matches(p.joints, p.xyz, [p.rz, p.ry, p.rx]):
                self.local_ik = True
            else:
                print("NOTE: the laptop arm model does not match this controller (tool frame or "
                      "DH changed in the HMI?) - using the firmware's own moves.")

    @staticmethod
    def _conv(p):
        return {"x": p.x, "y": p.y, "z": p.z, "rot": [p.rz, p.ry, p.rx],
                "joints": p.joints, "flag": ""}

    def pose(self):
        from ar4link import AR4Error
        try:
            return self._conv(self.a.pose())
        except AR4Error:
            # 6.x answers the first request after a refused move with a stale "ER"
            time.sleep(0.3)
            try:
                return self._conv(self.a.pose())
            except AR4Error as e:
                raise RobotFault(str(e))

    def move_linear(self, x, y, z, rot, speed=10):
        from ar4link import AR4Error
        _check_speed(speed)
        try:
            if not self.local_ik:
                return self._conv(self.a.move_linear(x, y, z, *rot, speed_pct=speed))
            q0 = self.a.pose().joints
            waypoints = self.K.plan_line(q0, (x, y, z), rot)
            p = None
            for q in waypoints:
                p = self.a.move_joints(list(q), speed_pct=speed)
            return self._conv(p) if p is not None else self.pose()
        except AR4Error as e:
            raise RobotFault(str(e))
        except self.K.KinematicsError as e:
            raise RobotFault(f"Move refused: {e}")

    def move_joints(self, angles, speed=10):
        from ar4link import AR4Error
        _check_speed(speed)
        try:
            return self._conv(self.a.move_joints(angles, speed_pct=speed))
        except AR4Error as e:
            raise RobotFault(str(e))

    def move_joint(self, x, y, z, rot, speed=10):
        """Joint-interpolated move to a pose (path not straight). Used for parking."""
        from ar4link import AR4Error
        _check_speed(speed)
        try:
            if self.local_ik:
                import numpy as np
                q0 = self.a.pose().joints
                q = self.K.ik(np.array([x, y, z], float), self.K.rotvec_to_R(rot), q0)
                self.K.check_limits(q)
                return self._conv(self.a.move_joints(list(q), speed_pct=speed))
            return self._conv(self.a.move_joint(x, y, z, *rot, speed_pct=speed))
        except self.K.KinematicsError as e:
            raise RobotFault(f"Move refused: {e}")
        except AR4Error as e:
            raise RobotFault(str(e))

    def stop(self):
        pass  # not supported by 6.x firmware mid-move; use the hardware E-stop

    def clear_stop(self):
        pass

    def close(self):
        self.a.close()


def connect(firmware="8", port=None, loop_mode="000000"):
    if firmware == "8":
        return V8Robot(port)
    return LegacyRobot(port, loop_mode)
