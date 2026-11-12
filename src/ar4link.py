"""Minimal serial link to an AR4 Teensy running Annin firmware 6.x.

Adapted from BengalPirate/AR4_MK4_Team-Writing (ar4link.py), which was tested
on Ubuntu 24.04 with Annin firmware 6.7.1. Changes: loop mode (Lm) is a
parameter, and the serial port defaults to /dev/ttyACM0 like Test_Connection.py.

Every read has a timeout, so a missing reply raises an error instead of
freezing the program.
"""
import glob
import os
import subprocess
import time

import serial

BAUD = 9600


def find_port():
    ports = sorted(glob.glob("/dev/serial/by-id/usb-Teensyduino_USB_Serial_*-if00"))
    if len(ports) == 1:
        return ports[0]
    if len(ports) > 1:
        raise RuntimeError(f"More than one Teensy connected, pass --port: {ports}")
    if os.path.exists("/dev/ttyACM0"):
        return "/dev/ttyACM0"
    raise RuntimeError("No Teensy found - is the USB cable plugged in?")


def stock_app_running():
    out = subprocess.run(["ps", "-axo", "args="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        argv = line.split()
        if len(argv) > 1 and "python" in os.path.basename(argv[0]) \
                and os.path.basename(argv[1]) == "AR4.py":
            return True
    return False


class Pose:
    """Joint angles (deg) and tool pose (mm / deg) parsed from the RP reply."""

    def __init__(self, reply):
        keys = "ABCDEFGHIJKL"
        idx = [reply.index(k) for k in keys] + [reply.index("M")]
        vals = [float(reply[idx[i] + 1:idx[i + 1]]) for i in range(12)]
        self.joints = vals[:6]
        self.x, self.y, self.z, self.rz, self.ry, self.rx = vals[6:]

    @property
    def xyz(self):
        return (self.x, self.y, self.z)

    @property
    def rot(self):
        return (self.rz, self.ry, self.rx)

    def as_dict(self):
        return {"joints": self.joints, "x": self.x, "y": self.y, "z": self.z,
                "rz": self.rz, "ry": self.ry, "rx": self.rx}

    def __str__(self):
        j = " ".join(f"J{i + 1}={a:.1f}" for i, a in enumerate(self.joints))
        return (f"X={self.x:.1f} Y={self.y:.1f} Z={self.z:.1f} "
                f"Rz={self.rz:.1f} Ry={self.ry:.1f} Rx={self.rx:.1f} | {j}")


class AR4Error(RuntimeError):
    pass


class AR4:
    MAX_SPEED = 25  # percent; hard cap for this project

    def __init__(self, port=None, loop_mode="000000"):
        if stock_app_running():
            raise RuntimeError("The AR4 HMI (AR4.py) is open. Close it first - two "
                               "programs on one port corrupt the arm's position.")
        if len(loop_mode) != 6 or set(loop_mode) - {"0", "1"}:
            raise ValueError("loop_mode must be six 0/1 characters, e.g. 000000")
        self.loop_mode = loop_mode
        self.port = port or find_port()
        self.ser = serial.Serial(self.port, BAUD, timeout=0.1)
        time.sleep(2.0)  # Teensy may reset on open
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def _cmd(self, cmd, timeout=3.0):
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\n").encode())
        buf = b""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            buf += self.ser.read(self.ser.in_waiting or 1)
            if buf.endswith(b"\n"):
                break
        reply = buf.decode(errors="replace").strip()
        if not reply:
            raise AR4Error(f"No reply to {cmd[:2]} within {timeout:.0f}s "
                           "(arm still moving, power off, or board hung)")
        return reply

    def pose(self):
        reply = self._cmd("RP")
        if not reply.startswith("A"):
            raise AR4Error(f"Position request returned an alarm: {reply}")
        return Pose(reply)

    def _move(self, kind, x, y, z, rz, ry, rx, speed_pct, timeout):
        # Percent speed only (firmware 6.7.1 mm/s timing can hang the board on short moves).
        if not 1 <= speed_pct <= self.MAX_SPEED:
            raise ValueError(f"speed_pct must be 1-{self.MAX_SPEED}")
        wrist = "F" if self.pose().joints[4] > 0 else "N"
        cmd = (f"{kind}X{x:.3f}Y{y:.3f}Z{z:.3f}Rz{rz:.3f}Ry{ry:.3f}Rx{rx:.3f}"
               f"J70.00J80.00J90.00Sp{speed_pct:.0f}Ac10Dc10Rm50")
        if kind == "ML":
            cmd += "Rnd0"
        cmd += f"W{wrist}Lm{self.loop_mode}"
        if kind == "ML":
            cmd += "Q0"
        reply = self._cmd(cmd, timeout)
        if not reply.startswith("A"):
            raise AR4Error(f"Move refused or faulted ({reply}) - target may be "
                           "out of reach or outside joint limits")
        return Pose(reply)

    def move_joints(self, angles, speed_pct=10):
        """Move to absolute joint angles J1..J6 (deg) with the RJ command."""
        if not 1 <= speed_pct <= self.MAX_SPEED:
            raise ValueError(f"speed_pct must be 1-{self.MAX_SPEED}")
        wrist = "F" if angles[4] > 0 else "N"
        cmd = ("RJ" + "".join(f"{L}{a:.3f}" for L, a in zip("ABCDEF", angles))
               + f"J70.00J80.00J90.00Sp{speed_pct:.0f}Ac10Dc10Rm80W{wrist}Lm{self.loop_mode}")
        travel = max(abs(a - b) for a, b in zip(angles, self.pose().joints))
        reply = self._cmd(cmd, timeout=20 + 3 * travel / speed_pct)
        if not reply.startswith("A"):
            raise AR4Error(f"Joint move refused or faulted ({reply}) - outside joint limits?")
        return Pose(reply)

    def move_linear(self, x, y, z, rz, ry, rx, speed_pct=10):
        dist = sum((a - b) ** 2 for a, b in zip((x, y, z), self.pose().xyz)) ** 0.5
        return self._move("ML", x, y, z, rz, ry, rx, speed_pct,
                          timeout=10 + 2 * dist / speed_pct)

    def move_joint(self, x, y, z, rz, ry, rx, speed_pct=10):
        dist = sum((a - b) ** 2 for a, b in zip((x, y, z), self.pose().xyz)) ** 0.5
        return self._move("MJ", x, y, z, rz, ry, rx, speed_pct,
                          timeout=20 + 3 * dist / speed_pct)
