"""Annin servo gripper, driven by the Arduino Nano I/O board (its own USB port).

The Teensy does not drive the servo: the Nano does, on servo header 0. Two Nano
firmware generations exist:
  * Nano sketch 2.x (ships with Annin 8.x): JSON  {"cmd":"servo","num":0,"pos":45}
  * Nano sketch 1.x (ships with Annin 6.x): text  SV0P45
Gripper(protocol="auto") asks the board "hello" in JSON and falls back to text.

Positions are servo angles (degrees). Annin's defaults: open 45, closed 0.
Tune OPEN_POS / CLOSED_POS for your gripper and piece size (or pass them in).
"""
import glob
import json
import time

import serial

BAUD = 9600
SERVO_NUM = 0
OPEN_POS = 45
CLOSED_POS = 0


def find_port():
    """The Nano usually appears as /dev/ttyUSB0 (CH340/FTDI); the Teensy is /dev/ttyACM*."""
    ports = sorted(glob.glob("/dev/serial/by-id/*") )
    ports = [p for p in ports if "Teensy" not in p] or sorted(glob.glob("/dev/ttyUSB*"))
    return ports[0] if ports else None


class Gripper:
    def __init__(self, port=None, protocol="auto", open_pos=OPEN_POS, closed_pos=CLOSED_POS):
        self.port = port or find_port()
        if self.port is None:
            raise RuntimeError("No gripper board found (expected the Nano on /dev/ttyUSB*). "
                               "Is its USB cable plugged in?")
        self.open_pos, self.closed_pos = open_pos, closed_pos
        self.ser = serial.Serial(self.port, BAUD, timeout=0.5)
        time.sleep(2.0)  # Nano resets when the port opens
        self.ser.reset_input_buffer()
        self.protocol = self._detect() if protocol == "auto" else protocol
        self.state = None  # "open" / "closed" once commanded

    def _readline(self, timeout=1.0):
        end = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < end:
            buf += self.ser.read(self.ser.in_waiting or 1)
            if buf.endswith(b"\n"):
                break
        return buf.decode(errors="replace").strip()

    def _detect(self):
        self.ser.write(b'{"cmd":"hello"}\n')
        reply = self._readline(1.5)
        try:
            json.loads(reply)
            return "json"
        except (json.JSONDecodeError, ValueError):
            self.ser.reset_input_buffer()
            return "text"

    def servo(self, pos):
        pos = int(max(0, min(180, pos)))
        if self.protocol == "json":
            self.ser.write((json.dumps({"cmd": "servo", "num": SERVO_NUM, "pos": pos}) + "\n").encode())
        else:
            self.ser.write(f"SV{SERVO_NUM}P{pos}\n".encode())
        reply = self._readline(1.0)  # 1.x may not answer; that is fine
        if '"error"' in reply:
            raise RuntimeError(f"Gripper error: {reply}")
        return reply

    def open(self, settle=0.6):
        self.servo(self.open_pos)
        time.sleep(settle)
        self.state = "open"

    def close(self, settle=0.6):
        self.servo(self.closed_pos)
        time.sleep(settle)
        self.state = "closed"

    def toggle(self):
        (self.open if self.state != "open" else self.close)()

    def shutdown(self):
        try:
            self.ser.close()
        except Exception:
            pass
