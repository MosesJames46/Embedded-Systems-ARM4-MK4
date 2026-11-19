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
import os
import time

import serial

BAUD = 9600            # Annin ARNano default; the tool can also try 115200
SERVO_NUM = 0
OPEN_POS = 45
CLOSED_POS = 0
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_settings.json")


def load_settings():
    """Saved by `python main.py gripper` (port, baud, protocol, open, closed)."""
    try:
        with open(SETTINGS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def candidate_ports():
    """Serial ports that are not the Teensy (the arm controller)."""
    out = []
    for p in sorted(glob.glob("/dev/serial/by-id/*")):
        if "teensy" not in p.lower():
            out.append(p)
    real = {os.path.realpath(p) for p in out}
    for p in sorted(glob.glob("/dev/ttyUSB*")):
        if p not in real:
            out.append(p)
    return out


def find_port():
    """The Nano usually appears as /dev/ttyUSB0 (CH340/FTDI); the Teensy is /dev/ttyACM*."""
    saved = load_settings().get("port")
    if saved and os.path.exists(saved):
        return saved
    ports = candidate_ports()
    return ports[0] if ports else None


class Gripper:
    def __init__(self, port=None, protocol=None, open_pos=None, closed_pos=None, baud=None):
        st = load_settings()
        self.port = port or find_port()
        if self.port is None:
            raise RuntimeError("No gripper board found (expected the Nano on /dev/ttyUSB*). "
                               "Is its USB cable plugged in? Run: python main.py gripper")
        self.open_pos = st.get("open", OPEN_POS) if open_pos is None else open_pos
        self.closed_pos = st.get("closed", CLOSED_POS) if closed_pos is None else closed_pos
        self.baud = baud or st.get("baud", BAUD)
        protocol = protocol or st.get("protocol", "auto")
        self.ser = serial.Serial(self.port, self.baud, timeout=0.5)
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

    def open(self, settle=0.6, pos=None):
        self.servo(self.open_pos if pos is None else pos)
        time.sleep(settle)
        self.state = "open"

    def close(self, settle=0.6, pos=None):
        self.servo(self.closed_pos if pos is None else pos)
        time.sleep(settle)
        self.state = "closed"

    def toggle(self):
        (self.open if self.state != "open" else self.close)()

    def shutdown(self):
        try:
            self.ser.close()
        except Exception:
            pass
