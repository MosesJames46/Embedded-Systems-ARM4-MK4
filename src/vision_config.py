"""Shared settings for the vision-guided move demo. Edit these to match the setup."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "robot_calib.json")
LOG_FILE = os.path.join(HERE, "vision_moves.csv")

# Measured centres of the ArUco markers on the board, in mm.
# !!! PLACEHOLDER (200 mm square from main.py). MEASURE the real mat with a tape:
# marker 4 = origin, 4->5 along +X, 4->7 along +Y, 6 diagonal from 4.
MARKER_POSITIONS_MM = {
    4: (0, 0),
    5: (200, 0),
    6: (200, 200),
    7: (0, 200),
}

CAMERA_INDEX = 3          # same as main.py

# Loop mode sent to the firmware: one char per joint, 0 = encoders on (closed loop),
# 1 = encoders off. Test_Connection.py used 000000. Must match the HMI setting.
LOOP_MODE = "000000"

SPEED_PCT = 10            # move speed, percent (capped at 25 in ar4link)

# Heights relative to the taught board height (mm)
TARGET_HEIGHT_MM = 25     # height of the cube top above the board - MEASURE IT
HOVER_CLEARANCE_MM = 40   # how far above the cube top the tool stops
SAFE_Z_MM = 150           # travel height above the board

# Safety
BOARD_MARGIN_MM = 10      # target must be inside the marker square, shrunk by this
STABLE_SAMPLES = 10       # how many recent positions must agree
STABLE_SPREAD_MM = 4.0    # ...within this spread
HOLD_SECONDS = 2.0        # pause above the target before retreating

# "8" = Annin firmware 8.1 (recommended, supports software stop). "6" = old 6.x firmware.
FIRMWARE = "8"
