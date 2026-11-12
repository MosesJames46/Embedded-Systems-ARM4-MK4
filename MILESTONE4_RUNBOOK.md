# Milestone 4: Vision-Guided AR4 Movement (Runbook)

Goal (Checkpoint 1): detect a target in a live webcam image, convert it to a reachable
robot pose, move the physical AR4 safely to hover above it, and do it again after the
target is moved. Grasping is not required.

## System

```
webcam -> ColorDetector (HSV colour, 5-frame stability)
       -> ArUcoCalibration: pixel -> board mm (homography from markers 4-7)
       -> robot_calib.json: board mm -> robot mm (affine, taught by jogging the tool over the markers)
       -> vision_move.py safety checks -> robot_backend.py -> Teensy 4.1 (Annin firmware 8.1)
```

**Everything runs through `src/main.py`:**

| Command | Step |
|---|---|
| `python main.py check` | Find the webcam, confirm all 4 markers are visible |
| `python main.py vision` | Original vision-only test (was `main.py`, now `vision_test.py`) |
| `python main.py teach` | Teach the board-to-robot calibration |
| `python main.py dry` | Full pipeline and safety checks, no motion |
| `python main.py run` | Live demo |
| `python main.py park` | Move the arm up and to the saved park pose |
| `python main.py jog` | Keyboard jog window (also saves calibration points) |

| File | Purpose |
|---|---|
| `src/main.py` | Entry point (above) |
| `src/vision_test.py` | The team's original vision-only main.py |
| `src/vision_config.py` | All settings: marker positions, camera index, heights, speed, safety limits |
| `src/teach_robot.py` | Jog the tool over each marker and save the board-to-robot calibration |
| `src/vision_move.py` | The demo: live detection, safety checks, hover moves, CSV log |
| `src/robot_backend.py` | One interface for firmware 8.1 (official Annin API) and 6.x (fallback) |
| `src/ARrobots/ARrobot.py` | Official Annin Python API (unmodified, from Annin-Robotics/ar4-hmi @ 214558c) |
| `src/ar4link.py` | 6.x serial link, adapted from BengalPirate/AR4_MK4_Team-Writing |
| `firmware/AR4_teensy41_sketch_v8.1/` | Official Annin firmware 8.1 (unmodified) |

## 0. One-time setup on the Ubuntu laptop

```bash
cd ~/Embedded-Systems-ARM4-MK4          # wherever the repo is cloned
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-linux.txt
```

## 1. Flash firmware 8.1 (once)

Why 8.1: it is Annin's current release. It accepts a software **stop** mid-move
(SPACE key in our demo), solves inverse kinematics at the wrist singularity
(J5 = 0, near the straight-down tool pose we use), returns JSON errors for
unreachable targets, and keeps the hardware E-stop interrupt on pin 39 (the red lid button).

1. Arduino IDE → Library Manager: install **ModbusMaster** and **ArduinoJson** (version 7.x).
2. Open `firmware/AR4_teensy41_sketch_v8.1/AR4_teensy41_sketch_v8.1.ino`.
3. Tools → Board: **Teensy 4.1**, USB Type: **Serial**, Port: the Teensy. Upload.
4. Get the matching HMI (8.1) and run it with the existing HMI's Python environment:
   ```bash
   cd ~ && git clone https://github.com/Annin-Robotics/ar4-hmi.git ar4-hmi-8.1
   cd ar4-hmi-8.1 && ~/ar4-hmi/venv/bin/python AR4.py
   ```
   If it reports a missing module, run `~/ar4-hmi/venv/bin/pip install <module>`.
5. In the HMI, select robot model **AR4-MK4** and send it to the board (it reboots).
   Until this is done the board is in safe mode and our scripts refuse to connect.
6. Re-enter your calibration offsets / motor direction settings from the old HMI
   config, then follow the manual's startup checks (jog each joint, limit switch test)
   and **calibrate**. Close the HMI. (Do not power-cycle or replug the Teensy after
   calibrating, or calibrate again.)

Fallback: if 8.1 misbehaves, flash 6.7.1 again and add `--firmware 6` to every command
(for example `python main.py run --firmware 6`; there is no software stop on 6.x, so use the red E-stop).

## 2. Board setup (before every session)

- **Measure** the centre-to-centre distances between the markers: X = marker 4 to 5,
  Y = marker 4 to 7. In `src/vision_config.py` set
  `MARKER_POSITIONS_MM = {4: (0, 0), 5: (X, 0), 6: (X, Y), 7: (0, Y)}`.
  The current 200 mm values are placeholders. (Layout checked: 5 and 7 are next to 4, 6 is diagonal.)
- Run `python main.py check`. It must say **ALL 4 MARKERS VISIBLE** with the robot in its
  park pose. If the wrong camera opens, set `CAMERA_INDEX` in `vision_config.py`.
- Use a **flat** coloured target (paper or card). The camera views the board at an angle, so
  the top of a tall cube appears shifted away from the camera. Keep `TARGET_HEIGHT_MM` small.
- Tape the mat to the table. Mount the camera rigidly. If the mat, camera or robot moves,
  re-teach (step 3).

## 3. Teach the board-to-robot calibration (after any setup change)

HMI closed, arm calibrated:

```bash
cd src && python main.py teach
```

Jog with `x+10`, `y-5`, `z-2` etc. until the tool tip is a few mm above the **centre of
marker 4**, then `save 4`. Repeat for 5, 6 and 7. Jog to a pose clear of the camera view and
`save park`. Then `w` writes `robot_calib.json` and prints the fit error for each marker.
This error goes in the report as calibration evidence. Errors of a few mm are good; tens of
mm mean a wrong marker ID or a wrong measured position.

## 4. Dry run (no motion)

```bash
python main.py dry
```

Drag a box over the cube colour and press ENTER. Check that the board coordinates match a
ruler measurement from marker 4, that moving the cube changes them, and that READY only
appears when all markers are visible and the cube is still and inside the board.

## 5. Live demo

```bash
python main.py run
```

| Key | Action |
|---|---|
| G | Move to hover above the target (only when READY) |
| SPACE | Stop the arm immediately (latched until the next G/P) |
| P | Park |
| C | Choose a new target (re-select its colour) without restarting. Only while the arm is idle |
| M | Switch single-target / multi-target mode |
| 1-9 or click | Multi mode: select a target by its number, or click it in the video |
| A | Multi mode: visit every READY target in turn (hover, retreat), then park |
| + / - | Raise / lower the hover clearance by 5 mm |
| ] / [ | Raise / lower the target height by 5 mm |
| Q | Quit |

The height keys work while the arm is idle. The values are saved in `src/live_settings.json`
and reused on the next run (they override `TARGET_HEIGHT_MM` / `HOVER_CLEARANCE_MM` in
`vision_config.py`; delete the file to go back to the config values). Hover height =
taught mat height + target height + clearance. It is kept at least 20 mm below the travel height.

Motion sequence: rise to the safe height, travel at the safe height, descend to
`TARGET_HEIGHT + HOVER_CLEARANCE` above the board, hold 2 s, rise, park. Every segment is
a straight-line move at 10% speed (capped at 25% in code).

**Safety layers (the safety section of the rubric):**
1. A target is rejected unless all 4 markers are visible, the detector reports a stable
   target, and the last 10 readings agree within 4 mm.
2. A target outside the marker square (minus a 10 mm margin) is rejected, which keeps
   motion inside the workspace bounds.
3. The firmware rejects unreachable poses or poses outside the joint limits (JSON error).
   The move is aborted and the arm retreats upward.
4. The hover height is never below the board plus 20 mm, and every travel happens at the safe height.
5. SPACE gives a software stop mid-move. The red lid button gives a hardware E-stop
   (firmware interrupt on pin 39). The 24 V rocker switch cuts motor power.
6. The scripts refuse to start while the HMI holds the serial port.

### Multi-target mode (several objects of the same colour)

Single mode (the default) uses the team's tracker, which follows one blob. With several
same-colour objects it picks one arbitrarily. Press **M** for multi mode: every blob of the
selected colour is detected (`src/multi_target.py`), converted to board mm and tracked
across frames, so each object keeps its number and gets its own stability check.
Labels show `ready`, `settling`, `moving`, `off board` or `merged? separate objects` (a
blob much larger than the others, likely two touching objects, which is refused). If only
one object is ready it is selected automatically. **A** captures all ready positions
before the arm moves (the arm blocks the camera during motion), visits them in number order,
and parks. SPACE (firmware 8.1) or the E-stop aborts the whole sequence.

## Keyboard jog (`python main.py jog`)

A pygame window that replaces typing `x+10` in teach. Click the window so it has the keys.

| Key | Action |
|---|---|
| W / S | +X / -X (away from / toward the base) |
| A / D | +Y / -Y (robot's left / right) |
| SPACE / CTRL | Up / down |
| UP / DOWN | Step size 1, 2, 5, 10, 20, 50 mm (hold a key to repeat steps) |
| LEFT / RIGHT | Speed 5-25 % |
| G | Gripper open / close (Annin servo gripper on the Nano board) |
| 0-9 / P | Save the tool position as the marker with that ID (IDs in `MARKER_POSITIONS_MM`) / as park |
| ENTER | Write `robot_calib.json` (same file as teach) and show fit errors |
| ESC | Stop the move (8.1 only; on 6.x use the red E-stop) |
| Q | Quit |

With a calibration saved, the tool cannot go below the taught mat height minus 3 mm.
The gripper is driven by the Arduino Nano I/O board (its own USB port, usually
`/dev/ttyUSB0`), not the Teensy. `src/gripper.py` detects whether the Nano runs Annin's
JSON sketch (2.x) or the older text sketch (1.x). Annin's default angles are open 45 and
closed 0; tune `OPEN_POS` / `CLOSED_POS` in `gripper.py`. Use `--gripper-port` if it is
not found, or `--no-gripper` to skip it.

## 6. Evidence for the report

Every move is appended to `src/vision_moves.csv`: pixel position, board mm, commanded
robot XYZ, the robot's reported XYZ and the result. For **positioning error**,
measure the horizontal offset between the tool tip and the cube centre with a ruler
while the arm hovers, and type it into `measured_error_mm`. Do at least 5 positions,
plus 3 repeats of one position for **repeatability**. The reported XYZ shows what the
controller reached; only the ruler measurement shows the true error.

Demonstrate failure handling for the rubric: cover a marker (refused), move the cube off
the board (refused), wave the cube around (refused, jittering), and press SPACE mid-move.

## Acknowledgements

- Annin Robotics: firmware 8.1 and `ARrobot.py` (Annin Open Source Non-Commercial License, see `firmware/ANNIN_LICENSE.txt`).
- `ar4link.py` adapted from BengalPirate/AR4_MK4_Team-Writing.
- `teach_robot.py`, `vision_move.py`, `robot_backend.py` and this runbook were written with AI
  assistance (Claude). They were tested against a simulated firmware 8.1 serial device,
  including stops at each motion phase, and must be validated on the robot with `--dry` first.
