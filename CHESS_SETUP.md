# Chess Phase 1: Teaching the Robot the Board

Goal: the robot and the camera both know where every one of the 64 squares is, in robot
coordinates, with nothing measured by hand.

```
jog the gripper over 4 ArUco cards  ->  camera pixels map to robot mm   (robot_calib.json)
jog the gripper over 4 corner squares -> all 64 square centres in robot mm (chess_board.json)
```

## Firmware

Keep the firmware **already on the Teensy** (Annin 6.x, used with the HMI in `~/ar4-hmi`).
It is proven with this code. Flash 8.1 (in `firmware/`) only if testing shows the
wrist-singularity problem (J5 near 0 makes the wrist spin). All commands below use
`--firmware 6` once; the calibration files then remember it.

## A. Every session

1. Power the robot. In a terminal, calibrate in the HMI, then **close the HMI**:
   ```bash
   cd ~/ar4-hmi && source venv/bin/activate && python AR4.py
   ```
2. In a **new** terminal, open the project:
   ```bash
   cd ~/Embedded-Systems-ARM4-MK4 && git pull
   cd src && source ../.venv/bin/activate      # or ../venv, whichever exists
   ```

## B. One-time setup (repeat if the board, cards, camera or robot move)

3. **Marker IDs.** The cards are now 0-3, and distances are not needed. Run once:
   ```bash
   python3 - <<'EOF'
   import re
   p = "vision_config.py"; s = open(p).read()
   s = re.sub(r"MARKER_POSITIONS_MM = \{.*?\n\}",
              "MARKER_POSITIONS_MM = {\n    0: (0, 0),\n    1: (300, 0),\n    2: (300, 300),\n    3: (0, 300),\n}",
              s, flags=re.S)
   open(p, "w").write(s)
   EOF
   python main.py check          # must say ALL 4 MARKERS VISIBLE
   ```
   (The 300 mm values are placeholders. Once the cards are taught, their jogged positions
   are used instead.)

4. **Check the gripper does not turn by itself** before teaching (see the jog-window J/rot
   numbers). Teaching with a drifting wrist gives wrong positions.

5. **Teach the cards, park and board corners** in one jog session:
   ```bash
   python main.py jog --firmware 6
   ```
   Click the window first. **First raise the arm (SPACE) above Z 120 and press `V`**: the
   gripper turns to point **straight down** and stays that way. This matters: a tilted
   gripper forces J4/J5 to twist (and flip 180 deg) whenever the base turns; straight down,
   only J6 turns. Use `Z` / `X` to turn the fingers if needed. Then use the **same height**
   (tip about 10 mm above the surface) for every point. If J4 shows more than 90 deg,
   the window warns you.

   | Do | Key |
   |---|---|
   | Tip over the centre of card **0** | `0` |
   | ... cards **1**, **2**, **3** | `1` `2` `3` |
   | Raise, move clear of the board and out of the camera view | `P` (park) |
   | Write the camera calibration | `ENTER` |
   | Tip over the centre of square **a1** | `B` |
   | ... **h1**, then **h8**, then **a8** (the window says which is next) | `B` `B` `B` |

   **a1** = the dark corner square on **White's left**. Decide now which side the robot
   plays: the side closest to the robot is ranks 7-8 if the robot is Black.
   After the 4th corner the window shows the square size, edge lengths and corner angle.
   **Amber lines mean re-teach** (press `N` to restart the corners).

   If a corner cannot be reached (the move is refused), the board must move closer to the
   robot. Every square must be reachable for chess.

## C. Verify (the evidence for the report)

6. **Arm check**: the arm hovers over any square you type:
   ```bash
   python main.py board
   ```
   Type `corners` (it visits a1, h1, h8, a8), then a few squares such as `e4`, `d5`, `a5`, `h4`.
   The tip should sit over each square's centre. Note any offset in mm for the report.
   `h 20` lowers the hover height, `p` parks, `q` quits.

7. **Camera check**: the 64-square grid drawn on the live image:
   ```bash
   python main.py boardview
   ```
   The yellow grid should sit on the real squares. Click any square: its name and robot
   X/Y are printed. Clicks on a line or off the board are refused (`not on a square`).

Phase 1 is done when both checks pass. Commit the calibration so the team shares it:
```bash
cd ~/Embedded-Systems-ARM4-MK4
git add src/robot_calib.json src/chess_board.json src/vision_config.py
git commit -m "Board calibration" && git push
```

## D. Piece recognition (classifier, no tags)

The camera crops every square (plus the space above it, where a standing piece appears)
and a classifier trained on **your** pieces, board, camera angle and lighting labels it:
empty, or one of the 12 pieces. Labelling is automatic: you set up a position the
program already knows and every square is saved with its correct label.

8. Install the new dependencies once: `pip install -r ../requirements-linux.txt`
9. **Capture training data** (`python main.py capture`), in the lighting you will demo in:
   - `S` (starting position), set the real board to match, then `B` (burst of 5 frames). Repeat 2 times,
     shifting a lamp or blind between bursts.
   - `N` (random placement): the terminal prints where all 32 pieces go. Set them up exactly,
     then `B`. **Do 15-20 different placements.** This is what teaches every piece on
     light and dark squares, near and far from the camera.
   - The more placements, the better: in testing, accuracy rose from 78% to 94% when the
     data went from 13 to 36 positions.
10. **Train**: `python main.py train` (about a minute). It prints accuracy on frames it never
    saw, plus a confusion matrix showing which pieces it mixes up. Capture more
    placements if a piece type scores low, then retrain.
11. **Check live**: `python main.py recognize`. Every square is labelled, and red `?` marks
    a square below 60% confidence. `C` compares with the starting position and lists
    mismatches. `SPACE` prints the board and its FEN.

**How errors are handled in the game:** the chess program always knows what the board
*should* be. A human move is accepted only if it is a **legal** move that matches what the
camera sees; otherwise the system stops and asks for correction. Empty vs occupied was
100% in testing, so recognition mistakes show up as a mismatch, never as a silent wrong
move.

`dataset/` (thousands of crop images) stays on the laptop and is not committed. Commit
`src/piece_model.joblib` (the trained model) instead.

## Next phases

3. **Chess program**: python-chess for rules, turns and special moves; Stockfish selects moves.
4. **Pick and place**: grasp and release heights per piece, captures to a drop zone, and
   board verification after every move.
