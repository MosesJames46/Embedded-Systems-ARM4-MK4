import serial
import time

PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

SAFE_COMMAND = (
    "MJX299.419Y-307.921Z460.991"
    "Rz2.163Ry129.588Rx-25.402"
    "J70.00J80.00J90.00"
    "Sp25Ac15Dc15Rm80WFLm000000\n"
)

def read_position(robot):
    robot.reset_input_buffer()
    robot.write(b"RP\n")
    robot.flush()
    response = robot.readline().decode(
        "utf-8", errors="replace"
    ).strip()
    return response


with serial.Serial(PORT, BAUD_RATE, timeout=10) as robot:
    time.sleep(2)

    print("Reading current position...")
    current = read_position(robot)
    print("Current:", current)

    if not current.startswith("A"):
        raise RuntimeError(
            "Could not verify current robot position."
        )

    print("\nTarget: X299.419 Y-307.921 Z460.991")
    print("Orientation: Rz2.163 Ry129.588 Rx-25.402")
    print("Speed: 25")

    print("\nVerify:")
    print("- Robot is calibrated and ready")
    print("- Movement path has been tested in GUI")
    print("- GUI is closed")
    print("- Workspace is clear")
    print("- Emergency stop is accessible")

    confirmation = input(
        "\nType MOVE to send command: "
    )

    if confirmation != "MOVE":
        print("Movement cancelled.")
    else:
        robot.reset_input_buffer()
        robot.write(SAFE_COMMAND.encode("utf-8"))
        robot.flush()

        response = robot.readline().decode(
            "utf-8", errors="replace"
        ).strip()

        print("Controller response:", response)

        if response.startswith("E"):
            print("Controller reported an error.")
        elif not response:
            print("No response received; verify robot state.")
        else:
            print("Movement response received.")

