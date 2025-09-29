# -*- coding: utf-8 -*-
"""
Robot Performance Calculator

This program calculates the number of cycles and points a robot can score
in a robotics competition based on user-defined inputs.
"""

# --- Constants for Game Periods and Points ---
AUTO_PERIOD_TIME = 30
TELEOP_PERIOD_TIME = 846  # 1 minute and 40 seconds
ENDGAME_PERIOD_TIME = 20
PARKING_TIME = 3

CLASSIFIED_ARTIFACT_POINTS = 3
OVERFLOW_ARTIFACT_POINTS = 1

ARTIFACTS_PER_CYCLE = 3
CYCLES_BEFORE_GATE_OPENS = 3


def calculate_performance():
    print("Robot Performance Calculator")

    try:
        cycle_time = int(input("Enter the robot's average cycle time (in seconds): "))
        gate_openings = int(input("Enter the number of times the robot will open the gate: "))

        if cycle_time <= 0 or gate_openings < 0:
            print("\nError: Please enter positive values for cycle time and a non-negative number for gate openings.")
            return

    except ValueError:
        print("\nError: Invalid input. Please enter numbers only.")
        return

    auto_cycles = AUTO_PERIOD_TIME // cycle_time
    teleop_cycles = TELEOP_PERIOD_TIME // cycle_time

    endgame_available_time = ENDGAME_PERIOD_TIME - PARKING_TIME
    if endgame_available_time < 0:
        endgame_available_time = 0
    endgame_cycles = endgame_available_time // cycle_time

    total_cycles = auto_cycles + teleop_cycles + endgame_cycles

    max_classified_cycles = gate_openings * CYCLES_BEFORE_GATE_OPENS

    if total_cycles <= max_classified_cycles:
        classified_cycles_scored = total_cycles
        overflow_cycles_scored = 0
    else:
        classified_cycles_scored = max_classified_cycles
        overflow_cycles_scored = total_cycles - max_classified_cycles

    classified_points = classified_cycles_scored * ARTIFACTS_PER_CYCLE * CLASSIFIED_ARTIFACT_POINTS
    overflow_points = overflow_cycles_scored * ARTIFACTS_PER_CYCLE * OVERFLOW_ARTIFACT_POINTS

    total_points = classified_points + overflow_points

    print("" + "=" * 30)
    print("PERFORMANCE RESULTS")
    print("=" * 30)

    print("\n[Cycle Breakdown]")
    print(f"Autonomous Cycles : {int(auto_cycles)}")
    print(f"TeleOp Cycles     : {int(teleop_cycles)}")
    print(f"Endgame Cycles    : {int(endgame_cycles)} (after reserving {PARKING_TIME}s for parking)")
    print("---------------------------------")
    print(f"Total Possible Cycles: {int(total_cycles)}")

    print("[Scoring Breakdown]")
    print(
        f"Classified Artifacts Scored: {int(classified_cycles_scored * ARTIFACTS_PER_CYCLE)} ({int(classified_points)} points)")
    print(
        f"Overflow Artifacts Scored  : {int(overflow_cycles_scored * ARTIFACTS_PER_CYCLE)} ({int(overflow_points)} points)")
    print("---------------------------------")
    print(f"Total Estimated Points: {int(total_points)}")
    print("=" * 30)


# --- Call the function ---
if __name__ == "__main__":
    calculate_performance()
