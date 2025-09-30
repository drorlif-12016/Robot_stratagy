# -*- coding: utf-8 -*-
"""
Robot Performance Calculator (with Gate, Intake & Travel)

This program calculates the number of points a robot can score
based on user-defined inputs:
- Artifacts it can hold
- Shooting speed (shots per second)
- Shooting zone (close or far, for distance only in inches)
- Intake time
- Travel speed (seconds)
- Times the robot opens the gate
"""

# --- Constants ---
AUTO_PERIOD_TIME = 30
TELEOP_PERIOD_TIME = 100  # 1 min 24s
ENDGAME_PERIOD_TIME = 20
PARKING_TIME = 3

# Shooting distances
CLOSE_SHOOTING_ZONE = 64  # inches
FAR_SHOOTING_ZONE = 84    # inches

# Gate scoring system
CLASSIFIED_ARTIFACT_POINTS = 3
OVERFLOW_ARTIFACT_POINTS = 1
CYCLES_BEFORE_GATE_OPENS = 3


def calculate_performance():
    print("Robot Performance Calculator")

    try:
        artifacts_per_load = int(input("Enter how many artifacts the robot can hold at once: "))
        shooting_speed = float(input("Enter shooting speed (shots per second): "))
        shooting_zone = input("Enter shooting zone (close/far): ").strip().lower()
        intake_time = float(input("Enter time to intake balls (seconds per full load): "))
        travel_time = float(input("Enter travel time (seconds per trip between intake and shooting): "))
        gate_openings = int(input("Enter the number of times the robot will open the gate: "))

        if artifacts_per_load <= 0 or shooting_speed <= 0 or intake_time < 0 or travel_time < 0 or gate_openings < 0:
            print("\nError: Please enter positive values for artifacts per load, shooting speed, and non-negative values for intake/travel/gate openings.")
            return

        if shooting_zone not in ["close", "far"]:
            print("\nError: Invalid shooting zone. Please enter 'close' or 'far'.")
            return

    except ValueError:
        print("\nError: Invalid input. Please enter numbers only where required.")
        return

    # --- Select shooting distance ---
    if shooting_zone == "close":
        distance = CLOSE_SHOOTING_ZONE
    else:
        distance = FAR_SHOOTING_ZONE

    # --- Time breakdown ---
    auto_time = AUTO_PERIOD_TIME
    teleop_time = TELEOP_PERIOD_TIME
    endgame_time = max(0, ENDGAME_PERIOD_TIME - PARKING_TIME)
    total_time = auto_time + teleop_time + endgame_time

    # --- Cycle time calculation ---
    shooting_time = artifacts_per_load / shooting_speed
    cycle_time = intake_time + travel_time + shooting_time

    if cycle_time <= 0:
        print("\nError: Cycle time calculated as zero or negative. Check inputs.")
        return

    # --- How many full cycles fit in match ---
    total_cycles = int(total_time // cycle_time)

    # --- Gate scoring logic ---
    max_classified_cycles = gate_openings * CYCLES_BEFORE_GATE_OPENS
    classified_cycles = min(total_cycles, max_classified_cycles)
    overflow_cycles = max(0, total_cycles - max_classified_cycles)

    classified_artifacts = classified_cycles * artifacts_per_load
    overflow_artifacts = overflow_cycles * artifacts_per_load

    classified_points = classified_artifacts * CLASSIFIED_ARTIFACT_POINTS
    overflow_points = overflow_artifacts * OVERFLOW_ARTIFACT_POINTS

    total_points = classified_points + overflow_points

    # --- Performance results ---
    print("\n" + "=" * 30)
    print("PERFORMANCE RESULTS")
    print("=" * 30)

    print(f"Shooting Zone Selected: {shooting_zone.capitalize()} ({distance} inches)")
    print(f"Artifacts per Load    : {artifacts_per_load}")
    print(f"Shooting Speed        : {shooting_speed:.2f} shots/sec")
    print(f"Intake Time per Load  : {intake_time:.2f} sec")
    print(f"Travel Time per Trip  : {travel_time:.2f} sec")
    print(f"Gate Openings Allowed : {gate_openings}")

    print("\n[Match Time Breakdown]")
    print(f"Autonomous Period : {auto_time} sec")
    print(f"TeleOp Period     : {teleop_time} sec")
    print(f"Endgame Period    : {endgame_time} sec (after reserving {PARKING_TIME}s for parking)")
    print("---------------------------------")
    print(f"Total Match Time  : {total_time} sec")

    print("\n[Cycle Breakdown]")
    print(f"Shooting Time per Cycle : {shooting_time:.2f} sec")
    print(f"Total Cycle Time        : {cycle_time:.2f} sec")
    print(f"Total Cycles Possible   : {total_cycles}")

    print("\n[Scoring Breakdown]")
    print(f"Classified Cycles  : {classified_cycles}  → {classified_artifacts} artifacts → {classified_points} points")
    print(f"Overflow Cycles    : {overflow_cycles}  → {overflow_artifacts} artifacts → {overflow_points} points")
    print("---------------------------------")
    print(f"TOTAL ESTIMATED POINTS: {total_points}")
    print("=" * 30)


# --- Call the function ---
if __name__ == "__main__":
    calculate_performance()

