# -*- coding: utf-8 -*-
"""
Robot Performance Calculator (with Gate Timing Breakdown)
---------------------------------------------------------
Calculates robot performance including how often the gate opens
and how it affects cycle counts in each period.
"""

# --- Constants for Game Periods and Points ---
AUTO_PERIOD_TIME = 30
TELEOP_PERIOD_TIME = 100  # 1 minute and 40 seconds
ENDGAME_PERIOD_TIME = 20
PARKING_TIME = 3  # reserve for parking

# --- Scoring Values ---
CLASSIFIED_ARTIFACT_POINTS = 3
OVERFLOW_ARTIFACT_POINTS = 1

# --- Robot/Game Config ---
ARTIFACTS_PER_CYCLE = 3
CYCLES_BEFORE_GATE_OPENS = 3
GATE_OPEN_DELAY = 2  # seconds per opening

def calculate_performance():
    print("Robot Performance Calculator")
    print("=" * 40)

    try:
        cycle_time = float(input("Enter the robot's average cycle time (in seconds): "))
        gate_openings = int(input("Enter the total number of gate openings expected: "))

        if cycle_time <= 0 or gate_openings < 0:
            print("\nError: Please enter positive values for cycle time and non-negative for gate openings.")
            return
    except ValueError:
        print("\nError: Invalid input. Please enter valid numbers.")
        return

    # --- Adjust available time due to gate opening delays ---
    total_gate_time = gate_openings * GATE_OPEN_DELAY
    total_available_time = AUTO_PERIOD_TIME + TELEOP_PERIOD_TIME + ENDGAME_PERIOD_TIME - PARKING_TIME
    adjusted_total_time = max(total_available_time - total_gate_time, 0)

    # --- Time ratios ---
    total_period_time = AUTO_PERIOD_TIME + TELEOP_PERIOD_TIME + ENDGAME_PERIOD_TIME
    auto_ratio = AUTO_PERIOD_TIME / total_period_time
    teleop_ratio = TELEOP_PERIOD_TIME / total_period_time
    endgame_ratio = ENDGAME_PERIOD_TIME / total_period_time

    # --- Adjusted period times based on total gate time ---
    auto_time = AUTO_PERIOD_TIME - (total_gate_time * auto_ratio)
    teleop_time = TELEOP_PERIOD_TIME - (total_gate_time * teleop_ratio)
    endgame_time = ENDGAME_PERIOD_TIME - (total_gate_time * endgame_ratio) - PARKING_TIME
    if endgame_time < 0:
        endgame_time = 0

    # --- Cycle counts ---
    auto_cycles = auto_time // cycle_time
    teleop_cycles = teleop_time // cycle_time
    endgame_cycles = endgame_time // cycle_time
    total_cycles = auto_cycles + teleop_cycles + endgame_cycles

    # --- Gate open distribution ---
    if total_cycles > 0:
        cycles_per_opening = total_cycles / max(gate_openings, 1)
    else:
        cycles_per_opening = 0

    # Split gate openings per period based on cycle ratios
    total_time_used = auto_time + teleop_time + endgame_time
    auto_gate_opens = gate_openings * (auto_time / total_time_used)
    teleop_gate_opens = gate_openings * (teleop_time / total_time_used)
    endgame_gate_opens = gate_openings * (endgame_time / total_time_used)

    # --- Scoring logic ---
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

    # --- Output Breakdown ---
    print("\nPERFORMANCE RESULTS")
    print("=" * 40)

    print("\n[Cycle Breakdown]")
    print(f"Autonomous Cycles : {int(auto_cycles)}")
    print(f"TeleOp Cycles     : {int(teleop_cycles)}")
    print(f"Endgame Cycles    : {int(endgame_cycles)} (after reserving {PARKING_TIME}s for parking)")
    print("---------------------------------")
    print(f"Total Possible Cycles: {int(total_cycles)}")

    print("\n[Gate Opening Breakdown]")
    print(f"Total Gate Openings: {gate_openings} (adds {total_gate_time:.1f}s total delay)")
    print(f"Autonomous Openings : {auto_gate_opens:.1f} ({(auto_gate_opens/gate_openings*100 if gate_openings else 0):.1f}%)")
    print(f"TeleOp Openings     : {teleop_gate_opens:.1f} ({(teleop_gate_opens/gate_openings*100 if gate_openings else 0):.1f}%)")
    print(f"Endgame Openings    : {endgame_gate_opens:.1f} ({(endgame_gate_opens/gate_openings*100 if gate_openings else 0):.1f}%)")

    print("\n[Scoring Breakdown]")
    print(f"Classified Artifacts Scored: {int(classified_cycles_scored * ARTIFACTS_PER_CYCLE)} ({int(classified_points)} points)")
    print(f"Overflow Artifacts Scored  : {int(overflow_cycles_scored * ARTIFACTS_PER_CYCLE)} ({int(overflow_points)} points)")
    print("---------------------------------")
    print(f"Total Estimated Points: {int(total_points)}")
    print("=" * 40)


# --- Run the program ---
if __name__ == "__main__":
    calculate_performance()
