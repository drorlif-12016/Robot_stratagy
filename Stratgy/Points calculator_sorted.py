# -*- coding: utf-8 -*-
"""
Robot Performance Calculator (with Gate Timing Breakdown)
---------------------------------------------------------
Calculates robot performance including how often the gate opens,
sorted cycles bonus, and total points.
"""

# --- Constants for Game Periods and Points ---
AUTO_PERIOD_TIME = 30
TELEOP_PERIOD_TIME = 100
ENDGAME_PERIOD_TIME = 20
PARKING_TIME = 3  # Reserve time for parking

# --- Scoring Values ---
CLASSIFIED_ARTIFACT_POINTS = 3
OVERFLOW_ARTIFACT_POINTS = 1
PATTERN_POINTS = 2  # 2 points per cycle
  # Extra 2 points per artifact for sorted cycles

# --- Robot/Game Config ---
ARTIFACTS_PER_CYCLE = 3
CYCLES_BEFORE_GATE_OPENS = 3
GATE_OPEN_DELAY = 2  # seconds per opening


def calculate_performance():
    print("Robot Performance Calculator")
    print("=" * 40)

    try:
        cycle_time_unsorted = float(input("Enter the robot's UNSORTED cycle time (in seconds): "))
        cycle_time_sorted = float(input("Enter the robot's SORTED cycle time (in seconds): "))
        sorted_cycles = int(input("Enter the number of SORTED cycles the robot completes: "))
        gate_openings = int(input("Enter the total number of gate openings expected: "))

        if cycle_time_unsorted <= 0 or cycle_time_sorted <= 0 or sorted_cycles < 0 or gate_openings < 0:
            print("\nError: Please enter valid positive values.")
            return
    except ValueError:
        print("\nError: Invalid input. Please enter numbers.")
        return

    # --- Time Setup ---
    total_game_time = AUTO_PERIOD_TIME + TELEOP_PERIOD_TIME + ENDGAME_PERIOD_TIME
    total_gate_time = gate_openings * GATE_OPEN_DELAY
    available_time = total_game_time - PARKING_TIME - total_gate_time

    # --- Time for Sorted Cycles ---
    time_for_sorted = sorted_cycles * cycle_time_sorted

    # If sorted cycles take more time than available, cap it
    if time_for_sorted > available_time:
        print("\nWarning: Not enough time for all sorted cycles. Reducing to fit available time.")
        sorted_cycles = int(available_time // cycle_time_sorted)
        time_for_sorted = sorted_cycles * cycle_time_sorted

    # --- Time left for Unsorted Cycles ---
    time_remaining = available_time - time_for_sorted
    unsorted_cycles = int(time_remaining // cycle_time_unsorted)
    total_cycles = sorted_cycles + unsorted_cycles

    # --- Time Ratios for Period Splitting ---
    total_time_used = time_for_sorted + unsorted_cycles * cycle_time_unsorted
    auto_ratio = AUTO_PERIOD_TIME / total_game_time
    teleop_ratio = TELEOP_PERIOD_TIME / total_game_time
    endgame_ratio = ENDGAME_PERIOD_TIME / total_game_time

    auto_time = AUTO_PERIOD_TIME - (total_gate_time * auto_ratio)
    teleop_time = TELEOP_PERIOD_TIME - (total_gate_time * teleop_ratio)
    endgame_time = ENDGAME_PERIOD_TIME - (total_gate_time * endgame_ratio) - PARKING_TIME
    if endgame_time < 0:
        endgame_time = 0

    time_for_cycle = lambda n_sorted: (n_sorted * cycle_time_sorted) + ((total_cycles - n_sorted) * cycle_time_unsorted)

    total_time_distributed = auto_time + teleop_time + endgame_time
    auto_cycles = total_cycles * (auto_time / total_time_distributed) if total_time_distributed > 0 else 0
    teleop_cycles = total_cycles * (teleop_time / total_time_distributed) if total_time_distributed > 0 else 0
    endgame_cycles = total_cycles * (endgame_time / total_time_distributed) if total_time_distributed > 0 else 0

    # --- Gate Opens per Period ---
    auto_gate_opens = gate_openings * (auto_time / total_time_distributed) if gate_openings else 0
    teleop_gate_opens = gate_openings * (teleop_time / total_time_distributed) if gate_openings else 0
    endgame_gate_opens = gate_openings * (endgame_time / total_time_distributed) if gate_openings else 0

    # --- Classification vs Overflow ---
    max_classified_cycles = gate_openings * CYCLES_BEFORE_GATE_OPENS
    if total_cycles <= max_classified_cycles:
        classified_cycles = total_cycles
        overflow_cycles = 0
    else:
        classified_cycles = max_classified_cycles
        overflow_cycles = total_cycles - max_classified_cycles

    # --- Sorted vs Unsorted classified breakdown ---
    sorted_classified = min(sorted_cycles, classified_cycles)
    unsorted_classified = classified_cycles - sorted_classified

    # --- Scoring ---
    classified_points = classified_cycles * ARTIFACTS_PER_CYCLE * CLASSIFIED_ARTIFACT_POINTS
    overflow_points = overflow_cycles * ARTIFACTS_PER_CYCLE * OVERFLOW_ARTIFACT_POINTS
    pattern_points = ARTIFACTS_PER_CYCLE *CLASSIFIED_ARTIFACT_POINTS + PATTERN_POINTS + PATTERN_POINTS + PATTERN_POINTS
    sorted_bonus_points = sorted_classified * ARTIFACTS_PER_CYCLE * PATTERN_POINTS

    total_points = classified_points + overflow_points +  sorted_bonus_points

    # --- Output ---
    print("\nPERFORMANCE RESULTS")
    print("=" * 40)

    print("\n[Cycle Breakdown]")
    print(f"Sorted Cycles      : {sorted_cycles}")
    print(f"Unsorted Cycles    : {unsorted_cycles}")
    print(f"Autonomous Cycles  : {int(auto_cycles)}")
    print(f"TeleOp Cycles      : {int(teleop_cycles)}")
    print(f"Endgame Cycles     : {int(endgame_cycles)} (after reserving {PARKING_TIME}s for parking)")
    print("---------------------------------")
    print(f"Total Cycles       : {total_cycles}")

    print("\n[Gate Opening Breakdown]")
    print(f"Total Gate Openings: {gate_openings} (adds {total_gate_time:.1f}s delay)")
    print(f"Autonomous Openings: {auto_gate_opens:.1f}")
    print(f"TeleOp Openings    : {teleop_gate_opens:.1f}")
    print(f"Endgame Openings   : {endgame_gate_opens:.1f}")

    print("\n[Scoring Breakdown]")
    print(f"Classified Artifacts Scored: {classified_cycles * ARTIFACTS_PER_CYCLE} ({classified_points} points)")
    print(f"  • of which were sorted   : {sorted_classified * ARTIFACTS_PER_CYCLE} (+{sorted_bonus_points} bonus points)")
    print(f"Overflow Artifacts Scored  : {overflow_cycles * ARTIFACTS_PER_CYCLE} ({overflow_points} points)")
    print(f"Pattern points overall     : {pattern_points} points (from {total_cycles} cycles)")
    print("---------------------------------")
    print(f"Total Estimated Points     : {int(total_points)}")
    print("=" * 40)


# --- Run the program ---
if __name__ == "__main__":
    calculate_performance()
