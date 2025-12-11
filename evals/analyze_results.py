#!/usr/bin/env python3
"""
Script to analyze simulation results from JSON log files.
Calculates average time to goal (excluding incomplete runs) and completion rate.
"""

import json
import sys
import numpy as np


def analyze_results(json_path):
    """
    Analyze simulation results from a JSON log file.
    
    Args:
        json_path: Path to the JSON log file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_runs = data.get('all_runs', [])
    
    if not all_runs:
        print("No runs found in the log file.")
        return
    
    # Collect time_to_goal and max_steps for each run
    time_to_goal_values = []
    for run in all_runs:
        max_steps = run['sim_params']['max_steps']
        time_to_goal = run['metrics']['robot']['time_to_goal']
        time_to_goal_values.append((time_to_goal, max_steps))
    
    # Filter completed runs (time_to_goal < max_steps)
    completed_times = [t for t, max_s in time_to_goal_values if t < max_s]
    
    # Calculate metrics
    total_runs = len(time_to_goal_values)
    completed_runs = len(completed_times)
    completion_rate = completed_runs / total_runs if total_runs > 0 else 0.0
    
    if completed_times:
        avg_time_to_goal = np.mean(completed_times)
    else:
        avg_time_to_goal = float('nan')
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Analysis of: {json_path}")
    print(f"{'='*50}")
    print(f"Total runs: {total_runs}")
    print(f"Completed runs: {completed_runs}")
    print(f"\nCompletion Rate: {completion_rate:.4f} ({completion_rate*100:.2f}%)")
    print(f"Average Time to Goal (completed only): {avg_time_to_goal:.2f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    analyze_results(json_path)
