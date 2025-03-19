import subprocess
import os
from tqdm import tqdm

# Script to run
script_to_run = "updated-main.py"

# Base command
base_command = ["python", script_to_run]

# Run for each ball count from 5 down to 1
for num_balls in range(5, 0, -1):
    print(f"\n--- Starting set with {num_balls} balls ---\n")
    
    # Create command with current number of balls
    command = base_command + ["--num_balls", str(num_balls)]
    
    # Run the script 500 times with this ball count using tqdm for progress bar
    for i in tqdm(range(500), desc=f"Balls: {num_balls}", unit="run"):
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"\n--- Completed all runs with {num_balls} balls ---\n")

print("All runs completed successfully")