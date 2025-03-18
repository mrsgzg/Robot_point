import subprocess
import os

# Script to run
script_to_run = "updated-main.py"

# Arguments to pass to the script
# Add any command line arguments here
args = ["--collect-data", "--auto-record"]

# Create base command
command = ["python", script_to_run] 

# Run the script 10 times
for i in range(10):
    print(f"Starting run {i+1}/10")
    subprocess.run(command)
    print(f"Completed run {i+1}/10")