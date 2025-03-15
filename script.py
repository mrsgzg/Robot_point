import mujoco
import numpy as np
import random
import mediapy as media
from pathlib import Path
import time

# Configuration
num_balls = 10
ball_size = 0.03  # Ball radius in meters
table_size = 0.9  # Table half-size (the usable area will be slightly smaller)
ball_height = 0.05  # Height of balls above table
margin = 0.1  # Margin from the table edge
xml_path = "panda_modified.xml"  # Path to the modified XML file

# Define ball colors (can be randomized too)
ball_colors = [
    [1.0, 0.0, 0.0, 1.0],  # Red
    [0.0, 1.0, 0.0, 1.0],  # Green
    [0.0, 0.0, 1.0, 1.0],  # Blue
    [1.0, 1.0, 0.0, 1.0],  # Yellow
    [1.0, 0.0, 1.0, 1.0],  # Magenta
    [0.0, 1.0, 1.0, 1.0],  # Cyan
    [1.0, 0.5, 0.0, 1.0],  # Orange
    [0.5, 0.0, 0.5, 1.0],  # Purple
    [0.0, 0.5, 0.0, 1.0],  # Dark Green
    [0.5, 0.5, 1.0, 1.0],  # Light Blue
]

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Function to generate non-overlapping random positions
def generate_random_positions(num_positions, ball_radius, table_half_size, min_distance, margin):
    positions = []
    max_attempts = 1000
    usable_area = table_half_size - margin
    
    for _ in range(num_positions):
        attempts = 0
        valid_position = False
        
        while not valid_position and attempts < max_attempts:
            attempts += 1
            # Generate random position within usable table area
            x = random.uniform(-usable_area, usable_area)
            y = random.uniform(-usable_area, usable_area)
            new_pos = [x, y]
            
            # Check if position is valid (doesn't overlap with existing balls)
            valid_position = True
            for pos in positions:
                distance = np.sqrt((new_pos[0] - pos[0])**2 + (new_pos[1] - pos[1])**2)
                if distance < min_distance:
                    valid_position = False
                    break
            
        if valid_position:
            positions.append(new_pos)
        else:
            print(f"Warning: Could not find valid position for ball {len(positions)+1}")
    
    return positions

# Add balls to the model
def add_balls_to_model(model, positions, ball_colors, ball_radius, table_height, ball_height):
    for i, (pos_x, pos_y) in enumerate(positions):
        color = ball_colors[i % len(ball_colors)]
        
        # Create a material for the ball
        material_name = f"ball_mat_{i}"
        model.add_material(
            material_name,
            texture=-1,
            texrepeat=[1, 1],
            emission=0,
            specular=0.5,
            shininess=0.3,
            reflectance=0,
            rgba=color
        )
        
        # Create a geom for the ball
        body_id = model.body("table").id
        model.add_geom(
            f"ball_{i}",
            body_id,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[ball_radius, 0, 0],
            pos=[