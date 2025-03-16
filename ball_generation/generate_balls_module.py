import numpy as np
import random
import os

def generate_random_positions(num_positions, ball_radius, min_x, max_x, min_y, max_y, min_distance):
    """Generate non-overlapping random positions for the balls in a specified area
    
    Args:
        num_positions (int): Number of ball positions to generate
        ball_radius (float): Radius of each ball in meters
        min_x (float): Minimum x coordinate
        max_x (float): Maximum x coordinate
        min_y (float): Minimum y coordinate
        max_y (float): Maximum y coordinate
        min_distance (float): Minimum distance between ball centers
        
    Returns:
        list: List of [x, y] positions
    """
    positions = []
    max_attempts = 1000
    
    for _ in range(num_positions):
        attempts = 0
        valid_position = False
        
        while not valid_position and attempts < max_attempts:
            attempts += 1
            # Generate random position within the specified area
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
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

def generate_balls_xml(xml_path, output_path="model_with_balls.xml", num_balls=10, 
                       ball_size=0.03, ball_height=0.05, 
                       min_x=-0.5, max_x=0.5, min_y=-0.5, max_y=0.3):
    """Add white balls by modifying the XML and saving to a new file
    
    Args:
        xml_path (str): Path to the input XML file
        output_path (str): Path to save the modified XML file
        num_balls (int): Number of balls to generate
        ball_size (float): Radius of each ball in meters
        ball_height (float): Height of balls above table
        min_x (float): Minimum x coordinate for ball placement
        max_x (float): Maximum x coordinate for ball placement
        min_y (float): Minimum y coordinate for ball placement
        max_y (float): Maximum y coordinate for ball placement
        
    Returns:
        tuple: (output_path, num_balls_added)
    """
    # Minimum distance between ball centers (2.2 times the diameter to prevent overlap)
    min_distance = 2.2 * ball_size
    
    # Read the XML file
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    
    # Find the comment that indicates where the balls were removed
    balls_comment = "<!-- Balls removed from here -->"
    insert_index = xml_content.find(balls_comment)

    if insert_index == -1:
        # Fallback logic (unchanged)
        table_body_start = xml_content.find('<body name="table"')
        leg4_end = xml_content.find('</geom>', xml_content.find('name="leg4"'))
        camera_body_start = xml_content.find('<body name="camera_body"', table_body_start)
        
        if leg4_end == -1 or camera_body_start == -1:
            raise ValueError("Could not find proper insertion point for balls")
        
        insert_index = xml_content.find('>', leg4_end) + 1
    else:
        # Position right after the comment AND the line ending
        insert_index = xml_content.find('\n', insert_index) + 1
    
    # Generate random positions
    positions = generate_random_positions(num_balls, ball_size, min_x, max_x, min_y, max_y, min_distance)
    
    # Prepare the ball XML elements
    ball_elements = []
    for i, (pos_x, pos_y) in enumerate(positions):
        # Use white color for all balls (1,1,1,1)
        color_str = "1 1 1 1"
        
        # Create ball XML element - using material="ball_mat_white" which is defined in the XML
        ball_element = f'      <geom name="ball{i+1}" type="sphere" material="ball_mat_white" pos="{pos_x} {pos_y} {ball_height}" ' \
                      f'size="{ball_size}" rgba="{color_str}" contype="1" conaffinity="1" solimp="0.9 0.95 0.001" solref="0.01 1"/>'
        ball_elements.append(ball_element)
    
    # Insert balls at the identified position
    balls_xml = '\n' + '\n'.join(ball_elements) + '\n      '
    new_xml_content = xml_content[:insert_index] + balls_xml + xml_content[insert_index:]
    
    # Write the modified XML to a new file
    with open(output_path, 'w') as f:
        f.write(new_xml_content)
    
    return output_path, len(positions)

if __name__ == "__main__":
    # If the script is run directly, demonstrate usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MuJoCo scene XML with randomly placed balls')
    parser.add_argument('--input_xml', '-x', default='experiment_scen.xml',help='Path to the input XML file')
    parser.add_argument('--output', '-o', default='model_with_balls.xml', help='Path to save the modified XML file')
    parser.add_argument('--num-balls', '-n', type=int, default=9, help='Number of balls to generate')
    parser.add_argument('--ball-size', '-s', type=float, default=0.03, help='Radius of each ball in meters')
    parser.add_argument('--ball-height', '-z', type=float, default=0.05, help='Height of balls above table')
    parser.add_argument('--min-x', type=float, default=-0.4, help='Minimum x coordinate')
    parser.add_argument('--max-x', type=float, default=0.4, help='Maximum x coordinate')
    parser.add_argument('--min-y', type=float, default=-0.5, help='Minimum y coordinate')
    parser.add_argument('--max-y', type=float, default=0.1, help='Maximum y coordinate')
    
    args = parser.parse_args()
    
    output_path, num_balls_added = generate_balls_xml(
        args.input_xml, args.output, args.num_balls, 
        args.ball_size, args.ball_height,
        args.min_x, args.max_x, args.min_y, args.max_y
    )
    
    print(f"Created model with {num_balls_added} white balls at {output_path}")