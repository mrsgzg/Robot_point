import mujoco
import numpy as np
import time
import argparse
from ball_generation import generate_balls_module
from robot_controller import RobotController
from simulation_visualizer import SimulationVisualizer

def main():
    """
    Main function to generate a scene with balls and run the robot pointing simulation.
    Uses the table_cam as the main view for counting balls.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run robot ball counting simulation with camera tracking')
    parser.add_argument('--xml_path', default='experiment_scen.xml', help='Path to the base XML model file')
    parser.add_argument('--output_path', default='model_with_balls.xml', help='Path to save the modified XML file')
    parser.add_argument('--num_balls', type=int, default=15, help='Number of balls to generate')
    parser.add_argument('--ball_size', type=float, default=0.01, help='Size of the balls')
    parser.add_argument('--min_x', type=float, default=-0.1, help='Minimum x-coordinate for ball placement')
    parser.add_argument('--max_x', type=float, default=0.1, help='Maximum x-coordinate for ball placement')
    parser.add_argument('--min_y', type=float, default=-0.15, help='Minimum y-coordinate for ball placement')
    parser.add_argument('--max_y', type=float, default=-0.05, help='Maximum y-coordinate for ball placement')
    
    args = parser.parse_args()
    
    # Generate a scene with balls in well-positioned locations
    xml_path = args.xml_path
    output_path = args.output_path
    ball_size = args.ball_size
    table_surface_z = 0.025  # Height of table surface
    ball_height = table_surface_z + ball_size
    
    # Generate the XML with multiple balls in reachable positions
    output_path, num_balls_added = generate_balls_module.generate_balls_xml(
        xml_path=xml_path,
        output_path=output_path, 
        num_balls=args.num_balls,
        ball_size=ball_size, 
        ball_height=ball_height,
        # Place balls in a wider area in a row for left-to-right counting
        min_x=args.min_x, max_x=args.max_x,
        min_y=args.min_y, max_y=args.max_y
    )
    
    print(f"Created model with {num_balls_added} balls at {output_path}")
    
    # Load the generated model and create data
    try:
        model = mujoco.MjModel.from_xml_path(output_path)
        data = mujoco.MjData(model)
        print("Successfully loaded the model")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return
    
    # Initialize the robot controller
    controller = RobotController(model, data)
    
    # Initialize and start the visualization
    visualizer = SimulationVisualizer(model, data, controller)
    visualizer.visualize_pointing_sequence()

if __name__ == "__main__":
    main()