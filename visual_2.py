import cv2
import numpy as np
import os
import argparse
import glob
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def load_json_data(json_path):
    """
    Load joint position data from a JSON file
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        list: List of frame data dictionaries
    """
    try:
        with open(json_path, 'r') as f:
            # The JSON file contains a list of frame data
            # The content from paste.txt suggests it's not properly formatted
            # It might be missing the opening and closing brackets
            content = f.read()
            
            # Check if the content starts with a curly brace (object) instead of a bracket (array)
            if content.strip().startswith('{'):
                # Add array brackets if missing
                content = '[' + content + ']'
                
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try with some common fixes for malformed JSON
                # For example, the content might be missing commas between objects
                content = content.replace('}\n  {', '},\n  {')
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # If still failing, try another approach - read line by line
                    print("Warning: JSON file is malformed. Attempting to parse manually...")
                    lines = content.strip().split('\n')
                    # Manually reconstruct the JSON with proper formatting
                    fixed_content = '[\n'
                    for i, line in enumerate(lines):
                        fixed_content += line
                        if i < len(lines) - 1 and line.strip().endswith('}'):
                            fixed_content += ',\n'
                        else:
                            fixed_content += '\n'
                    fixed_content += ']\n'
                    data = json.loads(fixed_content)
        
        return data
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None

def create_joint_position_plot(frame_data, width=800, height=500):
    """
    Create a visualization of joint positions
    
    Args:
        frame_data (dict): Frame data containing joint positions
        width (int): Width of the output image
        height (int): Height of the output image
        
    Returns:
        numpy.ndarray: Image containing the joint position plot
    """
    # Create a figure with the desired size
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    canvas = FigureCanvas(fig)
    
    # Clear the axes for fresh plotting
    plt.clf()
    
    # Check if we have joint position data
    if 'joint_positions' not in frame_data:
        plt.text(0.5, 0.5, "No joint position data available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        
        # Convert the plot to an image
        canvas.draw()
        plot_image = np.array(canvas.renderer.buffer_rgba())
        
        # Convert RGBA to BGR (for OpenCV)
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)
        
        # Clean up
        plt.close(fig)
        
        return plot_image
    
    # Get joint positions
    joint_positions = frame_data['joint_positions']
    
    # Separate arm joints and other joints for better visualization
    arm_joints = {}
    other_joints = {}
    
    for key, value in joint_positions.items():
        if key.startswith('joint'):
            arm_joints[key] = value
        else:
            other_joints[key] = value
    
    # Sort keys for consistent display
    arm_joint_keys = sorted(arm_joints.keys())
    other_joint_keys = sorted(other_joints.keys())
    
    # Set up subplots in a single row
    plt.subplot(1, 2, 1)
    
    # Plot arm joint positions as a bar chart
    positions = []
    labels = []
    for key in arm_joint_keys:
        positions.append(arm_joints[key])
        labels.append(key)
    
    plt.bar(labels, positions, color='blue', alpha=0.7)
    plt.title('Arm Joint Positions', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_ylim([min(min(positions) - 0.2, -0.5), max(max(positions) + 0.2, 0.5)])
    
    # Plot other joint positions
    plt.subplot(1, 2, 2)
    
    positions = []
    labels = []
    for key in other_joint_keys:
        positions.append(other_joints[key])
        labels.append(key)
    
    plt.bar(labels, positions, color='green', alpha=0.7)
    plt.title('Other Joint Positions', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    if positions:
        plt.gca().set_ylim([min(min(positions) - 0.2, -0.5), max(max(positions) + 0.2, 0.5)])
    
    # Add additional info from the frame if available
    additional_info = []
    if 'motion_phase' in frame_data:
        additional_info.append(f"Phase: {frame_data['motion_phase']}")
    if 'count' in frame_data:
        additional_info.append(f"Count: {frame_data['count']}")
    if 'current_ball_index' in frame_data:
        additional_info.append(f"Ball Index: {frame_data['current_ball_index']}")
    
    # Add the info to the plot
    if additional_info:
        info_text = ' | '.join(additional_info)
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=9, 
                    bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make space for the info text
    
    # Convert the plot to an image
    canvas.draw()
    plot_image = np.array(canvas.renderer.buffer_rgba())
    
    # Convert RGBA to BGR (for OpenCV)
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)
    
    # Clean up
    plt.close(fig)
    
    return plot_image

def process_image(img, lower_threshold=200, upper_threshold=255):
    """
    Process a single image to detect balls
    
    Args:
        img (numpy.ndarray): Input image
        lower_threshold (int): Lower threshold for white color (0-255)
        upper_threshold (int): Upper threshold for white color (0-255)
        
    Returns:
        tuple: (binary_image, detection_image, ball_count, ball_centers)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to isolate white objects
    _, binary = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
    
    # Optional: Clean up the binary image using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create a colored binary image for display
    binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Create a copy for ball detection visualization
    detection_img = img.copy()
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count valid contours (balls)
    valid_count = 0
    ball_centers = []
    
    # Process each contour
    for contour in contours:
        # Filter small contours (noise)
        area = cv2.contourArea(contour)
        if area < 50:  # Adjust threshold as needed
            continue
            
        # Find the enclosing circle for each contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Only draw if radius is reasonable
        if radius > 5:
            # Draw the circle outline
            cv2.circle(detection_img, center, radius, (0, 255, 0), 2)
            
            # Draw the center point
            cv2.circle(detection_img, center, 3, (0, 0, 255), -1)
            
            # Label the ball
            cv2.putText(detection_img, f"#{valid_count+1}", (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            ball_centers.append(center)
            valid_count += 1
    
    # Add ball count to the image
    cv2.putText(detection_img, f"Balls: {valid_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return binary_colored, detection_img, valid_count, ball_centers

def extract_frame_number(filename):
    """
    Extract frame number from filename
    
    Args:
        filename (str): Filename like "frame_000001.png"
        
    Returns:
        int: Frame number
    """
    try:
        # Extract the numeric part from the filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        number_part = base_name.split('_')[-1]
        return int(number_part)
    except:
        # If we can't extract a number, return a large number to sort it last
        return 999999

def visualize_sequence(image_dir, json_path, image_pattern="frame_*.png", 
                        lower_threshold=200, upper_threshold=255, 
                        delay=500, resize_factor=1.0, display_mode="full"):
    """
    Visualize a sequence of images with ball detection and joint position data
    
    Args:
        image_dir (str): Directory containing the image sequence
        json_path (str): Path to the JSON file with joint data
        image_pattern (str): Glob pattern to match image files
        lower_threshold (int): Lower threshold for white color detection
        upper_threshold (int): Upper threshold for white color detection
        delay (int): Delay between frames in milliseconds
        resize_factor (float): Factor to resize images for display
        display_mode (str): Display mode ("full", "side_by_side", or "detection")
    """
    # Get all matching image files in the directory
    image_files = sorted(glob.glob(os.path.join(image_dir, image_pattern)), 
                        key=extract_frame_number)
    
    if not image_files:
        print(f"No images found in {image_dir} matching pattern {image_pattern}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Load the JSON data
    frame_data_list = load_json_data(json_path)
    if not frame_data_list:
        print(f"No valid data found in {json_path}")
        return
    
    print(f"Loaded {len(frame_data_list)} frames of joint data")
    
    # Variables for tracking and visualization
    frame_count = 0
    ball_counts = []
    current_speed = delay
    
    # Create a named window with a larger default size
    window_name = "Ball Detection with Joint Positions"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set a larger initial window size (1280x900)
    cv2.resizeWindow(window_name, 1280, 900)
    
    # Get first image dimensions for display setup
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Could not read first image: {image_files[0]}")
        return
        
    height, width = first_img.shape[:2]
    
    # Create a help text image
    help_img = np.zeros((100, 600, 3), dtype=np.uint8)
    cv2.putText(help_img, "Controls:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(help_img, "Space: Pause/Resume | +/-: Speed Up/Down | ESC/Q: Quit", 
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(help_img, "Left/Right Arrows or A/D: Navigate | V: Change View Mode", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Process each image in the sequence
    i = 0
    while i < len(image_files) and i < len(frame_data_list):
        image_file = image_files[i]
        frame_data = frame_data_list[i]
        
        # Read image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Could not read image: {image_file}")
            i += 1
            continue
        
        # Process the image
        binary_img, detection_img, ball_count, ball_centers = process_image(
            img, lower_threshold, upper_threshold)
        
        # Store ball count for this frame
        ball_counts.append(ball_count)
        
        # Display the current frame number and filename
        filename = os.path.basename(image_file)
        frame_info = f"Frame: {i+1}/{len(image_files)} - {filename}"
        
        # Add frame info to the original image
        img_with_info = img.copy()
        cv2.putText(img_with_info, frame_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Motion phase info if available
        if 'motion_phase' in frame_data:
            motion_phase = f"Phase: {frame_data['motion_phase']}"
            cv2.putText(img_with_info, motion_phase, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        
        # Create joint position plot (now side by side instead of stacked)
        joint_plot = create_joint_position_plot(frame_data, width, height=500)
        
        # Create the display image based on the display mode
        if display_mode == "full":
            # Arrange all views: original, binary, detection 
            top_row = np.hstack((img_with_info, binary_img, detection_img))
            # Use joint plot as second row - don't resize the height, keep it tall
            joint_plot_resized = cv2.resize(joint_plot, (top_row.shape[1], joint_plot.shape[0]))
            # Combine with help text
            help_resized = cv2.resize(help_img, (top_row.shape[1], help_img.shape[0]))
            display_img = np.vstack((top_row, joint_plot_resized, help_resized))
        elif display_mode == "side_by_side":
            # Show original and detection side by side with joint plot below
            top_row = np.hstack((img_with_info, detection_img))
            # Resize joint plot to match the width of top row
            joint_plot_resized = cv2.resize(joint_plot, (top_row.shape[1], joint_plot.shape[0]))
            # Combine with help text
            help_resized = cv2.resize(help_img, (top_row.shape[1], help_img.shape[0]))
            display_img = np.vstack((top_row, joint_plot_resized, help_resized))
        else:  # "detection" mode
            # Just show the detection image with joint plot below
            # Resize joint plot to match the width of detection image
            joint_plot_resized = cv2.resize(joint_plot, (detection_img.shape[1], joint_plot.shape[0]))
            # Add frame info to detection image
            cv2.putText(detection_img, frame_info, (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Combine with help text
            help_resized = cv2.resize(help_img, (detection_img.shape[1], help_img.shape[0]))
            display_img = np.vstack((detection_img, joint_plot_resized, help_resized))
        
        # Add playback speed indicator
        speed_text = f"Delay: {current_speed}ms"
        cv2.putText(display_img, speed_text, 
                    (display_img.shape[1] - 150, display_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Resize the display image if needed
        if resize_factor != 1.0:
            display_img = cv2.resize(
                display_img, 
                (int(display_img.shape[1] * resize_factor), 
                 int(display_img.shape[0] * resize_factor))
            )
        
        # Show the image
        cv2.imshow(window_name, display_img)
        
        # Update the frame count
        frame_count += 1
        
        # Wait for key press or the specified delay
        key = cv2.waitKey(current_speed)
        
        # Check for control keys
        if key == 27 or key == ord('q'):  # ESC or q - Exit
            break
        elif key == 32:  # Space - Pause/Resume
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
        elif key == ord('+') or key == ord('='):  # Speed up
            current_speed = max(10, current_speed - 50)
            print(f"Speed increased: {current_speed}ms delay")
        elif key == ord('-') or key == ord('_'):  # Slow down
            current_speed = min(2000, current_speed + 50)
            print(f"Speed decreased: {current_speed}ms delay")
        elif key == 81 or key == ord('a'):  # Left arrow or a - Previous frame
            i = max(0, i - 1)
            continue  # Skip the increment at the end
        elif key == 83 or key == ord('d'):  # Right arrow or d - Next frame
            # Will be incremented at the end
            pass
        elif key == ord('v'):  # Change view mode
            if display_mode == "full":
                display_mode = "side_by_side"
            elif display_mode == "side_by_side":
                display_mode = "detection"
            else:
                display_mode = "full"
            print(f"View mode changed to: {display_mode}")
        
        # Move to next frame
        i += 1
    
    # Show a final ball count summary
    print(f"Processed {frame_count} frames")
    if ball_counts:
        print(f"Average ball count: {sum(ball_counts) / len(ball_counts):.1f}")
        print(f"Min ball count: {min(ball_counts)}")
        print(f"Max ball count: {max(ball_counts)}")
    
    # Clean up
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Visualize ball detection with joint position data')
    parser.add_argument('image_dir', help='Directory containing the image sequence')
    parser.add_argument('json_path', help='Path to the JSON file with joint data')
    parser.add_argument('--image_pattern', default='frame_*.png', help='Glob pattern to match image files')
    parser.add_argument('--lower', '-l', type=int, default=200, 
                        help='Lower threshold for white color (0-255)')
    parser.add_argument('--upper', '-u', type=int, default=255,
                        help='Upper threshold for white color (0-255)')
    parser.add_argument('--delay', '-d', type=int, default=1,
                        help='Delay between frames in milliseconds (use 0 for step-by-step)')
    parser.add_argument('--resize', '-r', type=float, default=1.0,
                        help='Resize factor for display (e.g., 0.5 for half size)')
    parser.add_argument('--mode', '-m', choices=['full', 'side_by_side', 'detection'], default='full',
                        help='Display mode: full (all views), side_by_side (original+detection), or detection only')
    
    args = parser.parse_args()
    
    # Process and display the sequence
    visualize_sequence(
        args.image_dir,
        args.json_path,
        args.image_pattern,
        args.lower,
        args.upper,
        args.delay,
        args.resize,
        args.mode
    )

if __name__ == "__main__":
    main()