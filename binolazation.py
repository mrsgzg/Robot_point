import cv2
import numpy as np
import os
import argparse
import glob
import time

def process_image(img, lower_threshold=200, upper_threshold=255):
    """
    Process a single image to detect balls
    
    Args:
        img (numpy.ndarray): Input image
        lower_threshold (int): Lower threshold for white color (0-255)
        upper_threshold (int): Upper threshold for white color (0-255)
        
    Returns:
        tuple: (binary_image, detection_image)
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

def visualize_sequence(directory, pattern="*.png", lower_threshold=200, upper_threshold=255, 
                       delay=500, resize_factor=1.0, side_by_side=True):
    """
    Visualize a sequence of images with ball detection
    
    Args:
        directory (str): Directory containing the image sequence
        pattern (str): Glob pattern to match image files
        lower_threshold (int): Lower threshold for white color detection
        upper_threshold (int): Upper threshold for white color detection
        delay (int): Delay between frames in milliseconds
        resize_factor (float): Factor to resize images for display
        side_by_side (bool): Whether to show frames side by side
    """
    # Get all matching image files in the directory
    image_files = sorted(glob.glob(os.path.join(directory, pattern)))
    
    if not image_files:
        print(f"No images found in {directory} matching pattern {pattern}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Variables for tracking and visualization
    frame_count = 0
    ball_counts = []
    
    # Create a named window
    window_name = "Ball Detection Sequence"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Get first image dimensions for display setup
    first_img = cv2.imread(image_files[0])
    height, width = first_img.shape[:2]
    
    # Set the window size based on the display mode
    if side_by_side:
        display_width = int((width * 3) * resize_factor)  # 3 images side by side
        display_height = int(height * resize_factor)
    else:
        display_width = int(width * resize_factor)
        display_height = int(height * resize_factor)
    
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # Process each image in the sequence
    for i, image_file in enumerate(image_files):
        # Read image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Could not read image: {image_file}")
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
        
        # Create the display image based on the mode
        if side_by_side:
            # Concatenate the three images horizontally
            display_img = np.hstack((img_with_info, binary_img, detection_img))
        else:
            # Just show the detection image
            display_img = detection_img
            
            # Add more info in this case
            cv2.putText(display_img, frame_info, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
        key = cv2.waitKey(delay)
        
        # Check for exit keys (ESC or 'q')
        if key == 27 or key == ord('q'):
            break
        # Space bar to pause/resume
        elif key == 32:
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
        # Left/right arrows for navigation
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            # Go back one frame (if not at the beginning)
            if i > 0:
                i -= 2  # Will be incremented in the next loop
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            # Skip to next frame - no action needed as it will happen naturally
            pass
    
    # Show a final ball count summary
    print(f"Processed {frame_count} frames")
    if ball_counts:
        print(f"Average ball count: {sum(ball_counts) / len(ball_counts):.1f}")
        print(f"Min ball count: {min(ball_counts)}")
        print(f"Max ball count: {max(ball_counts)}")
    
    # Clean up
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Visualize a sequence of images with ball detection')
    parser.add_argument('directory', help='Directory containing the image sequence')
    parser.add_argument('--pattern', default='*.png', help='Glob pattern to match image files')
    parser.add_argument('--lower', '-l', type=int, default=200, 
                        help='Lower threshold for white color (0-255)')
    parser.add_argument('--upper', '-u', type=int, default=255,
                        help='Upper threshold for white color (0-255)')
    parser.add_argument('--delay', '-d', type=int, default=500,
                        help='Delay between frames in milliseconds (use 0 for step-by-step)')
    parser.add_argument('--resize', '-r', type=float, default=1.0,
                        help='Resize factor for display (e.g., 0.5 for half size)')
    parser.add_argument('--single', '-s', action='store_false', dest='side_by_side',
                        help='Show only detection image instead of side-by-side view')
    
    args = parser.parse_args()
    
    # Process and display the sequence
    visualize_sequence(
        args.directory,
        args.pattern,
        args.lower,
        args.upper,
        args.delay,
        args.resize,
        args.side_by_side
    )

if __name__ == "__main__":
    main()