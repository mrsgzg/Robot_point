import mujoco
import numpy as np
import random
import time
import os

# Try to import glfw for visualization
try:
    import glfw
    has_glfw = True
except ImportError:
    print("GLFW not available. Install with: pip install glfw")
    has_glfw = False

# Configuration
num_balls = 10
ball_size = 0.03  # Ball radius in meters
ball_height = 0.05  # Height of balls above table

# Position the balls within reach of the robot arm
# The robot is at position (0, -0.9, 0), so we'll position balls closer to this area
min_x = -0.5
max_x = 0.5
min_y = -0.5  # Closer to the robot
max_y = 0.3
min_distance = 2.2 * ball_size  # Minimum distance between ball centers

xml_path = "experiment_scen.xml"  # Path to your provided XML file

# Joint control settings
joint_step = 0.05  # Radians to move per keypress
selected_joint = 0  # Currently selected joint
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "camera_pan"]
joint_indices = []  # Will be filled when model is loaded

def generate_random_positions(num_positions, ball_radius, min_distance):
    """Generate non-overlapping random positions for the balls in a more accessible area"""
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

def add_balls_to_model_xml(xml_path, output_path="model_with_balls.xml"):
    """Add white balls by modifying the XML and saving to a new file"""
    # Read the XML file
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    
    # Find the comment that indicates where the balls were removed
    balls_comment = "<!-- Balls removed from here -->"
    insert_index = xml_content.find(balls_comment)
    
    if insert_index == -1:
        # Fallback: Find the table section and the camera body to locate where to insert balls
        table_body_start = xml_content.find('<body name="table"')
        leg4_end = xml_content.find('</geom>', xml_content.find('name="leg4"'))
        camera_body_start = xml_content.find('<body name="camera_body"', table_body_start)
        
        if leg4_end == -1 or camera_body_start == -1:
            raise ValueError("Could not find proper insertion point for balls")
        
        insert_index = xml_content.find('>', leg4_end) + 1
    else:
        # Position right after the comment
        insert_index += len(balls_comment)
    
    # Generate random positions
    positions = generate_random_positions(num_balls, ball_size, min_distance)
    
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

def find_joint_indices(model):
    """Find the indices of all the joints we want to control"""
    global joint_indices
    joint_indices = []
    
    for joint_name in joint_names:
        # Try different methods to find joint IDs based on MuJoCo version
        try:
            # First try mj_name2id
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            except AttributeError:
                # If that doesn't work, try direct access
                joint_id = -1
                for i in range(model.njnt):
                    try:
                        if model.joint_name(i) == joint_name:
                            joint_id = i
                            break
                    except:
                        pass
        except:
            joint_id = -1
            
        joint_indices.append(joint_id)
    
    # Remove any joints not found (-1)
    valid_joint_names = []
    valid_joint_indices = []
    
    for i, joint_id in enumerate(joint_indices):
        if joint_id >= 0:
            valid_joint_names.append(joint_names[i])
            valid_joint_indices.append(joint_id)
    
    joint_names.clear()
    joint_names.extend(valid_joint_names)
    joint_indices.clear()
    joint_indices.extend(valid_joint_indices)
    
    print(f"Found {len(joint_indices)} controllable joints:")
    for i, (name, idx) in enumerate(zip(joint_names, joint_indices)):
        print(f"  {i+1}: {name} (ID: {idx})")
    
    return len(joint_indices) > 0

# Keyboard callback to handle joint control and camera switching
def keyboard_callback(window, key, scancode, action, mods):
    global current_camera, camera_count, selected_joint
    
    if action == glfw.PRESS or action == glfw.REPEAT:
        # Camera controls
        if key == glfw.KEY_SPACE:
            # Switch camera on space press
            current_camera = (current_camera + 1) % (camera_count + 1)  # +1 for free camera
            print(f"Switched to camera {current_camera}")
        
        # Joint selection
        elif key == glfw.KEY_TAB:
            # Select next joint
            selected_joint = (selected_joint + 1) % len(joint_indices)
            print(f"Selected joint: {joint_names[selected_joint]}")
        
        # Joint controls
        elif key == glfw.KEY_UP or key == glfw.KEY_W:
            # Increase joint angle
            if 0 <= selected_joint < len(joint_indices):
                joint_id = joint_indices[selected_joint]
                data.qpos[joint_id] += joint_step
                update_joint_control(joint_id, data.qpos[joint_id])
                print(f"Joint {joint_names[selected_joint]} position: {data.qpos[joint_id]:.4f}")
        
        elif key == glfw.KEY_DOWN or key == glfw.KEY_S:
            # Decrease joint angle
            if 0 <= selected_joint < len(joint_indices):
                joint_id = joint_indices[selected_joint]
                data.qpos[joint_id] -= joint_step
                update_joint_control(joint_id, data.qpos[joint_id])
                print(f"Joint {joint_names[selected_joint]} position: {data.qpos[joint_id]:.4f}")
        
        # Exit control
        elif key == glfw.KEY_ESCAPE:
            # Close window on escape
            glfw.set_window_should_close(window, True)

def update_joint_control(joint_id, position):
    """Update control signals to match the joint position to prevent spring-back"""
    # Find all actuators that control this joint
    for i in range(model.nu):
        try:
            actuator_joint = model.actuator_trnid[i, 0]  # Get the joint this actuator controls
            if actuator_joint == joint_id:
                # Update control signal to match the position
                data.ctrl[i] = position
        except:
            # If actuator_trnid access fails, try a more general approach
            # Zero out all controls to disable actuators
            data.ctrl[i] = 0

def run_simulation_with_joint_control():
    """Run the simulation with a visible window using GLFW with joint control"""
    global current_camera, camera_count, data, model
    
    if not has_glfw:
        print("GLFW not available. Joint control requires GLFW.")
        return False
    
    # Create a temporary XML file with balls
    temp_xml_path, num_balls_added = add_balls_to_model_xml(xml_path)
    print(f"Created temporary model with {num_balls_added} white balls positioned closer to the robot")
    
    try:
        # Initialize GLFW
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
        
        # Create a window
        window_width, window_height = 1200, 900
        window = glfw.create_window(window_width, window_height, "MuJoCo Simulation with Joint Control", None, None)
        if not window:
            glfw.terminate()
            print("Failed to create GLFW window")
            return False
        
        # Make the window's context current
        glfw.make_context_current(window)
        
        # Load the model
        model = mujoco.MjModel.from_xml_path(temp_xml_path)
        data = mujoco.MjData(model)
        
        # Find joint indices
        if not find_joint_indices(model):
            print("Warning: No controllable joints found")
        
        # Set keyboard callback
        glfw.set_key_callback(window, keyboard_callback)
        
        # Initialize the cameras
        camera = mujoco.MjvCamera()
        
        # Get camera count from model
        camera_count = model.ncam
        print(f"Model has {camera_count} camera(s)")
        
        # Set current camera to free camera (not one of the model cameras)
        current_camera = 0  # Start with free camera
        
        # Initialize the free camera view
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 4.0
        camera.elevation = -30.0
        camera.azimuth = 90.0
        
        # Initialize the perturbation object
        perturb = mujoco.MjvPerturb()
        
        # Initialize the visualization options
        opt = mujoco.MjvOption()
        
        # Create scene and context
        scene = mujoco.MjvScene(model, maxgeom=1000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Set the window size
        glfw.set_window_size_callback(window, lambda win, width, height: mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context))
        
        # Initialize hint text
        font_scale = mujoco.mjtFontScale.mjFONTSCALE_150
        help_text = "SPACE: switch camera, TAB: select joint, UP/DOWN: move joint, ESC: quit"
        
        # Reset data to a home position
        mujoco.mj_resetData(model, data)
        
        # Disable actuators initially to allow free positioning
        for i in range(model.nu):
            data.ctrl[i] = 0
        
        # Run the simulation while the window is open
        while not glfw.window_should_close(window):
            # Step simulation (but less frequently when doing manual control)
            mujoco.mj_step1(model, data)  # Only do position update
            
            # Update camera based on current_camera setting
            if current_camera == 0:
                # Free camera (already set)
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                # Use model camera (1-based indexing for model cameras)
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = current_camera - 1
            
            # Get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            # Update scene and render
            mujoco.mjv_updateScene(model, data, opt, perturb, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            
            # Add camera and joint information overlay
            if current_camera == 0:
                camera_name = "Free Camera"
            else:
                try:
                    camera_name = f"Camera {current_camera}: {model.camera_name(current_camera-1)}"
                except:
                    camera_name = f"Camera {current_camera}"
            
            joint_info = ""
            if len(joint_indices) > 0:
                joint_name = joint_names[selected_joint]
                joint_pos = data.qpos[joint_indices[selected_joint]]
                joint_info = f"Joint: {joint_name} ({joint_pos:.4f})"
            
            overlay_text = f"{camera_name}\n{joint_info}\n{help_text}"
            mujoco.mjr_overlay(font_scale, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, overlay_text, "", context)
            
            # Swap front and back buffers
            glfw.swap_buffers(window)
            
            # Poll for and process events
            glfw.poll_events()
            
            # Complete the simulation step (dynamics)
            mujoco.mj_step2(model, data)
        
        # Clean up
        glfw.terminate()
        
        # Delete the temporary XML
        try:
            #os.remove(temp_xml_path)
            print(f"Removed temporary file: {temp_xml_path}")
        except:
            print(f"Note: Could not remove temporary file: {temp_xml_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        return False

# Global variables 
current_camera = 0  # Start with free camera
camera_count = 0    # Will be set based on model
data = None         # Will hold the simulation data
model = None        # Will hold the MuJoCo model

if __name__ == "__main__":
    try:
        success = run_simulation_with_joint_control()
        if success:
            print("Successfully ran simulation with joint control")
        else:
            print("Simulation failed")
    except Exception as e:
        print(f"Overall error: {e}")