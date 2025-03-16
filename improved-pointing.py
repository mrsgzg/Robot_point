import mujoco
import numpy as np
import glfw
import time
from ball_generation import generate_balls_module

# Configuration
POINT_OFFSET_DISTANCE = 0.25  # Height above the ball when pointing down
XML_PATH = "model_with_balls.xml"  # Path to the generated XML file
PAUSE_TIME = 1.0  # Seconds to pause at each ball when counting

class ImprovedBallPointingTest:
    def __init__(self, xml_path):
        """Initialize MuJoCo simulation with improved ball-pointing capabilities"""
        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Camera state
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 2.0
        self.camera.elevation = -20
        self.camera.azimuth = 90
        self.camera.lookat = np.array([0.0, -0.5, 0.1])
        
        # Mouse interaction state
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Window reference
        self.window = None
        
        # Prepare joint control
        self.joint_names = [
            "joint1", "joint2", "joint3", 
            "joint4", "joint5", "joint6", "joint7"
        ]
        
        # Get joint indices
        self.joint_indices = []
        for name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    self.joint_indices.append(joint_id)
                else:
                    print(f"Warning: Joint {name} not found in model")
            except Exception as e:
                print(f"Error finding joint {name}: {e}")
        
        # Find finger joints for the pointing gesture
        self.finger_joint_names = ["finger_joint1", "finger_joint2"]
        self.finger_joint_indices = []
        for name in self.finger_joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    self.finger_joint_indices.append(joint_id)
                else:
                    print(f"Warning: Finger joint {name} not found in model")
            except Exception as e:
                print(f"Error finding finger joint {name}: {e}")
        
        # Find hand body and site
        try:
            self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
            self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
            print(f"Found hand body at id {self.hand_body_id}")
        except Exception as e:
            print(f"Error finding hand body: {e}")
            self.hand_body_id = -1
        
        # Find balls and sort them from left to right
        self.ball_geoms = self._find_and_sort_balls()
        
        # Pointing state
        self.initial_joint_positions = None
        self.target_joint_positions = None
        self.interpolation_progress = 0.0
        
        # Counting state
        self.current_count = 0
        self.counted_balls = []
    
    def _find_and_sort_balls(self):
        """Find all ball geoms in the simulation and sort them from left to right"""
        ball_geoms = []
        ball_positions = []
        
        for i in range(self.model.ngeom):
            try:
                # Try to get the geom name and type
                geom_name = self.model.geom(i).name
                geom_type = self.model.geom(i).type
                
                # Check if it's a ball (sphere)
                if 'ball' in geom_name and geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    pos = self.model.geom(i).pos.copy()
                    ball_geoms.append(i)
                    ball_positions.append((i, pos))
                    print(f"Found ball {geom_name} at position {pos}")
            except Exception as e:
                print(f"Error processing geom {i}: {e}")
        
        # Sort balls by x position (left to right)
        ball_positions.sort(key=lambda x: x[1][0])
        sorted_ball_geoms = [ball_id for ball_id, _ in ball_positions]
        
        print(f"Found and sorted {len(sorted_ball_geoms)} ball geoms from left to right")
        return sorted_ball_geoms
    
    def setup_visualization(self):
        """Set up GLFW window with interactive camera controls"""
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1200, 900, "Improved Ball Pointing Test", None, None)
        if not self.window:
            glfw.terminate()
            print("Failed to create GLFW window")
            return False
        
        # Make the window's context current
        glfw.make_context_current(self.window)
        
        # Set callbacks
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        
        return True
    
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button interactions"""
        x, y = glfw.get_cursor_pos(window)
        
        # Check if the window is valid
        if not window:
            return
        
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.button_left = True
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                self.button_middle = True
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.button_right = True
        
        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.button_left = False
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                self.button_middle = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.button_right = False
        
        self.last_mouse_x = x
        self.last_mouse_y = y
    
    def mouse_move_callback(self, window, xpos, ypos):
        """Handle camera movement based on mouse interaction"""
        # Prevent movement if no buttons are pressed
        if not (self.button_left or self.button_middle or self.button_right):
            return
        
        # Calculate mouse movement
        dx = xpos - self.last_mouse_x
        dy = ypos - self.last_mouse_y
        
        # Left button: rotate camera
        if self.button_left:
            self.camera.azimuth += dx * 0.3
            self.camera.elevation -= dy * 0.3
            
            # Clamp elevation
            self.camera.elevation = max(-90, min(90, self.camera.elevation))
        
        # Middle button: translate camera
        elif self.button_middle:
            # Compute scale based on camera distance
            scale = self.camera.distance * 0.001
            
            # Update camera lookat point
            self.camera.lookat[0] -= dx * scale
            self.camera.lookat[1] += dy * scale
        
        # Right button: zoom
        elif self.button_right:
            # Adjust zoom
            zoom_factor = 1 - dy * 0.01
            self.camera.distance *= zoom_factor
            
            # Prevent extreme zooming
            self.camera.distance = max(0.1, min(self.camera.distance, 100))
        
        # Update last mouse position
        self.last_mouse_x = xpos
        self.last_mouse_y = ypos
    
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll for zooming"""
        # Zoom in/out
        zoom_factor = 1 - yoffset * 0.1
        self.camera.distance *= zoom_factor
        
        # Prevent extreme zooming
        self.camera.distance = max(0.1, min(self.camera.distance, 100))
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle key press events"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    
    def calculate_pointing_position(self, ball_pos):
        """
        Calculate a position above and slightly offset from the ball
        
        Args:
            ball_pos (np.array): Ball position [x, y, z]
            
        Returns:
            position for the hand
        """
        # Position the hand above and slightly offset from the ball
        # This offset helps create a better angle for pointing
        pointing_pos = ball_pos.copy()
        pointing_pos[2] += POINT_OFFSET_DISTANCE  # Move above the ball
        
        # Add a small offset in y direction toward the robot base
        # This helps create a better pointing angle
        pointing_pos[1] -= 0.05
        
        return pointing_pos
    
    def calculate_improved_ik(self, ball_pos):
        """
        Use MuJoCo's IK solver to position the hand in a natural pointing pose
        
        Args:
            ball_pos (np.array): Ball position [x, y, z]
            
        Returns:
            dict: Joint angles for arm and fingers
        """
        # Calculate the pointing position
        pointing_pos = self.calculate_pointing_position(ball_pos)
        
        # Reset to a reasonable starting position
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose (slightly raised arm)
        for i, joint_idx in enumerate(self.joint_indices):
            if i == 1:  # Joint 2 (shoulder)
                self.data.qpos[joint_idx] = 0.3  # Raised a bit
            elif i == 3:  # Joint 4 (elbow)
                self.data.qpos[joint_idx] = -0.5  # Bent slightly
        
        # Forward kinematics to apply the initial pose
        mujoco.mj_forward(self.model, self.data)
        
        # Create data for IK solver
        ik_data = mujoco.MjData(self.model)
        
        # Manually copy relevant state data
        ik_data.qpos[:] = self.data.qpos[:]
        ik_data.qvel[:] = self.data.qvel[:]
        ik_data.act[:] = self.data.act[:]
        ik_data.ctrl[:] = self.data.ctrl[:]
        
        # IK parameters
        self.model.opt.iterations = 1000  # Increase maximum iterations for IK solver
        max_ik_iterations = 500
        tolerance = 0.001  # Positional tolerance (1 mm)
        
        # Setup target
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        
        # Create workspace for IK corrections
        qpos = np.zeros(self.model.nq)
        
        # Run IK with a focus on position only (no orientation constraint)
        for iteration in range(max_ik_iterations):
            # Copy current joint positions
            qpos = ik_data.qpos.copy()
            
            # Get current hand position
            mujoco.mj_forward(self.model, ik_data)
            curr_pos = ik_data.xpos[self.hand_body_id].copy()
            
            # Calculate error
            err_pos = pointing_pos - curr_pos
            err_norm = np.linalg.norm(err_pos)
            
            # Check for convergence
            if err_norm < tolerance:
                print(f"IK converged after {iteration+1} iterations. Error: {err_norm:.6f}")
                break
            
            # Calculate Jacobian for hand body
            mujoco.mj_jacBody(self.model, ik_data, jac_pos, jac_rot, self.hand_body_id)
            
            # Use only position Jacobian columns for the joints we care about
            jac_limited = np.zeros((3, len(self.joint_indices)))
            for i, joint_idx in enumerate(self.joint_indices):
                dof_idx = self.model.jnt_dofadr[joint_idx]
                jac_limited[:, i] = jac_pos[:, dof_idx]
            
            # Compute pseudoinverse
            try:
                jac_pinv = np.linalg.pinv(jac_limited)
                
                # Small step size for stability
                alpha = 0.1
                
                # Compute joint corrections
                dq = alpha * jac_pinv @ err_pos
                
                # Apply corrections to joint positions
                for i, joint_idx in enumerate(self.joint_indices):
                    qpos[joint_idx] += dq[i]
                    
                    # Apply joint limits
                    try:
                        jnt_range = self.model.jnt_range[joint_idx]
                        lower_limit = jnt_range[0]
                        upper_limit = jnt_range[1]
                        
                        if lower_limit < upper_limit:  # Valid range
                            qpos[joint_idx] = max(lower_limit, min(qpos[joint_idx], upper_limit))
                    except Exception as e:
                        pass
            
            except Exception as e:
                print(f"Error in IK calculation: {e}")
                break
            
            # Update positions
            ik_data.qpos[:] = qpos
            
            # Every 50 iterations, print progress
            if iteration % 50 == 0:
                print(f"IK iteration {iteration}, error: {err_norm:.6f}")
        
        # Final forward kinematics
        mujoco.mj_forward(self.model, ik_data)
        
        # Get final hand position
        final_pos = ik_data.xpos[self.hand_body_id].copy()
        final_error = np.linalg.norm(pointing_pos - final_pos)
        print(f"Final IK error: {final_error:.6f}")
        
        # Dynamic calculation of joint7 angle to point at the ball
        # Get current hand position after IK
        mujoco.mj_forward(self.model, ik_data)
        hand_pos = ik_data.xpos[self.hand_body_id].copy()
        
        # Calculate vector from hand to ball
        hand_to_ball = ball_pos - hand_pos
        
        # Calculate the angle in the y-z plane that points to the ball
        # This gives us the rotation angle for joint7
        angle_yz = np.arctan2(hand_to_ball[2], -hand_to_ball[1])
        
        # Apply the calculated angle to joint7
        joint7_idx = self.joint_indices[6]  # joint7 (last joint)
        ik_data.qpos[joint7_idx] = angle_yz

        joint6_idx = self.joint_indices[5]
        ik_data.qpos[joint6_idx] = ik_data.qpos[joint6_idx] +0.35

        #print(angle_yz)
        
        # Run forward kinematics again to update hand position with new joint7 angle
        mujoco.mj_forward(self.model, ik_data)
        
        # Keep gripper fully closed - no finger movement
        for i, joint_idx in enumerate(self.finger_joint_indices):
            ik_data.qpos[joint_idx] = 0.0  # Fully closed position
        
        # Copy final joint positions
        result = {
            'arm_joints': np.zeros(len(self.joint_indices)),
            'finger_joints': np.zeros(len(self.finger_joint_indices))
        }
        
        for i, joint_idx in enumerate(self.joint_indices):
            result['arm_joints'][i] = ik_data.qpos[joint_idx]
            
        for i, joint_idx in enumerate(self.finger_joint_indices):
            result['finger_joints'][i] = ik_data.qpos[joint_idx]
        
        # Final check
        print(f"Target ball position: {ball_pos}")
        print(f"Pointing position: {pointing_pos}")
        print(f"Final joint angles: {result['arm_joints']}")
        
        return result
    
    def smooth_interpolation(self, t):
        """
        Create a more human-like motion profile using a smooth ease-in/ease-out curve
        
        Args:
            t (float): Linear interpolation parameter [0, 1]
            
        Returns:
            float: Smoothed interpolation parameter
        """
        # Cubic ease-in/ease-out function
        return 3*t**2 - 2*t**3
    
    def point_at_balls_in_sequence(self):
        """Position the gripper to point at balls in sequence from left to right"""
        # Setup visualization
        if not self.setup_visualization():
            return
        
        # Check if we have balls
        if not self.ball_geoms:
            print("No balls found in the model")
            glfw.terminate()
            return
        
        # Set up visualization
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Options and perturbation
        opt = mujoco.MjvOption()
        perturb = mujoco.MjvPerturb()
        
        # Reset the simulation and remember initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # Initial pose - slightly raised arm
        for i, joint_idx in enumerate(self.joint_indices):
            if i == 1:  # Joint 2 (shoulder)
                self.data.qpos[joint_idx] = 0.3  # Raised a bit
            elif i == 3:  # Joint 4 (elbow)
                self.data.qpos[joint_idx] = -0.5  # Bent slightly
        
        # Apply initial pose
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial joint positions
        initial_arm_joints = np.zeros(len(self.joint_indices))
        initial_finger_joints = np.zeros(len(self.finger_joint_indices))
        
        for i, joint_idx in enumerate(self.joint_indices):
            initial_arm_joints[i] = self.data.qpos[joint_idx]
            
        for i, joint_idx in enumerate(self.finger_joint_indices):
            initial_finger_joints[i] = self.data.qpos[joint_idx]
        
        # Timing variables
        last_time = time.time()
        pause_until = 0
        
        # Cycle through all balls in left-to-right order
        current_ball_index = 0
        motion_phase = "moving"  # "moving" or "paused"
        frame = 0
        interpolation_steps = 120  # More steps for smoother motion
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            
            # Handle pause timing for counting
            if motion_phase == "paused":
                if current_time >= pause_until:
                    # Move to next ball
                    current_ball_index += 1
                    if current_ball_index >= len(self.ball_geoms):
                        # Finished all balls, reset or exit
                        break
                    motion_phase = "moving"
                    frame = 0
                    
                    # Update initial joint positions
                    for i, joint_idx in enumerate(self.joint_indices):
                        initial_arm_joints[i] = self.data.qpos[joint_idx]
                    
                    for i, joint_idx in enumerate(self.finger_joint_indices):
                        initial_finger_joints[i] = self.data.qpos[joint_idx]
            
            # If in moving phase, calculate new positions
            if motion_phase == "moving":
                # Get the current ball
                ball_id = self.ball_geoms[current_ball_index]
                ball_pos = self.model.geom(ball_id).pos.copy()
                
                # Calculate joint angles for pointing at this ball (if not already calculated)
                if frame == 0:
                    joint_angles = self.calculate_improved_ik(ball_pos)
                    target_arm_joints = joint_angles['arm_joints']
                    target_finger_joints = joint_angles['finger_joints']
                
                # Calculate interpolation factor
                if frame < interpolation_steps:
                    # Linear interpolation parameter
                    t = frame / interpolation_steps
                    # Apply smooth interpolation function
                    smooth_t = self.smooth_interpolation(t)
                else:
                    smooth_t = 1.0
                    # Transition to paused state
                    motion_phase = "paused"
                    pause_until = current_time + PAUSE_TIME
                    self.counted_balls.append(current_ball_index)
                
                # Interpolate joint positions
                for i, joint_idx in enumerate(self.joint_indices):
                    if i < len(target_arm_joints):
                        # Calculate interpolated position
                        interp_pos = initial_arm_joints[i] + smooth_t * (target_arm_joints[i] - initial_arm_joints[i])
                        
                        # Apply joint limits if available
                        try:
                            jnt_range = self.model.jnt_range[joint_idx]
                            lower_limit = jnt_range[0]
                            upper_limit = jnt_range[1]
                            
                            if lower_limit < upper_limit:  # Valid range
                                interp_pos = max(lower_limit, min(interp_pos, upper_limit))
                        except:
                            pass
                        
                        # Set joint position
                        self.data.qpos[joint_idx] = interp_pos
                        
                        # Also set control to maintain position
                        for j in range(self.model.nu):
                            try:
                                if self.model.actuator_trnid[j, 0] == joint_idx:
                                    self.data.ctrl[j] = interp_pos
                            except:
                                pass
                
                # Interpolate finger positions for pointing gesture
                for i, joint_idx in enumerate(self.finger_joint_indices):
                    if i < len(target_finger_joints):
                        interp_pos = initial_finger_joints[i] + smooth_t * (target_finger_joints[i] - initial_finger_joints[i])
                        self.data.qpos[joint_idx] = interp_pos
                        
                        # Set control if needed
                        for j in range(self.model.nu):
                            try:
                                if self.model.actuator_trnid[j, 0] == joint_idx:
                                    self.data.ctrl[j] = interp_pos
                            except:
                                pass
                
                # Increment frame counter
                frame += 1
            
            # Forward kinematics
            try:
                mujoco.mj_forward(self.model, self.data)
            except Exception as e:
                print(f"Error in mj_forward: {e}")
            
            try:
                # Update scene
                mujoco.mjv_updateScene(self.model, self.data, opt, perturb, self.camera, 
                                     mujoco.mjtCatBit.mjCAT_ALL, scene)
                
                # Get viewport
                viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
                
                # Add visualization for all balls with different colors based on state
                for idx, ball_id in enumerate(self.ball_geoms):
                    ball_pos = self.model.geom(ball_id).pos.copy()
                    
                    # Add a number above each ball
                    number_pos = ball_pos.copy()
                    number_pos[2] += 0.05  # Slightly above the ball
                    
                    # Check if this is the current ball
                    is_current = (idx == current_ball_index)
                    
                    # Check if this ball has been counted
                    is_counted = idx in self.counted_balls
                    
                    if scene.ngeom + 1 <= scene.maxgeom:
                        # Mark the ball position with appropriate color
                        g = scene.geoms[scene.ngeom]
                        scene.ngeom += 1
                        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                        g.size[:] = [0.02, 0, 0]  # Small sphere
                        g.pos[:] = ball_pos
                        
                        if is_current:
                            # Current ball being pointed at - green highlight
                            g.rgba[:] = [0, 1, 0, 0.5]
                        elif is_counted:
                            # Already counted - blue
                            g.rgba[:] = [0, 0, 1, 0.5]
                        else:
                            # Not yet counted - white
                            g.rgba[:] = [1, 1, 1, 0.5]
                
                # Add visualization for pointing line (from finger to ball)
                if current_ball_index < len(self.ball_geoms):
                    ball_id = self.ball_geoms[current_ball_index]
                    ball_pos = self.model.geom(ball_id).pos.copy()
                    
                    # Get right finger position (index finger approximation)
                    finger_pos = self.data.xpos[self.right_finger_body_id].copy()
                    
                    if scene.ngeom + 1 <= scene.maxgeom:
                        # Draw point at hand location
                        g = scene.geoms[scene.ngeom]
                        scene.ngeom += 1
                        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                        g.size[:] = [0.01, 0, 0]  # Small sphere
                        g.pos[:] = finger_pos
                        g.rgba[:] = [1, 0, 0, 1]  # Red

                        # Draw point at ball location
                        if scene.ngeom + 1 <= scene.maxgeom:
                            g = scene.geoms[scene.ngeom]
                            scene.ngeom += 1
                            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                            g.size[:] = [0.01, 0, 0]  # Small sphere
                            g.pos[:] = ball_pos
                            g.rgba[:] = [1, 0, 0, 1]  # Red
                
                # Render scene
                mujoco.mjr_render(viewport, scene, context)
                
                # Add status text
                if motion_phase == "moving":
                    if frame < interpolation_steps:
                        progress_text = f"Counting: Ball {current_ball_index+1}/{len(self.ball_geoms)} - Moving: {(frame/interpolation_steps)*100:.1f}%"
                    else:
                        progress_text = f"Counting: Ball {current_ball_index+1}/{len(self.ball_geoms)} - Arrived"
                else:  # paused
                    countdown = pause_until - current_time
                    progress_text = f"Counting: Ball {current_ball_index+1}/{len(self.ball_geoms)} - Counting: {countdown:.1f}s"
                
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                                 mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                                 viewport, progress_text, "", context)
                
                # Count status display
                count_text = f"Counted: {len(self.counted_balls)}/{len(self.ball_geoms)} balls"
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                                 mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 
                                 viewport, count_text, "", context)
                
                # Swap buffers and poll events
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
            except Exception as e:
                print(f"Error in rendering loop: {e}")
                break
            
            # Maintain a reasonable frame rate
            time_diff = time.time() - last_time
            if time_diff < 0.016:  # ~60 FPS
                time.sleep(0.016 - time_diff)
            last_time = time.time()
        
        # Show completion message
        print(f"Counted {len(self.counted_balls)} out of {len(self.ball_geoms)} balls")
        
        # Cleanup
        glfw.terminate()
        print("Finished pointing at all balls")

def main():
    # Generate a scene with balls in well-positioned locations
    xml_path = 'experiment_scen.xml'
    output_path = "model_with_balls.xml"
    
    # Generate the XML with multiple balls in reachable positions
    output_path, num_balls_added = generate_balls_module.generate_balls_xml(
        xml_path=xml_path,
        output_path=output_path, 
        num_balls=8,  # Number of balls to count
        ball_size=0.01, 
        ball_height=0.05,
        # Place balls in a wider area in a row for left-to-right counting
        min_x=-0.1, max_x=0.1,  # Wide area from left to right
        min_y=-0.15, max_y=-0.05  # In front of robot, at a good distance
    )
    
    print(f"Created model with {num_balls_added} balls at {output_path}")
    
    # Create and run the improved pointing test
    pointing_test = ImprovedBallPointingTest(output_path)
    pointing_test.point_at_balls_in_sequence()

if __name__ == "__main__":
    main()