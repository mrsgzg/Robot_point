import mujoco
import numpy as np
import glfw
import time
from robot_controller import RobotController

class SimulationVisualizer:
    def __init__(self, model, data, controller):
        """Initialize visualization with camera controls and robot controller"""
        self.model = model
        self.data = data
        self.controller = controller
        
        # Camera state
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = -1  # Will be set to table_cam id
        
        # Try to find the table_cam
        for i in range(self.model.ncam):
            cam_name = self.model.cam(i).name
            if cam_name == "table_cam":
                self.camera.fixedcamid = i
                print(f"Using fixed camera: {cam_name} (id: {i})")
                break
                
            


        if self.camera.fixedcamid == -1:
            print("Warning: table_cam not found, using free camera")
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
        
        # Pointing state for visualization
        self.current_ball_index = 0
        self.motion_phase = "looking"  # New initial phase: "looking", "moving", or "paused"
        self.frame = 0
        self.interpolation_steps = 120  # More steps for smoother motion
        self.pause_until = time.time() + 1.0  # Start with a 1-second pause
        self.last_time = time.time()
        
        # Initial and target joint positions
        self.initial_arm_joints = np.zeros(len(controller.joint_indices))
        self.initial_finger_joints = np.zeros(len(controller.finger_joint_indices))
        self.target_arm_joints = None
        self.target_finger_joints = None
        
        # Camera control variables
        self.initial_camera_pan = 0.0
        self.initial_camera_tilt = 0.0
        self.target_camera_pan = 0.0
        self.target_camera_tilt = 0.0
        
    def setup_visualization(self):
        """Set up GLFW window with interactive camera controls"""
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1200, 900, "Robot Ball Counting", None, None)
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
        # Only use manual camera control if we're in free camera mode
        if self.camera.type != mujoco.mjtCamera.mjCAMERA_FREE:
            return
            
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
        # Only use scroll for free camera mode
        if self.camera.type != mujoco.mjtCamera.mjCAMERA_FREE:
            return
            
        # Zoom in/out
        zoom_factor = 1 - yoffset * 0.1
        self.camera.distance *= zoom_factor
        
        # Prevent extreme zooming
        self.camera.distance = max(0.1, min(self.camera.distance, 100))
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle key press events"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_C and action == glfw.PRESS:
            # Switch camera modes
            if self.camera.type == mujoco.mjtCamera.mjCAMERA_FREE:
                # Find table_cam id
                for i in range(self.model.ncam):
                    if self.model.cam(i).name == "table_cam":
                        self.camera.fixedcamid = i
                        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        print("Switched to fixed camera view")
                        break
            else:
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                print("Switched to free camera view")
    
    def visualize_pointing_sequence(self):
        """Run the visualization loop for the robot pointing sequence"""
        # Setup visualization
        if not self.setup_visualization():
            return
        
        # Check if we have balls
        if not self.controller.ball_geoms:
            print("No balls found in the model")
            glfw.terminate()
            return
        
        # Set up visualization
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Options and perturbation
        opt = mujoco.MjvOption()
        perturb = mujoco.MjvPerturb()
        
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.controller.camera_tilt_id] = -0.05
        # Initial pose - slightly raised arm
        for i, joint_idx in enumerate(self.controller.joint_indices):
            if i == 1:  # Joint 2 (shoulder)
                self.data.qpos[joint_idx] = 0.3  # Raised a bit
            elif i == 3:  # Joint 4 (elbow)
                self.data.qpos[joint_idx] = -0.5  # Bent slightly
        
        # Apply initial pose
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial joint positions
        for i, joint_idx in enumerate(self.controller.joint_indices):
            self.initial_arm_joints[i] = self.data.qpos[joint_idx]
            
        for i, joint_idx in enumerate(self.controller.finger_joint_indices):
            self.initial_finger_joints[i] = self.data.qpos[joint_idx]
            
        # Get initial camera angles
        if self.controller.camera_joint_id >= 0:
            self.initial_camera_pan = self.data.qpos[self.controller.camera_joint_id]
        if self.controller.camera_tilt_id >= 0:
            self.initial_camera_tilt = self.data.qpos[self.controller.camera_tilt_id]

        # Main rendering loop
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            # Handle the looking phase - waiting and camera orientation before starting movement
            if self.motion_phase == "looking":
                # Keep the robot in its initial pose
                for j in range(self.model.nu):
                    self.data.ctrl[j] = self.data.qpos[self.model.actuator_trnid[j, 0]]  # Hold current position
                
                current_time = time.time()
                if current_time >= self.pause_until:
                    # Move to the moving phase after the pause
                    self.motion_phase = "moving"
                    self.frame = 0
                    print("Starting ball pointing sequence...")
                    
                    # Calculate initial camera position (pan to the first ball)
                    if len(self.controller.ball_geoms) > 0:
                        ball_id = self.controller.ball_geoms[self.current_ball_index]
                        ball_pos = self.model.geom(ball_id).pos.copy()
                        self.target_camera_pan, self.target_camera_tilt = self.controller.get_camera_angles_for_ball(ball_pos)
                        
                        # Store initial camera angles for smooth transition
                        if self.controller.camera_joint_id >= 0:
                            self.initial_camera_pan = self.data.qpos[self.controller.camera_joint_id]
                        if self.controller.camera_tilt_id >= 0:
                            self.initial_camera_tilt = self.data.qpos[self.controller.camera_tilt_id]

            # Handle pause timing for counting
            if self.motion_phase == "paused":
                for j in range(self.model.nu):
                    self.data.ctrl[j] = self.data.qpos[self.model.actuator_trnid[j, 0]]  # Hold current position
                if current_time >= self.pause_until:
                    # Move to next ball
                    self.current_ball_index += 1
                    if self.current_ball_index >= len(self.controller.ball_geoms):
                        # Finished all balls, reset or exit
                        break
                    self.motion_phase = "moving"
                    self.frame = 0
                    
                    # Update initial joint positions
                    for i, joint_idx in enumerate(self.controller.joint_indices):
                        self.initial_arm_joints[i] = self.data.qpos[joint_idx]
                    
                    for i, joint_idx in enumerate(self.controller.finger_joint_indices):
                        self.initial_finger_joints[i] = self.data.qpos[joint_idx]
                        
                    # Update initial camera angles
                    if self.controller.camera_joint_id >= 0:
                        self.initial_camera_pan = self.data.qpos[self.controller.camera_joint_id]
                    if self.controller.camera_tilt_id >= 0:
                        self.initial_camera_tilt = self.data.qpos[self.controller.camera_tilt_id]
                        
                    self.controller.counted_balls.append(self.current_ball_index - 1)
            
            # If in moving phase, calculate new positions
            if self.motion_phase == "moving":
                # Get the current ball
                ball_id = self.controller.ball_geoms[self.current_ball_index]
                ball_pos = self.model.geom(ball_id).pos.copy()
                
                # Calculate joint angles for pointing at this ball (if not already calculated)
                if self.frame == 0:
                    joint_angles = self.controller.calculate_improved_ik(ball_pos)
                    self.target_arm_joints = joint_angles['arm_joints']
                    self.target_finger_joints = joint_angles['finger_joints']
                    
                    # Calculate camera angles to center the ball
                    self.target_camera_pan, self.target_camera_tilt = self.controller.get_camera_angles_for_ball(ball_pos)
                
                # Calculate interpolation factor
                if self.frame < self.interpolation_steps:
                    # Linear interpolation parameter
                    t = self.frame / self.interpolation_steps
                    # Apply smooth interpolation function
                    smooth_t = self.controller.smooth_interpolation(t)
                else:
                    smooth_t = 1.0
                    # Transition to paused state
                    self.motion_phase = "paused"
                    self.pause_until = current_time + 1.0  # Pause for 1 second
                
                # Interpolate joint positions
                for i, joint_idx in enumerate(self.controller.joint_indices):
                    if i < len(self.target_arm_joints):
                        # Calculate interpolated position
                        interp_pos = self.initial_arm_joints[i] + smooth_t * (self.target_arm_joints[i] - self.initial_arm_joints[i])
                        
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
                for i, joint_idx in enumerate(self.controller.finger_joint_indices):
                    if i < len(self.target_finger_joints):
                        interp_pos = self.initial_finger_joints[i] + smooth_t * (self.target_finger_joints[i] - self.initial_finger_joints[i])
                        self.data.qpos[joint_idx] = interp_pos
                        
                        # Set control if needed
                        for j in range(self.model.nu):
                            try:
                                if self.model.actuator_trnid[j, 0] == joint_idx:
                                    self.data.ctrl[j] = interp_pos
                            except:
                                pass
                                
                # Interpolate camera pan position
                if self.controller.camera_joint_id >= 0:
                    pan_interp = self.initial_camera_pan + smooth_t * (self.target_camera_pan - self.initial_camera_pan)
                    self.data.qpos[self.controller.camera_joint_id] = pan_interp
                    
                    # Set camera pan control
                    for j in range(self.model.nu):
                        try:
                            if self.model.actuator_trnid[j, 0] == self.controller.camera_joint_id:
                                self.data.ctrl[j] = pan_interp
                                break
                        except:
                            pass
                
                # Interpolate camera tilt position
                if self.controller.camera_tilt_id >= 0:
                    tilt_interp = self.initial_camera_tilt + smooth_t * (self.target_camera_tilt - self.initial_camera_tilt)
                    self.data.qpos[self.controller.camera_tilt_id] = tilt_interp
                    
                    # Set camera tilt control
                    for j in range(self.model.nu):
                        try:
                            if self.model.actuator_trnid[j, 0] == self.controller.camera_tilt_id:
                                self.data.ctrl[j] = tilt_interp
                                break
                        except:
                            pass
                
                # Increment frame counter
                self.frame += 1
            
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
                for idx, ball_id in enumerate(self.controller.ball_geoms):
                    ball_pos = self.model.geom(ball_id).pos.copy()
                    
                    # Add a number above each ball
                    number_pos = ball_pos.copy()
                    number_pos[2] += 0.05  # Slightly above the ball
                    
                    # Check if this is the current ball
                    is_current = (idx == self.current_ball_index)
                    
                    # Check if this ball has been counted
                    is_counted = idx in self.controller.counted_balls
                    
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
                if self.current_ball_index < len(self.controller.ball_geoms):
                    ball_id = self.controller.ball_geoms[self.current_ball_index]
                    ball_pos = self.model.geom(ball_id).pos.copy()
                    
                    # Get right finger position (index finger approximation)
                    finger_pos = self.data.xpos[self.controller.right_finger_body_id].copy()
                    
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

                if self.motion_phase == "looking":
                    countdown = self.pause_until - current_time
                    progress_text = f"Preparing to count: Looking at scene... {countdown:.1f}s"
                elif self.motion_phase == "moving":
                    if self.frame < self.interpolation_steps:
                        progress_text = f"Counting: Ball {self.current_ball_index+1}/{len(self.controller.ball_geoms)} - Moving: {(self.frame/self.interpolation_steps)*100:.1f}%"
                    else:
                        progress_text = f"Counting: Ball {self.current_ball_index+1}/{len(self.controller.ball_geoms)} - Arrived"
                else:  # paused
                    countdown = self.pause_until - current_time
                    progress_text = f"Counting: Ball {self.current_ball_index+1}/{len(self.controller.ball_geoms)} - Counting: {countdown:.1f}s"
                
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                                 mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                                 viewport, progress_text, "", context)
                
                # Count status display
                count_text = f"Counted: {len(self.controller.counted_balls)}/{len(self.controller.ball_geoms)} balls"
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                                 mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 
                                 viewport, count_text, "", context)
                
                # Camera view info
                cam_text = "View: Table Camera (Press 'C' to toggle free camera)" if self.camera.type == mujoco.mjtCamera.mjCAMERA_FIXED else "View: Free Camera (Press 'C' to toggle fixed camera)"
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                                 mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, 
                                 viewport, cam_text, "", context)
                
                # Swap buffers and poll events
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
            except Exception as e:
                print(f"Error in rendering loop: {e}")
                break
            
            # Maintain a reasonable frame rate
            time_diff = time.time() - self.last_time
            if time_diff < 0.016:  # ~60 FPS
                time.sleep(0.016 - time_diff)
            self.last_time = time.time()
        
        # Show completion message
        print(f"Counted {len(self.controller.counted_balls)} out of {len(self.controller.ball_geoms)} balls")
        
        # Cleanup
        glfw.terminate()
        print("Finished pointing at all balls")