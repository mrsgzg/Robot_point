import mujoco
import numpy as np
import glfw
import time
from ball_generation import generate_balls_module
# Configuration
BALL_HEIGHT_OFFSET = 0.05  # 5cm above ball surface
XML_PATH = "model_with_balls.xml"  # Assumes generate_balls.py creates this

class BallPointingTest:
    def __init__(self, xml_path):
        """Initialize MuJoCo simulation with ball-pointing capabilities"""
        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Camera state
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 4.0
        self.camera.elevation = -30
        self.camera.azimuth = 90
        self.camera.lookat = np.array([0.0, 0.0, 0.0])
        
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
        self.joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
            for name in self.joint_names
        ]
        
        # Find balls
        self.ball_geoms = self._find_ball_geoms()
        
        # Pointing state
        self.initial_joint_positions = None
        self.target_joint_positions = None
        self.interpolation_progress = 0.0
    
    def _find_ball_geoms(self):
        """Find all ball geoms in the simulation"""
        ball_geoms = []
        for i in range(self.model.ngeom):
            try:
                # Try to get the geom name and type
                geom_name = self.model.geom(i).name
                geom_type = self.model.geom(i).type
                
                # Check if it's a ball (sphere)
                if 'ball' in geom_name and geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    ball_geoms.append(i)
            except Exception as e:
                print(f"Error processing geom {i}: {e}")
        
        print(f"Found {len(ball_geoms)} ball geoms")
        return ball_geoms
    
    def _get_default_home_position(self):
        """Retrieve the default home position for specific joints"""
        # Try to get home position from keyframe first
        try:
            home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEYFRAME, "home")
            if home_key_id != -1:
                home_pos = self.model.key_qpos[home_key_id]
                
                # Extract positions for specific joints
                joint_positions = []
                for joint_name in self.joint_names:
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id != -1:
                        joint_positions.append(home_pos[joint_id])
                
                return np.array(joint_positions)
        except:
            pass
        
        # Fallback default position (zeros for specific joints)
        return np.zeros(len(self.joint_names))
    
    def setup_visualization(self):
        """Set up GLFW window with interactive camera controls"""
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1200, 900, "Ball Pointing Test", None, None)
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
        
        # Get window size for coordinate scaling
        width, height = glfw.get_framebuffer_size(window)
        
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
    
    def find_leftmost_ball(self):
        """Identify the leftmost ball based on X coordinate"""
        leftmost_ball = None
        leftmost_x = float('inf')
        
        for geom_id in self.ball_geoms:
            # Get ball position
            ball_pos = self.model.geom(geom_id).pos
            
            # Compare X coordinate
            if ball_pos[0] < leftmost_x:
                leftmost_x = ball_pos[0]
                leftmost_ball = geom_id
        
        if leftmost_ball is None:
            raise ValueError("No balls found in the simulation")
        
        #leftmost_ball = self.ball_geoms[9]

        return leftmost_ball, self.model.geom(leftmost_ball).pos
    
    def calculate_inverse_kinematics(self, target_pos):
        """Use MuJoCo's Jacobian-based approach for IK"""
        # Create a temporary data copy for IK calculations
        temp_data = mujoco.MjData(self.model)
        
        # Initialize with current joint positions
        for i, joint_idx in enumerate(self.joint_indices):
            temp_data.qpos[joint_idx] = self.data.qpos[joint_idx]
        
        # Get the end effector body ID (the hand)
        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        
        # Transform target position to be relative to the robot base
        # The robot is at (0, -0.9, 0) as defined in the XML
        robot_base_pos = np.array([0, -0.9, 0])
        rel_target = target_pos - robot_base_pos
        
        # IK parameters
        max_iter = 100
        tolerance = 1e-3
        step_size = 0.1
        damping = 0.1
        
        # Iterative IK loop
        for iteration in range(max_iter):
            # Forward kinematics to update current positions
            mujoco.mj_forward(self.model, temp_data)
            
            # Get current end effector position
            current_pos = temp_data.xpos[end_effector_id]
            
            # Calculate error (difference between target and current position)
            error = rel_target - current_pos
            error_magnitude = np.linalg.norm(error)
            
            # Check convergence
            if error_magnitude < tolerance:
                print(f"IK converged after {iteration} iterations with error {error_magnitude:.6f}")
                break
            
            # Calculate Jacobian matrix for the end effector
            jacp = np.zeros((3, self.model.nv))  # Position Jacobian
            jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
            mujoco.mj_jacBody(self.model, temp_data, jacp, jacr, end_effector_id)
            
            # Create a filtered Jacobian for just our controlled joints
            jac_filtered = np.zeros((3, len(self.joint_indices)))
            
            for i, joint_idx in enumerate(self.joint_indices):
                # Get the velocity index corresponding to this joint
                dof_adr = self.model.jnt_dofadr[joint_idx]
                jac_filtered[:, i] = jacp[:, dof_adr]
            
            # Damped Least Squares solution for IK
            jac_transpose = jac_filtered.T
            regularization = damping * np.eye(3)
            dls_term = np.linalg.inv(jac_filtered @ jac_transpose + regularization)
            dq = step_size * jac_transpose @ dls_term @ error
            
            # Apply joint updates
            for i, joint_idx in enumerate(self.joint_indices):
                new_pos = temp_data.qpos[joint_idx] + dq[i]
                
                # Check joint limits
                if joint_idx < len(self.model.jnt_range):
                    lower = self.model.jnt_range[joint_idx][0]
                    upper = self.model.jnt_range[joint_idx][1]
                    
                    if lower < upper:  # Valid range
                        new_pos = min(max(new_pos, lower), upper)
                
                # Update joint position
                temp_data.qpos[joint_idx] = new_pos
        else:
            print(f"IK did not converge after {max_iter} iterations")
        
        # Extract final joint positions
        result = np.zeros(len(self.joint_indices))
        for i, joint_idx in enumerate(self.joint_indices):
            result[i] = temp_data.qpos[joint_idx]
        
        return result
    
    def interpolate_joint_positions(self, start_pos, end_pos, progress):
        """
        Interpolate between start and end joint positions
        
        Args:
            start_pos (np.array): Starting joint positions
            end_pos (np.array): Target joint positions
            progress (float): Interpolation progress (0.0 to 1.0)
        
        Returns:
            np.array: Interpolated joint positions
        """
        # Linear interpolation
        interpolated_pos = start_pos + (end_pos - start_pos) * progress
        return interpolated_pos
    
    def point_at_leftmost_ball(self):
        """Demonstrate pointing at the leftmost ball with smooth motion"""
        # Setup visualization
        if not self.setup_visualization():
            return
        
        # Find the leftmost ball
        leftmost_ball, ball_pos = self.find_leftmost_ball()
        
        # Calculate pointing target (5cm above the ball)
        point_target = ball_pos.copy()
        point_target[2] += self.model.geom(leftmost_ball).size[0] + BALL_HEIGHT_OFFSET
        
        print(f"Pointing at ball at position: {point_target}")
        
        # Get default home position
        self.initial_joint_positions = self._get_default_home_position()
        
        # Calculate target joint positions
        self.target_joint_positions = self.calculate_inverse_kinematics(point_target)
        
        # Scene and context setup
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        opt = mujoco.MjvOption()
        perturb = mujoco.MjvPerturb()
        
        # Reset data to initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions
        for i, joint_index in enumerate(self.joint_indices):
            if i < len(self.initial_joint_positions):
                self.data.qpos[joint_index] = self.initial_joint_positions[i]
        
        # Interpolation parameters
        interpolation_duration = 3.0  # seconds
        start_time = time.time()
        
        # Simulation loop
        while not glfw.window_should_close(self.window):
            # Calculate interpolation progress
            current_time = time.time()
            self.interpolation_progress = min(1.0, (current_time - start_time) / interpolation_duration)
            
            # Interpolate joint positions
            current_joint_positions = self.interpolate_joint_positions(
                self.initial_joint_positions, 
                self.target_joint_positions, 
                self.interpolation_progress
            )
            
            # Set joint controls
            for i, joint_index in enumerate(self.joint_indices):
                if i < len(current_joint_positions):
                    # Zero out controls to prevent spring-back
                    self.data.ctrl[i] = current_joint_positions[i]
                    # Also set the joint position directly
                    self.data.qpos[joint_index] = current_joint_positions[i]
            
            # Perform simulation step
            mujoco.mj_step1(self.model, self.data)
            
            # Update scene for rendering
            mujoco.mjv_updateScene(self.model, self.data, opt, perturb, self.camera, 
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            
            # Get viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            # Render scene
            mujoco.mjr_render(viewport, scene, context)
            
            # Add text overlay with joint angles and progress
            help_text = f"Progress: {self.interpolation_progress:.2f}"
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                               viewport, help_text, "", context)
            
            # Swap front and back buffers
            glfw.swap_buffers(self.window)
            
            # Poll for and process events
            glfw.poll_events()
            
            # Complete simulation step
            mujoco.mj_step2(self.model, self.data)
            
            # Exit when interpolation is complete
            if self.interpolation_progress >= 1.0:
                time.sleep(2)  # Pause at final position
                break
        
        # Cleanup
        glfw.terminate()

def main():
    # Create the pointing test instance
    output_path, num_balls_added = generate_balls_module.generate_balls_xml(xml_path='experiment_scen.xml',
                                                    output_path="model_with_balls.xml", num_balls=1, 
                                                    ball_size=0.03, ball_height=0.05, 
                                                    min_x=-0.5, max_x=0.5, min_y=-0.5, max_y=0.3)
    print(f"Created model with {num_balls_added} white balls at {output_path}")
    pointing_test = BallPointingTest(XML_PATH)
    
    # Run the pointing demonstration
    pointing_test.point_at_leftmost_ball()

if __name__ == "__main__":
    main()