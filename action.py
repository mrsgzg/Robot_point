import mujoco
import numpy as np
import glfw

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
        
        # Mouse interaction state
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Find balls
        self.ball_geoms = self._find_ball_geoms()
        
        # Prepare joint control
        self.joint_names = [
            "joint1", "joint2", "joint3", 
            "joint4", "joint5", "joint6", "joint7"
        ]
        self.joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
            for name in self.joint_names
        ]
        
        # Setup window and callbacks
        self.setup_visualization()
    
    def _find_ball_geoms(self):
        """Find all ball geoms in the simulation"""
        ball_geoms = []
        for i in range(self.model.ngeom):
            try:
                geom_name = self.model.geom(i).name
                geom_type = self.model.geom(i).type
                
                # Check if it's a ball (sphere)
                if 'ball' in geom_name and geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    ball_geoms.append(i)
            except Exception as e:
                print(f"Error processing geom {i}: {e}")
        
        print(f"Found {len(ball_geoms)} ball geoms")
        return ball_geoms
    
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button interactions"""
        x, y = glfw.get_cursor_pos(window)
        
        # Convert window coordinates to MuJoCo coordinates
        width, height = glfw.get_framebuffer_size(window)
        
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
        # Calculate mouse movement
        dx = xpos - self.last_mouse_x
        dy = ypos - self.last_mouse_y
        
        # Left button: rotate camera
        if self.button_left:
            self.camera.azimuth += dx * 0.1
            self.camera.elevation -= dy * 0.1
            
            # Clamp elevation
            self.camera.elevation = max(-90, min(90, self.camera.elevation))
        
        # Middle button: translate camera
        elif self.button_middle:
            # Translate based on camera distance
            forward = self.camera.distance * 0.01
            right = forward * (width / height)
            
            # Adjust translation based on camera orientation
            self.camera.lookat[0] -= right * dx
            self.camera.lookat[1] += forward * dy
        
        # Right button: zoom
        elif self.button_right:
            # Zoom in/out
            self.camera.distance *= (1 - dy * 0.01)
            # Prevent zooming too close or too far
            self.camera.distance = max(0.1, min(self.camera.distance, 100))
        
        # Update last mouse position
        self.last_mouse_x = xpos
        self.last_mouse_y = ypos
    
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll for zooming"""
        # Zoom in/out
        self.camera.distance *= (1 - yoffset * 0.1)
        # Prevent zooming too close or too far
        self.camera.distance = max(0.1, min(self.camera.distance, 100))
    
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
        
        return True
    
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
        
        return leftmost_ball, self.model.geom(leftmost_ball).pos
    
    def calculate_inverse_kinematics(self, target_pos):
        """
        Simple placeholder for IK 
        In a real scenario, you'd use a proper IK solver
        """
        # This is a very basic and likely incorrect approximation
        # Real IK would require a sophisticated solver
        joint_angles = np.zeros(len(self.joint_indices))
        
        # Basic joint angle estimation (purely demonstrative)
        # You would replace this with a proper IK solution
        joint_angles[0] = np.arctan2(target_pos[1], target_pos[0])  # Base rotation
        joint_angles[1] = np.pi/4  # Rough angle to point downward
        joint_angles[2] = -np.pi/4  # Compensate for arm geometry
        
        return joint_angles
    
    def point_at_leftmost_ball(self):
        """Demonstrate pointing at the leftmost ball"""
        # Find the leftmost ball
        leftmost_ball, ball_pos = self.find_leftmost_ball()
        
        # Calculate pointing target (5cm above the ball)
        point_target = ball_pos.copy()
        point_target[2] += self.model.geom(leftmost_ball).size[0] + BALL_HEIGHT_OFFSET
        
        print(f"Pointing at ball at position: {point_target}")
        
        # Calculate rough joint angles
        target_joint_angles = self.calculate_inverse_kinematics(point_target)
        
        # Scene and context setup
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        opt = mujoco.MjvOption()
        perturb = mujoco.MjvPerturb()
        
        # Reset data to initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # Simulation loop
        while not glfw.window_should_close(self.window):
            # Set joint controls
            for i, joint_index in enumerate(self.joint_indices):
                if i < len(target_joint_angles):
                    # Zero out controls to prevent spring-back
                    self.data.ctrl[i] = target_joint_angles[i]
                    # Also set the joint position directly
                    self.data.qpos[joint_index] = target_joint_angles[i]
            
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
            
            # Add text overlay with joint angles
            help_text = " ".join([f"{name}: {angle:.2f}" for name, angle in zip(self.joint_names, target_joint_angles)])
            help_text += "\nLMB: Rotate, MMB: Pan, RMB/Scroll: Zoom"
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                               viewport, help_text, "", context)
            
            # Swap front and back buffers
            glfw.swap_buffers(self.window)
            
            # Poll for and process events
            glfw.poll_events()
            
            # Complete simulation step
            mujoco.mj_step2(self.model, self.data)
        
        # Cleanup
        glfw.terminate()

def main():import mujoco
import numpy as np
import glfw

# Configuration
BALL_HEIGHT_OFFSET = 0.05  # 5cm above ball surface
XML_PATH = "model_with_balls.xml"  # Assumes generate_balls.py creates this

class BallPointingTest:
    def __init__(self, xml_path):
        """Initialize MuJoCo simulation with ball-pointing capabilities"""
        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Find ball and end-effector geoms
        self.ball_geoms = self._find_ball_geoms()
        
        # Prepare joint control
        self.joint_names = [
            "joint1", "joint2", "joint3", 
            "joint4", "joint5", "joint6", "joint7"
        ]
        self.joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
            for name in self.joint_names
        ]
        
        # Camera and window setup
        self.setup_visualization()
    
    def _find_ball_geoms(self):
        """Find all ball geoms in the simulation"""
        ball_geoms = []
        for i in range(self.model.ngeom):
            try:
                geom_name = self.model.geom(i).name
                geom_type = self.model.geom(i).type
                
                # Check if it's a ball (sphere)
                if 'ball' in geom_name and geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    ball_geoms.append(i)
            except Exception as e:
                print(f"Error processing geom {i}: {e}")
        
        print(f"Found {len(ball_geoms)} ball geoms")
        return ball_geoms
    
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
        
        return leftmost_ball, self.model.geom(leftmost_ball).pos
    
    def setup_visualization(self):
        """Set up GLFW window for visualization"""
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
        return True
    
    def calculate_inverse_kinematics(self, target_pos):
        """
        Simple placeholder for IK 
        In a real scenario, you'd use a proper IK solver
        """
        # This is a very basic and likely incorrect approximation
        # Real IK would require a sophisticated solver
        joint_angles = np.zeros(len(self.joint_indices))
        
        # Basic joint angle estimation (purely demonstrative)
        # You would replace this with a proper IK solution
        joint_angles[0] = np.arctan2(target_pos[1], target_pos[0])  # Base rotation
        joint_angles[1] = np.pi/4  # Rough angle to point downward
        joint_angles[2] = -np.pi/4  # Compensate for arm geometry
        
        return joint_angles
    
    def point_at_leftmost_ball(self):
        """Demonstrate pointing at the leftmost ball"""
        # Find the leftmost ball
        leftmost_ball, ball_pos = self.find_leftmost_ball()
        
        # Calculate pointing target (5cm above the ball)
        point_target = ball_pos.copy()
        point_target[2] += self.model.geom(leftmost_ball).size[0] + BALL_HEIGHT_OFFSET
        
        print(f"Pointing at ball at position: {point_target}")
        
        # Calculate rough joint angles
        target_joint_angles = self.calculate_inverse_kinematics(point_target)
        
        # Visualization setup
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 4.0
        camera.elevation = -30
        camera.azimuth = 90
        
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        opt = mujoco.MjvOption()
        perturb = mujoco.MjvPerturb()
        
        # Reset data to initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # Simulation loop
        while not glfw.window_should_close(self.window):
            # Set joint controls
            for i, joint_index in enumerate(self.joint_indices):
                if i < len(target_joint_angles):
                    # Zero out controls to prevent spring-back
                    self.data.ctrl[i] = target_joint_angles[i]
                    # Also set the joint position directly
                    self.data.qpos[joint_index] = target_joint_angles[i]
            
            # Perform simulation step
            mujoco.mj_step1(self.model, self.data)
            
            # Update scene for rendering
            mujoco.mjv_updateScene(self.model, self.data, opt, perturb, camera, 
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            
            # Get viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            # Render scene
            mujoco.mjr_render(viewport, scene, context)
            
            # Add text overlay with joint angles
            help_text = " ".join([f"{name}: {angle:.2f}" for name, angle in zip(self.joint_names, target_joint_angles)])
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, 
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                               viewport, help_text, "", context)
            
            # Swap front and back buffers
            glfw.swap_buffers(self.window)
            
            # Poll for and process events
            glfw.poll_events()
            
            # Complete simulation step
            mujoco.mj_step2(self.model, self.data)
        
        # Cleanup
        glfw.terminate()

def main():
    # Create the pointing test instance
    pointing_test = BallPointingTest(XML_PATH)
    
    # Run the pointing demonstration
    pointing_test.point_at_leftmost_ball()

if __name__ == "__main__":
    main()
    # Create the pointing test instance
    pointing_test = BallPointingTest(XML_PATH)
    
    # Run the pointing demonstration
    pointing_test.point_at_leftmost_ball()

if __name__ == "__main__":
    main()