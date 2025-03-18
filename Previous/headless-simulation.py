import mujoco
import numpy as np
import time
import os
import json
import cv2
from robot_controller import RobotController

def extract_camera_image(model, data, camera_id, width=640, height=480):
    """Extract an image from a camera view"""
    # Create scene and context for rendering
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Create camera
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = camera_id
    
    # Update scene with camera view
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), 
        None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    
    # Set up viewport
    viewport = mujoco.MjrRect(0, 0, width, height)
    
    # Allocate RGB buffer
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Render to buffer
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
    mujoco.mjr_render(viewport, scene, context)
    
    # Read pixels from buffer
    mujoco.mjr_readPixels(rgb_buffer, None, viewport, context)
    
    # Flip image vertically (OpenGL convention)
    rgb_buffer = np.flipud(rgb_buffer)
    
    return rgb_buffer

class FastSimulator:
    def __init__(self, model, data, controller, output_dir="collected_data_fast", image_width=320, image_height=240, image_freq=5):
        """Initialize optimized simulation for fast data collection with images
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            controller: Robot controller
            output_dir: Directory to save collected data
            image_width: Width of captured images (smaller = faster)
            image_height: Height of captured images (smaller = faster)
            image_freq: Capture an image every N frames (higher = faster)
        """
        self.model = model
        self.data = data
        self.controller = controller
        self.output_dir = output_dir
        self.image_width = image_width
        self.image_height = image_height
        self.image_freq = image_freq  # Capture every N frames
        
        # Motion parameters (can be adjusted for faster simulation)
        self.current_ball_index = 0
        self.motion_phase = "looking"  # "looking", "moving" or "paused"
        self.frame = 0
        self.interpolation_steps = 40  # Reduced for faster execution (original was 120)
        self.pause_time = 0.3  # Reduced pause time (original was 1.0)
        self.initial_pause_time = 0.2  # Reduced initial pause
        self.pause_until = time.time() + self.initial_pause_time
        
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
        
        # Data collection
        self.img_dir = os.path.join(output_dir, "images")
        self.data_dir = os.path.join(output_dir, "joint_data")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.data_frames = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Find camera ID
        self.camera_id = -1
        for i in range(self.model.ncam):
            if self.model.cam(i).name == "table_cam":
                self.camera_id = i
                print(f"Found table_cam with ID {i}")
                break
        
        if self.camera_id == -1:
            print("Warning: table_cam not found, will use camera 0")
            self.camera_id = 0
    
    def run_simulation(self):
        """Run the simulation with optimized rendering and image collection"""
        print(f"Starting optimized simulation for {len(self.controller.ball_geoms)} balls...")
        print(f"Capturing images every {self.image_freq} frames at {self.image_width}x{self.image_height} resolution")
        
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
        
        # Main simulation loop
        loop_start_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Handle looking phase - initial pause
            if self.motion_phase == "looking":
                # Keep the robot in its initial pose
                for j in range(self.model.nu):
                    self.data.ctrl[j] = self.data.qpos[self.model.actuator_trnid[j, 0]]
                
                if current_time >= self.pause_until:
                    # Move to the moving phase after the pause
                    self.motion_phase = "moving"
                    self.frame = 0
                    print("Starting ball pointing sequence...")
                    
                    # Calculate initial camera position for the first ball
                    if len(self.controller.ball_geoms) > 0:
                        ball_id = self.controller.ball_geoms[self.current_ball_index]
                        ball_pos = self.model.geom(ball_id).pos.copy()
                        self.target_camera_pan, self.target_camera_tilt = self.controller.get_camera_angles_for_ball(ball_pos)
            
            # Handle pause timing for counting
            elif self.motion_phase == "paused":
                for j in range(self.model.nu):
                    self.data.ctrl[j] = self.data.qpos[self.model.actuator_trnid[j, 0]]
                if current_time >= self.pause_until:
                    # Move to next ball
                    self.current_ball_index += 1
                    if self.current_ball_index >= len(self.controller.ball_geoms):
                        # Finished all balls
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
            elif self.motion_phase == "moving":
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
                    self.pause_until = current_time + self.pause_time
                
                # Interpolate joint positions
                for i, joint_idx in enumerate(self.controller.joint_indices):
                    if i < len(self.target_arm_joints):
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
                
                # Interpolate camera pan position
                if self.controller.camera_joint_id >= 0:
                    pan_interp = self.initial_camera_pan + smooth_t * (self.target_camera_pan - self.initial_camera_pan)
                    self.data.qpos[self.controller.camera_joint_id] = pan_interp
                
                # Interpolate camera tilt position
                if self.controller.camera_tilt_id >= 0:
                    tilt_interp = self.initial_camera_tilt + smooth_t * (self.target_camera_tilt - self.initial_camera_tilt)
                    self.data.qpos[self.controller.camera_tilt_id] = tilt_interp
                
                # Increment frame counter
                self.frame += 1
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Collect data every frame, but only capture images periodically
            camera_img = None
            if self.frame_count % self.image_freq == 0:
                # Capture camera image (this is what takes most of the time)
                try:
                    camera_img = extract_camera_image(
                        self.model, self.data, 
                        self.camera_id, 
                        width=self.image_width, height=self.image_height
                    )
                except Exception as e:
                    print(f"Error capturing image: {e}")
            
            # Always collect frame data
            self._collect_data_frame(camera_img)
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Progress tracking
            if self.frame_count % 100 == 0:
                balls_counted = len(self.controller.counted_balls)
                total_balls = len(self.controller.ball_geoms)
                elapsed = time.time() - loop_start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {balls_counted}/{total_balls} balls, {self.frame_count} frames, {elapsed:.2f} seconds ({fps:.1f} FPS)")
            
            self.frame_count += 1
        
        # Save final data
        self._save_data()
        
        # Print statistics
        elapsed_time = time.time() - loop_start_time
        print(f"Simulation complete: {len(self.controller.counted_balls)} balls counted")
        print(f"Total frames: {self.frame_count}")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Frames per second: {self.frame_count / elapsed_time:.2f}")
    
    def _collect_data_frame(self, camera_img=None):
        """Collect data for the current frame"""
        timestamp = time.time() - self.start_time
        
        # Save image if provided
        img_filename = None
        if camera_img is not None:
            img_filename = f"frame_{self.frame_count:06d}.png"
            img_path = os.path.join(self.img_dir, img_filename)
            cv2.imwrite(img_path, camera_img)
        
        # Collect joint positions
        joint_positions = {}
        for i in range(self.model.njnt):
            joint_name = self.model.jnt(i).name
            joint_adr = self.model.jnt(i).qposadr
            if joint_adr >= 0 and joint_adr < self.data.qpos.shape[0]:
                joint_positions[joint_name] = float(self.data.qpos[joint_adr])
        
        # Collect ball positions
        ball_positions = {}
        for i in range(self.model.ngeom):
            geom_name = self.model.geom(i).name
            if 'ball' in geom_name and self.model.geom(i).type == 5:  # mjGEOM_SPHERE = 5
                pos = self.model.geom(i).pos.copy().tolist()
                ball_positions[geom_name] = pos
        
        # Create frame data
        frame_data = {
            "frame": self.frame_count,
            "timestamp": timestamp,
            "current_ball_index": self.current_ball_index,
            "counted_balls": self.controller.counted_balls.copy(),
            "count": len(self.controller.counted_balls),
            "motion_phase": self.motion_phase,
            "joint_positions": joint_positions,
            "ball_positions": ball_positions,
            "image_file": img_filename  # Will be None if no image was captured
        }
        
        # Add to collected data
        self.data_frames.append(frame_data)
        
        # Save data periodically
        if self.frame_count % 500 == 0 and self.frame_count > 0:
            self._save_data()
            # Clear the list to save memory
            self.data_frames = []
    
    def _save_data(self):
        """Save collected data to a JSON file"""
        if not self.data_frames:
            return
            
        data_path = os.path.join(self.data_dir, f"data_{self.frame_count:06d}.json")
        with open(data_path, 'w') as f:
            json.dump(self.data_frames, f)
        
        print(f"Saved {len(self.data_frames)} frames to {data_path}")

# Main function for fast simulation
def run_fast_simulation(xml_path, num_balls=15, min_x=-0.1, max_x=0.1, min_y=-0.15, max_y=-0.05, 
                        output_dir="collected_data_fast", image_width=320, image_height=240, image_freq=5):
    from ball_generation import generate_balls_module
    
    # Generate model with balls
    ball_size = 0.01
    table_surface_z = 0.025
    ball_height = table_surface_z + ball_size
    
    output_path = "model_with_balls_fast.xml"
    
    # Generate the XML with balls
    output_path, num_balls_added = generate_balls_module.generate_balls_xml(
        xml_path=xml_path,
        output_path=output_path, 
        num_balls=num_balls,
        ball_size=ball_size, 
        ball_height=ball_height,
        min_x=min_x, max_x=max_x,
        min_y=min_y, max_y=max_y
    )
    
    print(f"Created model with {num_balls_added} balls")
    
    # Load the model and create data
    model = mujoco.MjModel.from_xml_path(output_path)
    data = mujoco.MjData(model)
    
    # Initialize controller
    controller = RobotController(model, data)
    
    # Initialize and run fast simulator
    simulator = FastSimulator(
        model, data, controller, output_dir,
        image_width=image_width, 
        image_height=image_height,
        image_freq=image_freq
    )
    simulator.run_simulation()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimized robot ball counting simulation with image capture')
    parser.add_argument('--xml_path', default='experiment_scen.xml', help='Path to the base XML model file')
    parser.add_argument('--num_balls', type=int, default=5, help='Number of balls to generate')
    parser.add_argument('--min_x', type=float, default=-0.025, help='Minimum x-coordinate for ball placement')
    parser.add_argument('--max_x', type=float, default=0.025, help='Maximum x-coordinate for ball placement')
    parser.add_argument('--min_y', type=float, default=-0.2, help='Minimum y-coordinate for ball placement')
    parser.add_argument('--max_y', type=float, default=-0.15, help='Maximum y-coordinate for ball placement')
    parser.add_argument('--output_dir', default='collected_data_fast', help='Directory to save collected data')
    parser.add_argument('--image_width', type=int, default=320, help='Width of captured images')
    parser.add_argument('--image_height', type=int, default=240, help='Height of captured images')
    parser.add_argument('--image_freq', type=int, default=5, help='Capture an image every N frames')
    
    args = parser.parse_args()
    dir_name = "Dataset/"+str(args.num_balls)+"/"+str(time.time())
    run_fast_simulation(
        xml_path=args.xml_path,
        num_balls=args.num_balls,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        output_dir=args.output_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        image_freq=args.image_freq
    )