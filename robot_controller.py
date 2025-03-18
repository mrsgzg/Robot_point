import mujoco
import numpy as np
import time
from ball_generation import generate_balls_module

# Configuration
POINT_OFFSET_DISTANCE = 0.25  # Height above the ball when pointing down
PAUSE_TIME = 1.0  # Seconds to pause at each ball when counting

class RobotController:
    def __init__(self, model, data):
        """Initialize robot controller with MuJoCo model and data"""
        self.model = model
        self.data = data
        
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
            
        # Find camera joints
        try:
            self.camera_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "camera_pan")
            self.camera_tilt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "camera_tilt")
            self.camera_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "camera_body")
            self.camera_mount_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "camera_mount")
            print(f"Found camera at pan joint id {self.camera_joint_id}, tilt joint id {self.camera_tilt_id}")
            print(f"Camera body id {self.camera_body_id}, camera mount id {self.camera_mount_id}")
        except Exception as e:
            print(f"Error finding camera: {e}")
            self.camera_joint_id = -1
            self.camera_tilt_id = -1
            
        # Find balls and sort them from left to right
        self.ball_geoms = self._find_and_sort_balls()
        
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
        #pointing_pos[1] -= 0.0
        
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
        ik_data.qpos[joint6_idx] = ik_data.qpos[joint6_idx] + 0.38
        
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

    def get_camera_angles_for_ball(self, ball_pos):
        """
        Calculate the camera pan and tilt angles to center the ball in the camera view
        
        Args:
            ball_pos (np.array): Ball position [x, y, z]
            
        Returns:
            tuple: (pan_angle, tilt_angle) in radians
        """
        if self.camera_joint_id < 0:
            return 0.0, 0.0
            
        # Get camera position
        camera_pos = self.data.xpos[self.camera_body_id].copy()
        
        # Calculate vector from camera to ball
        camera_to_ball = ball_pos - camera_pos
        
        # Calculate the angle in the x-y plane (pan)
        pan_angle = np.arctan2(-camera_to_ball[0], camera_to_ball[1])
        
        # Calculate the horizontal distance to the ball
        horizontal_distance = np.sqrt(camera_to_ball[0]**2 + camera_to_ball[1]**2)
        
        # Calculate the angle in the vertical plane (tilt)
        # Negative because tilting down is positive in most camera setups
        tilt_angle = -np.arctan2(camera_to_ball[2], horizontal_distance)
        
        print(f'***pan_angle{pan_angle},tilt_angle{tilt_angle}***')
        return pan_angle, -0.05
        
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