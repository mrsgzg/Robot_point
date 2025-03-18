import os
import numpy as np
import cv2
import time
import json

class DataCollector:
    def __init__(self, model, data, camera_id, output_dir="collected_data"):
        """Initialize the data collector
        
        Args:
            model: The MuJoCo model
            data: The MuJoCo data
            camera_id: ID of the camera to use for image collection
            output_dir: Directory to save collected data
        """
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self.output_dir = output_dir
        
        # Create output directories
        self.img_dir = os.path.join(output_dir, "images")
        self.data_dir = os.path.join(output_dir, "joint_data")
        
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Data collection state
        self.frame_count = 0
        self.start_time = time.time()
        self.collected_data = []
    
    def collect_frame(self, current_ball_index=-1, counted_balls=None, motion_phase="unknown", image=None):
        """Collect data for the current frame
        
        Args:
            current_ball_index: Index of the ball currently being pointed at (-1 if none)
            counted_balls: List of balls that have been counted so far
            motion_phase: Current motion phase ("looking", "moving", "paused")
            image: Optional pre-captured image (if None, will be captured here)
        """
        timestamp = time.time() - self.start_time
        
        # Save provided image or capture a new one
        if image is not None:
            img_path = os.path.join(self.img_dir, f"frame_{self.frame_count:06d}.png")
            cv2.imwrite(img_path, image)
        else:
            # Fallback to capture method if no image provided
            img = self._capture_camera_image()
            if img is not None:
                img_path = os.path.join(self.img_dir, f"frame_{self.frame_count:06d}.png")
                cv2.imwrite(img_path, img)
        
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
        
        # Collect camera positions
        camera_position = None
        camera_orientation = None
        for i in range(self.model.ncam):
            if i == self.camera_id:
                camera_position = self.model.cam(i).pos.copy().tolist()
                camera_orientation = self.model.cam(i).quat.copy().tolist()
                break
        
        # Create frame data
        frame_data = {
            "frame": self.frame_count,
            "timestamp": timestamp,
            "current_ball_index": current_ball_index,
            "counted_balls": counted_balls if counted_balls is not None else [],
            "count": len(counted_balls) if counted_balls is not None else 0,
            "motion_phase": motion_phase,  # Added motion phase
            "joint_positions": joint_positions,
            "ball_positions": ball_positions,
            "camera": {
                "position": camera_position,
                "orientation": camera_orientation
            }
        }
        
        # Add to collected data
        self.collected_data.append(frame_data)
        
        # Save data periodically (every 100 frames)
        if self.frame_count % 100 == 0:
            self._save_data()
        
        self.frame_count += 1
    
    def _capture_camera_image(self):
        """Capture an image from the camera - fallback method if image not provided"""
        try:
            # Use mj_render to render to the image buffer
            # (This function is defined outside in simulation_visualizer.py)
            # For actual implementation, we defer to external extract_camera_image
            return None
        except Exception as e:
            print(f"Error capturing camera image: {e}")
            return None
    
    def _save_data(self):
        """Save collected data to a JSON file"""
        data_path = os.path.join(self.data_dir, f"data_{self.frame_count:06d}.json")
        with open(data_path, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
        
        print(f"Saved data to {data_path} ({len(self.collected_data)} frames)")
    
    def finish(self):
        """Finish data collection and save all remaining data"""
        self._save_data()
        
        # Save metadata
        metadata = {
            "total_frames": self.frame_count,
            "duration": time.time() - self.start_time,
            "image_path": self.img_dir,
            "data_path": self.data_dir
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data collection complete. {self.frame_count} frames collected over {metadata['duration']:.2f} seconds.")