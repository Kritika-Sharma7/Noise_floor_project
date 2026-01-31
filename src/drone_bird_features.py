"""
NOISE FLOOR - Drone vs Bird Feature Extraction
================================================
Specialized feature extraction for airspace anomaly detection.

Detects behavioral differences between:
- Birds (natural, organic movement) → NORMAL
- Drones (mechanical, rigid movement) → ANOMALY

Key differentiating features:
1. Movement Pattern: Organic vs Mechanical
2. Speed Consistency: Variable vs Constant
3. Direction Changes: Smooth curves vs Sharp turns
4. Hover Behavior: Rare vs Common
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Drone-Bird specific behavioral features
DRONE_BIRD_FEATURES = [
    # Motion characteristics
    'motion_energy',           # Overall movement intensity
    'motion_consistency',      # How consistent is the motion
    'velocity_mean',           # Average speed
    'velocity_std',            # Speed variation (birds vary more)
    'velocity_max',            # Maximum speed
    
    # Direction features
    'direction_mean',          # Average direction
    'direction_std',           # Direction variation
    'direction_entropy',       # Randomness of direction
    'direction_change_rate',   # How often direction changes
    'turn_sharpness',          # Sharp turns = drone, smooth = bird
    
    # Trajectory features
    'path_linearity',          # How straight is the path
    'path_curvature',          # Curvature of movement
    'hover_ratio',             # Time spent hovering (drones hover more)
    'acceleration_mean',       # Average acceleration
    'acceleration_std',        # Acceleration variation
    
    # Spatial features
    'centroid_x',              # X position
    'centroid_y',              # Y position
    'bounding_area',           # Size of moving object
    'aspect_ratio',            # Shape of moving object
    'size_consistency',        # Drones have consistent size
    
    # Temporal features
    'temporal_stability',      # Movement stability over time
    'periodicity',             # Periodic motion (wing flaps vs rotors)
    'jerk_mean',               # Rate of acceleration change
    'smoothness_index',        # Overall motion smoothness
]


class DroneBirdFeatureExtractor:
    """
    Extract behavioral features from drone/bird videos.
    
    Birds: Organic, variable, curved movements
    Drones: Mechanical, constant, rigid movements
    """
    
    def __init__(self, sample_interval: int = 2):
        self.sample_interval = sample_interval
        self.prev_frame = None
        self.prev_flow = None
        self.trajectory_history = []
        self.velocity_history = []
        self.direction_history = []
        self.frame_count = 0
        
    def reset(self):
        """Reset extractor state for new video."""
        self.prev_frame = None
        self.prev_flow = None
        self.trajectory_history = []
        self.velocity_history = []
        self.direction_history = []
        self.frame_count = 0
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 24 behavioral features from a single frame.
        
        Args:
            frame: Grayscale or BGR frame
            
        Returns:
            Feature vector of length 24
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Resize for consistent processing
        gray = cv2.resize(gray, (320, 240))
        
        self.frame_count += 1
        
        # Initialize features
        features = np.zeros(len(DRONE_BIRD_FEATURES))
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return features
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Extract flow components
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)
        
        # Find moving regions (threshold)
        motion_mask = magnitude > 1.0
        
        if np.sum(motion_mask) > 100:  # Significant motion
            # Motion characteristics
            features[0] = np.mean(magnitude[motion_mask])  # motion_energy
            features[1] = 1.0 - (np.std(magnitude[motion_mask]) / (np.mean(magnitude[motion_mask]) + 1e-6))  # motion_consistency
            
            # Velocity features
            velocities = magnitude[motion_mask]
            features[2] = np.mean(velocities)  # velocity_mean
            features[3] = np.std(velocities)   # velocity_std
            features[4] = np.max(velocities)   # velocity_max
            
            # Direction features
            directions = angle[motion_mask]
            features[5] = np.mean(directions)  # direction_mean
            features[6] = np.std(directions)   # direction_std
            
            # Direction entropy
            hist, _ = np.histogram(directions, bins=8, range=(-np.pi, np.pi))
            hist = hist / (np.sum(hist) + 1e-6)
            features[7] = -np.sum(hist * np.log(hist + 1e-6)) / np.log(8)  # direction_entropy
            
            # Store for temporal analysis
            self.velocity_history.append(np.mean(velocities))
            self.direction_history.append(np.mean(directions))
            
            # Direction change rate (from history)
            if len(self.direction_history) >= 2:
                dir_changes = np.abs(np.diff(self.direction_history[-10:]))
                features[8] = np.mean(dir_changes) if len(dir_changes) > 0 else 0  # direction_change_rate
                
                # Turn sharpness (sharp = drone, smooth = bird)
                features[9] = np.max(dir_changes) if len(dir_changes) > 0 else 0  # turn_sharpness
            
            # Trajectory features
            # Path linearity (1 = straight line, 0 = random)
            if len(self.velocity_history) >= 5:
                vel_array = np.array(self.velocity_history[-10:])
                features[10] = 1.0 - (np.std(vel_array) / (np.mean(vel_array) + 1e-6))  # path_linearity
            
            # Path curvature
            if len(self.direction_history) >= 3:
                curvature = np.abs(np.diff(self.direction_history[-5:], n=2))
                features[11] = np.mean(curvature) if len(curvature) > 0 else 0  # path_curvature
            
            # Hover ratio (low velocity = hovering)
            hover_threshold = 0.5
            features[12] = np.sum(velocities < hover_threshold) / len(velocities)  # hover_ratio
            
            # Acceleration
            if len(self.velocity_history) >= 2:
                accelerations = np.diff(self.velocity_history[-10:])
                features[13] = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0  # acceleration_mean
                features[14] = np.std(accelerations) if len(accelerations) > 0 else 0  # acceleration_std
            
            # Spatial features - centroid of motion
            motion_coords = np.where(motion_mask)
            features[15] = np.mean(motion_coords[1]) / 320  # centroid_x (normalized)
            features[16] = np.mean(motion_coords[0]) / 240  # centroid_y (normalized)
            
            # Bounding area
            if len(motion_coords[0]) > 0:
                bbox_h = np.max(motion_coords[0]) - np.min(motion_coords[0])
                bbox_w = np.max(motion_coords[1]) - np.min(motion_coords[1])
                features[17] = (bbox_h * bbox_w) / (320 * 240)  # bounding_area (normalized)
                features[18] = bbox_w / (bbox_h + 1e-6)  # aspect_ratio
            
            # Size consistency (drones = consistent, birds = variable due to wing flaps)
            self.trajectory_history.append(features[17])
            if len(self.trajectory_history) >= 3:
                features[19] = 1.0 - np.std(self.trajectory_history[-10:])  # size_consistency
            
            # Temporal stability
            if len(self.velocity_history) >= 5:
                features[20] = 1.0 - (np.std(self.velocity_history[-10:]) / (np.mean(self.velocity_history[-10:]) + 1e-6))  # temporal_stability
            
            # Periodicity (wing flaps create periodic motion)
            if len(self.velocity_history) >= 10:
                vel_signal = np.array(self.velocity_history[-20:])
                if len(vel_signal) >= 10:
                    fft = np.abs(np.fft.fft(vel_signal))
                    features[21] = np.max(fft[1:len(fft)//2]) / (np.mean(fft) + 1e-6)  # periodicity
            
            # Jerk (rate of acceleration change)
            if len(self.velocity_history) >= 3:
                jerks = np.diff(self.velocity_history[-10:], n=2)
                features[22] = np.mean(np.abs(jerks)) if len(jerks) > 0 else 0  # jerk_mean
            
            # Smoothness index
            features[23] = 1.0 - features[22]  # smoothness_index (inverse of jerk)
        
        else:
            # No significant motion - likely hovering or static
            features[12] = 1.0  # hover_ratio = 1 (hovering)
        
        # Store previous frame
        self.prev_frame = gray
        self.prev_flow = flow
        
        # Clip and normalize
        features = np.clip(features, 0, 10)
        
        return features


class DroneBirdDatasetLoader:
    """
    Load videos from Drone-vs-Bird dataset.
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.bird_path = self.dataset_path / "bird"
        self.drone_path = self.dataset_path / "drone"
        
    def get_video_files(self, category: str = "bird") -> List[Path]:
        """Get list of video files for a category."""
        if category == "bird":
            search_path = self.bird_path
        else:
            search_path = self.drone_path
            
        if not search_path.exists():
            logger.warning(f"Path does not exist: {search_path}")
            return []
        
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(search_path.glob(f"*{ext}"))
        
        return sorted(videos)
    
    def load_video_frames(
        self, 
        video_path: Path, 
        max_frames: int = 200,
        sample_interval: int = 2
    ) -> List[np.ndarray]:
        """Load frames from a video file."""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return frames
        
        frame_count = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Loaded {len(frames)} frames from {video_path.name}")
        return frames
    
    def load_training_data(
        self, 
        max_videos: int = 10,
        max_frames_per_video: int = 100
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load bird videos for training (normal behavior).
        
        Returns:
            frames: List of all frames
            labels: List of video sources
        """
        all_frames = []
        all_labels = []
        
        bird_videos = self.get_video_files("bird")[:max_videos]
        
        for video_path in bird_videos:
            frames = self.load_video_frames(video_path, max_frames_per_video)
            all_frames.extend(frames)
            all_labels.extend([video_path.stem] * len(frames))
        
        logger.info(f"Loaded {len(all_frames)} training frames from {len(bird_videos)} bird videos")
        return all_frames, all_labels
    
    def load_test_data(
        self,
        max_videos: int = 5,
        max_frames_per_video: int = 100
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load drone videos for testing (anomaly).
        
        Returns:
            frames: List of all frames
            labels: List of video sources
        """
        all_frames = []
        all_labels = []
        
        drone_videos = self.get_video_files("drone")[:max_videos]
        
        for video_path in drone_videos:
            frames = self.load_video_frames(video_path, max_frames_per_video)
            all_frames.extend(frames)
            all_labels.extend([video_path.stem] * len(frames))
        
        logger.info(f"Loaded {len(all_frames)} test frames from {len(drone_videos)} drone videos")
        return all_frames, all_labels


def extract_features_from_video(
    video_path: Path,
    extractor: DroneBirdFeatureExtractor,
    max_frames: int = 200
) -> np.ndarray:
    """Extract features from entire video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return np.array([])
    
    features_list = []
    extractor.reset()
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 2 == 0:  # Sample every 2nd frame
            features = extractor.extract_features(frame)
            features_list.append(features)
        
        frame_count += 1
    
    cap.release()
    return np.array(features_list)


# Quick test
if __name__ == "__main__":
    print(f"Drone-Bird Features: {len(DRONE_BIRD_FEATURES)}")
    for i, feat in enumerate(DRONE_BIRD_FEATURES):
        print(f"  {i+1}. {feat}")
