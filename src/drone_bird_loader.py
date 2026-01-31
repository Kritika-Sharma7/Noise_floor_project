"""
NOISE FLOOR - Drone vs Bird Dataset Loader
============================================
Loads and processes the Drone vs Bird aerial detection dataset.

Dataset Structure:
- bird/: Normal aerial objects (baseline)
- drone/: Anomaly aerial objects (threat)

Use Case: Airspace defense, no-fly zone monitoring, military applications
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DroneBirdLoader:
    """
    Loader for Drone vs Bird dataset.
    
    Birds = Normal behavior (baseline)
    Drones = Anomaly (mechanical, purposeful movement)
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the loader.
        
        Args:
            dataset_path: Path to Drone_vs_Bird folder
        """
        self.dataset_path = Path(dataset_path)
        self.bird_path = self.dataset_path / "bird"
        self.drone_path = self.dataset_path / "drone"
        
        # Validate paths
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"DroneBirdLoader initialized: {dataset_path}")
    
    def get_image_files(self, category: str = "bird", max_files: int = None) -> List[Path]:
        """
        Get list of image files for a category.
        
        Args:
            category: 'bird' or 'drone'
            max_files: Maximum number of files to return
        
        Returns:
            List of image file paths
        """
        if category == "bird":
            folder = self.bird_path
        elif category == "drone":
            folder = self.drone_path
        else:
            raise ValueError(f"Unknown category: {category}")
        
        if not folder.exists():
            logger.warning(f"Folder not found: {folder}")
            return []
        
        # Get all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        files = []
        for ext in extensions:
            files.extend(folder.glob(ext))
        
        files = sorted(files)
        
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Found {len(files)} {category} images")
        return files
    
    def load_images(
        self, 
        category: str = "bird", 
        max_images: int = 100,
        resize: Tuple[int, int] = (320, 240),
        grayscale: bool = True
    ) -> List[np.ndarray]:
        """
        Load images from a category.
        
        Args:
            category: 'bird' or 'drone'
            max_images: Maximum images to load
            resize: Target size (width, height)
            grayscale: Convert to grayscale
        
        Returns:
            List of image arrays
        """
        files = self.get_image_files(category, max_images)
        images = []
        
        for file_path in files:
            try:
                if grayscale:
                    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(str(file_path))
                
                if img is not None:
                    if resize:
                        img = cv2.resize(img, resize)
                    images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(images)} {category} images")
        return images
    
    def load_training_data(
        self, 
        max_images: int = 200,
        resize: Tuple[int, int] = (320, 240)
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load training data: birds as normal, drones as anomaly.
        
        Returns:
            (bird_images, drone_images)
        """
        bird_images = self.load_images("bird", max_images, resize)
        drone_images = self.load_images("drone", max_images, resize)
        
        return bird_images, drone_images
    
    def get_dataset_info(self) -> Dict:
        """Get dataset statistics."""
        bird_files = self.get_image_files("bird")
        drone_files = self.get_image_files("drone")
        
        return {
            "total_images": len(bird_files) + len(drone_files),
            "bird_images": len(bird_files),
            "drone_images": len(drone_files),
            "bird_path": str(self.bird_path),
            "drone_path": str(self.drone_path),
        }


class DroneBirdFeatureExtractor:
    """
    Extract behavioral features from Drone vs Bird images.
    
    Features are designed to distinguish organic (bird) vs mechanical (drone) movement.
    """
    
    def __init__(self, sample_interval: int = 1):
        self.sample_interval = sample_interval
        self.prev_frame = None
        self.frame_count = 0
        self.flow_history = []
        
    def reset(self):
        """Reset extractor state."""
        self.prev_frame = None
        self.frame_count = 0
        self.flow_history = []
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 24 behavioral features from a frame.
        
        These features capture motion patterns that differ between
        organic (bird) and mechanical (drone) movement.
        """
        self.frame_count += 1
        
        # Ensure grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Compute optical flow if we have previous frame
        if self.prev_frame is not None:
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
            
            # Store for temporal analysis
            self.flow_history.append({
                'magnitude': magnitude,
                'angle': angle,
                'mean_mag': np.mean(magnitude),
                'mean_angle': np.mean(angle)
            })
            if len(self.flow_history) > 10:
                self.flow_history.pop(0)
            
            features = self._compute_features(magnitude, angle, gray)
        else:
            # First frame - return baseline features
            features = np.zeros(24)
        
        self.prev_frame = gray.copy()
        return features
    
    def _compute_features(self, magnitude: np.ndarray, angle: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Compute 24 behavioral features."""
        features = np.zeros(24)
        
        # 1-4: Motion Energy Features
        features[0] = np.mean(magnitude)                    # motion_energy
        features[1] = np.std(magnitude)                     # motion_variance
        features[2] = np.max(magnitude)                     # peak_motion
        features[3] = np.sum(magnitude > 1.0) / magnitude.size  # active_ratio
        
        # 5-8: Velocity Features (Drone = constant, Bird = variable)
        features[4] = np.mean(magnitude)                    # velocity_mean
        features[5] = np.std(magnitude)                     # velocity_std (LOW for drone!)
        features[6] = np.max(magnitude) - np.min(magnitude)  # velocity_range
        features[7] = np.median(magnitude)                  # velocity_median
        
        # 9-12: Direction Features (Drone = purposeful, Bird = random)
        angle_flat = angle.flatten()
        angle_hist, _ = np.histogram(angle_flat, bins=8, range=(-np.pi, np.pi))
        angle_hist = angle_hist / (np.sum(angle_hist) + 1e-6)
        features[8] = -np.sum(angle_hist * np.log(angle_hist + 1e-6))  # direction_entropy
        features[9] = np.std(angle_flat)                    # direction_variance
        
        # Dominant direction (drone has consistent direction)
        dominant_bin = np.argmax(angle_hist)
        features[10] = dominant_bin / 8.0                   # dominant_direction
        features[11] = angle_hist[dominant_bin]             # direction_consistency (HIGH for drone!)
        
        # 13-16: Acceleration Features (Drone = smooth, Bird = jerky)
        if len(self.flow_history) >= 2:
            prev_mag = self.flow_history[-2]['mean_mag']
            curr_mag = self.flow_history[-1]['mean_mag']
            features[12] = curr_mag - prev_mag              # acceleration
            features[13] = abs(curr_mag - prev_mag)         # acceleration_magnitude
        else:
            features[12] = 0
            features[13] = 0
        
        # Acceleration variance over time
        if len(self.flow_history) >= 3:
            mags = [f['mean_mag'] for f in self.flow_history]
            accels = np.diff(mags)
            features[14] = np.std(accels)                   # acceleration_variance
            features[15] = np.max(np.abs(accels))           # peak_acceleration
        else:
            features[14] = 0
            features[15] = 0
        
        # 17-20: Spatial Features
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Center vs edge motion (drones often centered in frame)
        center_region = magnitude[cy-h//4:cy+h//4, cx-w//4:cx+w//4]
        edge_region = np.concatenate([
            magnitude[:h//4, :].flatten(),
            magnitude[-h//4:, :].flatten(),
            magnitude[:, :w//4].flatten(),
            magnitude[:, -w//4:].flatten()
        ])
        
        features[16] = np.mean(center_region)               # center_motion
        features[17] = np.mean(edge_region)                 # edge_motion
        features[18] = features[16] / (features[17] + 1e-6)  # center_edge_ratio
        
        # Spatial coherence (drone = coherent, bird = scattered)
        features[19] = np.corrcoef(magnitude[:h//2].flatten(), 
                                   magnitude[h//2:].flatten())[0, 1] if magnitude.size > 0 else 0
        
        # 21-24: Temporal Stability (Drone = stable pattern, Bird = erratic)
        if len(self.flow_history) >= 5:
            recent_mags = [f['mean_mag'] for f in self.flow_history[-5:]]
            recent_angles = [f['mean_angle'] for f in self.flow_history[-5:]]
            
            features[20] = np.std(recent_mags)              # temporal_magnitude_stability
            features[21] = np.std(recent_angles)            # temporal_direction_stability
            features[22] = np.mean(recent_mags)             # sustained_motion
            
            # Pattern regularity (drone = regular, bird = irregular)
            features[23] = 1.0 / (np.std(recent_mags) + 0.1)  # pattern_regularity
        else:
            features[20] = 0
            features[21] = 0
            features[22] = np.mean(magnitude)
            features[23] = 0.5
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return features
    
    def extract_features_from_sequence(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a sequence of images.
        
        Args:
            images: List of grayscale images
        
        Returns:
            Array of shape (n_frames, 24)
        """
        self.reset()
        features_list = []
        
        for img in images:
            features = self.extract_features(img)
            features_list.append(features)
        
        return np.array(features_list)


def create_simulated_sequences(
    loader: DroneBirdLoader,
    n_normal: int = 50,
    n_anomaly: int = 50,
    seq_length: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create simulated video sequences from static images.
    
    Simulates motion by applying transformations to create
    frame-to-frame changes.
    
    Returns:
        (normal_features, anomaly_features, all_features)
    """
    extractor = DroneBirdFeatureExtractor()
    
    # Load images
    bird_images = loader.load_images("bird", max_images=n_normal * 2)
    drone_images = loader.load_images("drone", max_images=n_anomaly * 2)
    
    normal_features = []
    anomaly_features = []
    
    # Process bird images (normal)
    if bird_images:
        extractor.reset()
        for img in bird_images[:n_normal]:
            # Simulate slight motion by adding noise
            for _ in range(seq_length // len(bird_images) + 1):
                noisy = img.copy()
                noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
                noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                features = extractor.extract_features(noisy)
                if np.any(features != 0):  # Skip first frame
                    normal_features.append(features)
    
    # Process drone images (anomaly)
    if drone_images:
        extractor.reset()
        for img in drone_images[:n_anomaly]:
            for _ in range(seq_length // len(drone_images) + 1):
                noisy = img.copy()
                noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
                noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                features = extractor.extract_features(noisy)
                if np.any(features != 0):
                    anomaly_features.append(features)
    
    normal_features = np.array(normal_features) if normal_features else np.zeros((1, 24))
    anomaly_features = np.array(anomaly_features) if anomaly_features else np.zeros((1, 24))
    
    # Combine for timeline
    all_features = np.vstack([normal_features, anomaly_features])
    
    logger.info(f"Created sequences: {len(normal_features)} normal, {len(anomaly_features)} anomaly")
    
    return normal_features, anomaly_features, all_features


# Test function
if __name__ == "__main__":
    # Test loader
    loader = DroneBirdLoader("./data/Drone_vs_Bird")
    info = loader.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Test feature extraction
    bird_images = loader.load_images("bird", max_images=10)
    drone_images = loader.load_images("drone", max_images=10)
    
    extractor = DroneBirdFeatureExtractor()
    
    print("\nBird features (should show organic patterns):")
    bird_features = extractor.extract_features_from_sequence(bird_images)
    print(f"Shape: {bird_features.shape}")
    print(f"Direction entropy mean: {np.mean(bird_features[:, 8]):.3f}")
    print(f"Velocity std mean: {np.mean(bird_features[:, 5]):.3f}")
    
    extractor.reset()
    print("\nDrone features (should show mechanical patterns):")
    drone_features = extractor.extract_features_from_sequence(drone_images)
    print(f"Shape: {drone_features.shape}")
    print(f"Direction entropy mean: {np.mean(drone_features[:, 8]):.3f}")
    print(f"Velocity std mean: {np.mean(drone_features[:, 5]):.3f}")
