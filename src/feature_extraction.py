"""
NOISE FLOOR - Feature Extraction Module
========================================
Extracts behavioral (motion-based) features from video frames.

GRAY-BOX EXPLANATION:
---------------------
This module converts raw video frames into behavioral feature vectors.
We focus on MOTION features because:
1. Motion represents behavior, not appearance
2. Behavioral changes indicate operational drift
3. Motion features are interpretable and explainable

Feature Types Extracted:
1. Optical Flow Magnitude - Speed of movement
2. Motion Variance - Consistency of movement patterns
3. Directional Histogram - Distribution of movement directions
4. Motion Energy - Total activity level
5. Temporal Gradient - Rate of change in motion
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameFeatures:
    """
    Container for extracted features from a single frame.
    All features are interpretable and documented.
    """
    # Primary motion features
    flow_magnitude_mean: float      # Average speed of motion
    flow_magnitude_std: float       # Variation in motion speed
    flow_magnitude_max: float       # Peak motion speed
    
    # Directional features (8-bin histogram)
    direction_histogram: np.ndarray  # Distribution of motion directions
    direction_entropy: float         # Diversity of movement directions
    
    # Motion energy features
    motion_energy: float            # Total motion in frame
    motion_variance: float          # Spatial variance of motion
    
    # Temporal features (requires previous frame)
    temporal_gradient: float        # Rate of change from previous frame
    
    # Frame metadata
    frame_index: int
    timestamp: float
    
    def to_vector(self) -> np.ndarray:
        """Convert features to fixed-size vector for model input."""
        return np.concatenate([
            [self.flow_magnitude_mean],
            [self.flow_magnitude_std],
            [self.flow_magnitude_max],
            self.direction_histogram,
            [self.direction_entropy],
            [self.motion_energy],
            [self.motion_variance],
            [self.temporal_gradient],
        ])


class FeatureExtractor:
    """
    Extracts behavioral features from video frames using optical flow.
    
    GRAY-BOX DESIGN:
    ----------------
    - All parameters are configurable and documented
    - Feature extraction logic is transparent
    - Each feature has clear physical interpretation
    """
    
    def __init__(
        self,
        frame_size: Tuple[int, int] = (224, 224),
        num_direction_bins: int = 8,
        flow_params: Optional[Dict] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            frame_size: (width, height) for frame resizing
            num_direction_bins: Number of bins for directional histogram
            flow_params: Parameters for optical flow calculation
        """
        self.frame_size = frame_size
        self.num_direction_bins = num_direction_bins
        
        # Default optical flow parameters (Farneback method)
        self.flow_params = flow_params or {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
        }
        
        # State for temporal features
        self.prev_gray = None
        self.prev_magnitude = None
        self.frame_count = 0
        
        # Feature statistics for normalization
        self.feature_stats = None
        
        logger.info(f"FeatureExtractor initialized with frame_size={frame_size}")
    
    def reset(self):
        """Reset internal state for new video."""
        self.prev_gray = None
        self.prev_magnitude = None
        self.frame_count = 0
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for feature extraction.
        
        Steps:
        1. Resize to standard size (reduces computation)
        2. Convert to grayscale (motion doesn't need color)
        3. Apply Gaussian blur (reduces noise)
        """
        # Resize
        resized = cv2.resize(frame, self.frame_size)
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def compute_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dense optical flow between two frames.
        
        EXPLANATION:
        ------------
        Optical flow estimates the motion of each pixel between frames.
        - Returns flow vectors (dx, dy) for each pixel
        - We convert to polar coordinates: magnitude (speed) and angle (direction)
        
        This is the foundation of our behavioral feature extraction.
        """
        # Compute dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            self.flow_params['pyr_scale'],
            self.flow_params['levels'],
            self.flow_params['winsize'],
            self.flow_params['iterations'],
            self.flow_params['poly_n'],
            self.flow_params['poly_sigma'],
            0
        )
        
        # Convert to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return magnitude, angle
    
    def compute_direction_histogram(
        self,
        angle: np.ndarray,
        magnitude: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Compute histogram of motion directions.
        
        EXPLANATION:
        ------------
        - Divides 360° into bins (default 8 = N, NE, E, SE, S, SW, W, NW)
        - Weights each pixel by its motion magnitude
        - Only considers pixels with significant motion (above threshold)
        
        This captures the DISTRIBUTION of movement directions.
        - Uniform distribution = random/chaotic movement
        - Peaked distribution = organized/directional movement
        """
        # Create mask for significant motion
        motion_mask = magnitude > threshold
        
        if not np.any(motion_mask):
            return np.zeros(self.num_direction_bins)
        
        # Get angles and magnitudes for moving pixels
        valid_angles = angle[motion_mask]
        valid_magnitudes = magnitude[motion_mask]
        
        # Normalize angles to [0, 2π]
        valid_angles = valid_angles % (2 * np.pi)
        
        # Compute weighted histogram
        bin_edges = np.linspace(0, 2 * np.pi, self.num_direction_bins + 1)
        hist, _ = np.histogram(
            valid_angles,
            bins=bin_edges,
            weights=valid_magnitudes
        )
        
        # Normalize histogram
        total = hist.sum()
        if total > 0:
            hist = hist / total
        
        return hist
    
    def compute_direction_entropy(self, histogram: np.ndarray) -> float:
        """
        Compute entropy of direction histogram.
        
        EXPLANATION:
        ------------
        Entropy measures the "randomness" of the distribution.
        - High entropy = movement in many directions (chaotic)
        - Low entropy = movement concentrated in few directions (organized)
        
        Formula: H = -Σ p(x) * log(p(x))
        Normalized to [0, 1] range.
        """
        # Remove zeros to avoid log(0)
        hist = histogram[histogram > 0]
        
        if len(hist) == 0:
            return 0.0
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(self.num_direction_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def extract_features(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> Optional[FrameFeatures]:
        """
        Extract all behavioral features from a single frame.
        
        Returns None for the first frame (needs previous frame for flow).
        
        FEATURE SUMMARY:
        ----------------
        1. flow_magnitude_mean: Average motion speed
        2. flow_magnitude_std: Variation in motion speed
        3. flow_magnitude_max: Peak motion speed
        4. direction_histogram: Distribution of movement directions
        5. direction_entropy: Diversity of movement directions
        6. motion_energy: Total motion in frame
        7. motion_variance: Spatial variance of motion
        8. temporal_gradient: Rate of change from previous frame
        """
        # Preprocess frame
        curr_gray = self.preprocess_frame(frame)
        
        # First frame: store and return None
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.frame_count = 0
            return None
        
        # Compute optical flow
        magnitude, angle = self.compute_optical_flow(self.prev_gray, curr_gray)
        
        # === Extract Features ===
        
        # 1. Magnitude statistics (motion speed)
        flow_magnitude_mean = float(np.mean(magnitude))
        flow_magnitude_std = float(np.std(magnitude))
        flow_magnitude_max = float(np.max(magnitude))
        
        # 2. Direction histogram and entropy
        direction_histogram = self.compute_direction_histogram(angle, magnitude)
        direction_entropy = self.compute_direction_entropy(direction_histogram)
        
        # 3. Motion energy (total motion)
        motion_energy = float(np.sum(magnitude))
        
        # 4. Motion variance (spatial distribution)
        motion_variance = float(np.var(magnitude))
        
        # 5. Temporal gradient (change from previous frame)
        if self.prev_magnitude is not None:
            temporal_gradient = float(np.mean(np.abs(magnitude - self.prev_magnitude)))
        else:
            temporal_gradient = 0.0
        
        # Update state
        self.prev_gray = curr_gray
        self.prev_magnitude = magnitude.copy()
        self.frame_count += 1
        
        # Create feature container
        features = FrameFeatures(
            flow_magnitude_mean=flow_magnitude_mean,
            flow_magnitude_std=flow_magnitude_std,
            flow_magnitude_max=flow_magnitude_max,
            direction_histogram=direction_histogram,
            direction_entropy=direction_entropy,
            motion_energy=motion_energy,
            motion_variance=motion_variance,
            temporal_gradient=temporal_gradient,
            frame_index=self.frame_count,
            timestamp=timestamp,
        )
        
        return features
    
    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
        progress_callback=None
    ) -> List[FrameFeatures]:
        """
        Extract features from all frames in a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            frame_skip: Process every Nth frame
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of FrameFeatures objects
        """
        self.reset()
        features_list = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Total frames: {total_frames}")
        
        frame_idx = 0
        processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Extract features
            timestamp = frame_idx / fps if fps > 0 else frame_idx
            features = self.extract_features(frame, timestamp)
            
            if features is not None:
                features_list.append(features)
            
            processed += 1
            
            # Progress callback
            if progress_callback and processed % 50 == 0:
                progress_callback(processed, total_frames // frame_skip)
            
            # Check max frames
            if max_frames and processed >= max_frames:
                break
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted features from {len(features_list)} frames")
        
        return features_list
    
    def extract_from_frames(
        self,
        frames_dir: str,
        pattern: str = "*.tif",
        max_frames: Optional[int] = None
    ) -> List[FrameFeatures]:
        """
        Extract features from a directory of frame images.
        Useful for datasets like UCSD that provide frames as images.
        """
        self.reset()
        features_list = []
        
        frames_path = Path(frames_dir)
        frame_files = sorted(frames_path.glob(pattern))
        
        if not frame_files:
            # Try other common patterns
            for alt_pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                frame_files = sorted(frames_path.glob(alt_pattern))
                if frame_files:
                    break
        
        if not frame_files:
            raise ValueError(f"No frame files found in {frames_dir}")
        
        logger.info(f"Found {len(frame_files)} frames in {frames_dir}")
        
        for idx, frame_file in enumerate(frame_files):
            if max_frames and idx >= max_frames:
                break
            
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            features = self.extract_features(frame, timestamp=idx / 30.0)
            if features is not None:
                features_list.append(features)
        
        logger.info(f"Extracted features from {len(features_list)} frames")
        return features_list
    
    def features_to_matrix(
        self,
        features_list: List[FrameFeatures]
    ) -> np.ndarray:
        """Convert list of features to numpy matrix for model input."""
        if not features_list:
            return np.array([])
        
        vectors = [f.to_vector() for f in features_list]
        return np.array(vectors)
    
    def normalize_features(
        self,
        features_matrix: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """
        Normalize features to zero mean and unit variance.
        
        EXPLANATION:
        ------------
        Normalization is crucial for the autoencoder:
        - Ensures all features contribute equally
        - Improves training stability
        - Makes reconstruction error comparable across features
        
        During training (fit=True): Compute and store statistics
        During inference (fit=False): Use stored statistics
        """
        if fit or self.feature_stats is None:
            mean = np.mean(features_matrix, axis=0)
            std = np.std(features_matrix, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            self.feature_stats = {'mean': mean, 'std': std}
        
        normalized = (features_matrix - self.feature_stats['mean']) / self.feature_stats['std']
        return normalized
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for explainability."""
        names = [
            'flow_magnitude_mean',
            'flow_magnitude_std',
            'flow_magnitude_max',
        ]
        names.extend([f'direction_bin_{i}' for i in range(self.num_direction_bins)])
        names.extend([
            'direction_entropy',
            'motion_energy',
            'motion_variance',
            'temporal_gradient',
        ])
        return names


def create_synthetic_normal_data(
    num_samples: int = 1000,
    feature_dim: int = 14,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Create synthetic "normal" behavioral data for testing.
    
    This simulates typical pedestrian motion patterns:
    - Consistent average motion speed
    - Low variance in motion patterns
    - Dominant direction (e.g., people walking in similar directions)
    
    All values are normalized to similar scales for stable training.
    """
    np.random.seed(42)
    
    # Base normal patterns (all normalized to ~0-1 range)
    base_pattern = np.array([
        0.5,   # flow_magnitude_mean (moderate speed)
        0.1,   # flow_magnitude_std (consistent speed)
        0.8,   # flow_magnitude_max (occasional faster motion)
        0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.3,  # direction histogram (dominant directions)
        0.5,   # direction_entropy (some diversity)
        0.5,   # motion_energy (normalized)
        0.05,  # motion_variance (consistent spatial distribution)
        0.1,   # temporal_gradient (smooth changes)
    ])
    
    # Generate samples with noise
    samples = np.tile(base_pattern, (num_samples, 1))
    noise = np.random.normal(0, noise_level, samples.shape)
    samples = samples + noise
    
    # Clip to reasonable range
    samples = np.clip(samples, 0, 2)
    
    # Add temporal correlation
    for i in range(1, num_samples):
        samples[i] = 0.7 * samples[i] + 0.3 * samples[i-1]
    
    return samples


def create_synthetic_drift_data(
    num_samples: int = 500,
    feature_dim: int = 15,
    drift_rate: float = 0.01
) -> np.ndarray:
    """
    Create synthetic data with gradual behavioral drift.
    
    This simulates a slow change in behavior:
    - Gradually increasing motion speed
    - Increasing variance (less consistent)
    - Changing directional patterns
    """
    np.random.seed(43)
    
    # Start with normal pattern
    normal = create_synthetic_normal_data(1, feature_dim)[0]
    
    samples = []
    current = normal.copy()
    
    for i in range(num_samples):
        # Gradual drift in key features (capped to prevent overflow)
        drift_factor = min(3.0, 1 + drift_rate * i)
        
        # Increase motion magnitude (people moving faster)
        current[0] = min(2.0, normal[0] * drift_factor)
        current[1] = min(1.0, normal[1] * (drift_factor ** 1.5))
        
        # Increase entropy (more chaotic movement)
        current[11] = min(1.0, normal[11] * drift_factor)
        
        # Add noise
        noise = np.random.normal(0, 0.02, current.shape)
        sample = np.clip(current + noise, 0, 3)
        samples.append(sample.copy())
    
    return np.array(samples)


if __name__ == "__main__":
    # Test feature extraction
    print("Testing Feature Extraction Module")
    print("=" * 50)
    
    # Create synthetic data
    normal_data = create_synthetic_normal_data(100)
    drift_data = create_synthetic_drift_data(100)
    
    print(f"Normal data shape: {normal_data.shape}")
    print(f"Drift data shape: {drift_data.shape}")
    
    # Show feature names
    extractor = FeatureExtractor()
    print(f"\nFeature names: {extractor.get_feature_names()}")
    
    print("\nFeature extraction module ready!")
