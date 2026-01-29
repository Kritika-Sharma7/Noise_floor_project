"""
NOISE FLOOR - Behavioral Feature Extraction Module
===================================================
Advanced motion-based behavioral feature extraction for border surveillance.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

DESIGN PHILOSOPHY:
------------------
We extract BEHAVIORAL features, NOT object detection features.
- Motion energy: Overall activity level
- Optical flow variance: Movement consistency
- Directional consistency: Coordinated vs random movement
- Scene entropy: Visual complexity changes
- Velocity variance: Speed pattern consistency
- Idle/active ratio: Movement patterns over time

NO OBJECT DETECTION:
-------------------
This module intentionally avoids object detection because:
1. Object detection is computationally expensive
2. Behavioral drift can occur without new object types
3. We care about HOW things move, not WHAT they are
4. This enables faster, more explainable inference
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import deque
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# All behavioral features extracted from each temporal window
BEHAVIORAL_FEATURES = [
    # Motion Energy Features
    'motion_energy',            # Total motion in window
    'motion_energy_std',        # Variance of motion over window
    'peak_motion',              # Maximum motion intensity
    
    # Optical Flow Features
    'flow_magnitude_mean',      # Average movement speed
    'flow_magnitude_std',       # Speed consistency
    'flow_magnitude_max',       # Peak speed
    'flow_variance',            # Spatial flow variance
    
    # Directional Features
    'direction_consistency',    # How coordinated is movement
    'direction_entropy',        # Diversity of movement directions
    'dominant_direction',       # Primary movement angle (normalized)
    'direction_change_rate',    # How fast direction changes
    
    # Scene Analysis Features
    'scene_entropy',            # Visual complexity/texture
    'scene_entropy_change',     # Rate of scene change
    'spatial_coherence',        # Spatial organization of motion
    
    # Velocity Features
    'velocity_mean',            # Average velocity magnitude
    'velocity_std',             # Velocity consistency
    'velocity_variance',        # Velocity distribution spread
    'acceleration_mean',        # Average acceleration
    
    # Activity Pattern Features
    'idle_ratio',               # Proportion of low-motion frames
    'active_ratio',             # Proportion of high-motion frames
    'activity_transitions',     # Frequency of idle<->active changes
    
    # Temporal Features
    'temporal_gradient',        # Rate of change over time
    'temporal_stability',       # Consistency of patterns
    'movement_complexity',      # Complexity of movement patterns
]


@dataclass
class BehavioralFeatureVector:
    """
    Container for extracted behavioral features from a temporal window.
    All features are interpretable and documented.
    """
    # Feature values
    features: np.ndarray
    feature_names: List[str] = field(default_factory=lambda: BEHAVIORAL_FEATURES.copy())
    
    # Window metadata
    window_index: int = 0
    start_frame: int = 0
    end_frame: int = 0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    
    # Quality metrics
    confidence: float = 1.0
    valid_frames: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Return feature vector."""
        return self.features
    
    def to_dict(self) -> Dict[str, float]:
        """Return features as dictionary."""
        return {
            name: float(value) 
            for name, value in zip(self.feature_names, self.features)
        }
    
    @property
    def dim(self) -> int:
        """Return feature dimension."""
        return len(self.features)


@dataclass
class FrameMotionState:
    """Intermediate motion state for a single frame."""
    magnitude: np.ndarray
    angle: np.ndarray
    motion_energy: float
    mean_velocity: float
    scene_entropy: float


class BehavioralFeatureExtractor:
    """
    Extracts behavioral features from surveillance video temporal windows.
    
    PROCESSING PIPELINE:
    -------------------
    1. Frame preprocessing (resize, grayscale, denoise)
    2. Optical flow computation (frame-to-frame motion)
    3. Motion statistics aggregation
    4. Directional analysis
    5. Scene entropy computation
    6. Temporal pattern analysis
    
    OUTPUT FORMAT:
    -------------
    [timestamp, f1, f2, ..., fN] per temporal window
    Where each feature has clear physical interpretation.
    """
    
    def __init__(
        self,
        frame_size: Tuple[int, int] = (224, 224),
        num_direction_bins: int = 8,
        motion_threshold: float = 0.5,          # Threshold for "active" motion
        idle_threshold: float = 0.1,            # Threshold for "idle" state
        flow_params: Optional[Dict] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the behavioral feature extractor.
        
        Args:
            frame_size: (width, height) for frame resizing
            num_direction_bins: Number of bins for directional histogram
            motion_threshold: Threshold for classifying "active" motion
            idle_threshold: Threshold for classifying "idle" state
            flow_params: Parameters for optical flow calculation
            use_gpu: Whether to use GPU acceleration (requires CUDA OpenCV)
        """
        self.frame_size = frame_size
        self.num_direction_bins = num_direction_bins
        self.motion_threshold = motion_threshold
        self.idle_threshold = idle_threshold
        self.use_gpu = use_gpu
        
        # Optical flow parameters (Farneback method)
        self.flow_params = flow_params or {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
        }
        
        # State management
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_motion_state: Optional[FrameMotionState] = None
        self.frame_count: int = 0
        
        # Feature statistics for normalization
        self.feature_stats: Optional[Dict] = None
        self._running_stats: Dict[str, deque] = {}
        
        logger.info(f"BehavioralFeatureExtractor initialized")
        logger.info(f"  Frame size: {frame_size}")
        logger.info(f"  Feature dimension: {len(BEHAVIORAL_FEATURES)}")
    
    def reset(self):
        """Reset internal state for new video/stream."""
        self.prev_gray = None
        self.prev_motion_state = None
        self.frame_count = 0
        self._running_stats = {}
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for feature extraction.
        
        Steps:
        1. Resize to standard size (computational efficiency)
        2. Convert to grayscale (motion doesn't need color)
        3. Apply Gaussian blur (noise reduction)
        4. Histogram equalization (lighting normalization)
        """
        # Handle different input formats
        if frame is None:
            raise ValueError("Frame cannot be None")
        
        # Resize
        if frame.shape[:2] != self.frame_size[::-1]:
            resized = cv2.resize(frame, self.frame_size)
        else:
            resized = frame
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Histogram equalization for lighting normalization
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def compute_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dense optical flow between two frames.
        
        Returns magnitude (speed) and angle (direction) for each pixel.
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
    
    def compute_scene_entropy(self, frame: np.ndarray) -> float:
        """
        Compute scene entropy (visual complexity).
        
        Higher entropy = more complex/textured scene
        Lower entropy = more uniform/simple scene
        """
        # Compute histogram
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Normalize to probability distribution
        hist = hist / (hist.sum() + 1e-10)
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize to [0, 1]
        max_entropy = np.log2(256)
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)
    
    def compute_direction_consistency(
        self,
        angle: np.ndarray,
        magnitude: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Compute directional consistency (coordination of movement).
        
        High consistency = movement in similar directions (coordinated)
        Low consistency = movement in diverse directions (random/chaotic)
        """
        # Mask for significant motion
        motion_mask = magnitude > threshold
        
        if not np.any(motion_mask):
            return 1.0  # No motion = perfectly consistent
        
        valid_angles = angle[motion_mask]
        valid_magnitudes = magnitude[motion_mask]
        
        # Compute circular variance (inverse of consistency)
        # Using weighted circular statistics
        weights = valid_magnitudes / (valid_magnitudes.sum() + 1e-10)
        
        # Mean resultant length (R)
        cos_mean = np.sum(weights * np.cos(valid_angles))
        sin_mean = np.sum(weights * np.sin(valid_angles))
        R = np.sqrt(cos_mean**2 + sin_mean**2)
        
        # R ranges from 0 (uniform distribution) to 1 (all same direction)
        return float(R)
    
    def compute_direction_histogram(
        self,
        angle: np.ndarray,
        magnitude: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute histogram of motion directions.
        
        Returns:
            histogram: Normalized direction histogram
            entropy: Directional entropy
            dominant_direction: Primary movement direction (radians)
        """
        motion_mask = magnitude > threshold
        
        if not np.any(motion_mask):
            return (
                np.zeros(self.num_direction_bins),
                0.0,
                0.0
            )
        
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
        
        # Compute entropy
        hist_nonzero = hist[hist > 0]
        if len(hist_nonzero) > 0:
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
            max_entropy = np.log2(self.num_direction_bins)
            entropy = entropy / max_entropy
        else:
            entropy = 0.0
        
        # Find dominant direction
        dominant_bin = np.argmax(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        dominant_direction = bin_centers[dominant_bin] / (2 * np.pi)  # Normalize to [0,1]
        
        return hist, float(entropy), float(dominant_direction)
    
    def compute_spatial_coherence(
        self,
        magnitude: np.ndarray,
        angle: np.ndarray
    ) -> float:
        """
        Compute spatial coherence of motion field.
        
        High coherence = smooth, organized motion patterns
        Low coherence = fragmented, chaotic motion
        """
        # Compute local variance of motion vectors
        # Using Sobel gradients on magnitude field
        grad_x = cv2.Sobel(magnitude.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(magnitude.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Coherence inversely related to gradient magnitude
        mean_grad = np.mean(grad_mag)
        coherence = 1.0 / (1.0 + mean_grad)
        
        return float(coherence)
    
    def extract_frame_motion_state(
        self,
        curr_gray: np.ndarray
    ) -> Optional[FrameMotionState]:
        """
        Extract motion state from a single frame.
        
        Returns None for first frame (needs previous frame for flow).
        """
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return None
        
        # Compute optical flow
        magnitude, angle = self.compute_optical_flow(self.prev_gray, curr_gray)
        
        # Compute frame-level features
        motion_energy = float(np.sum(magnitude))
        mean_velocity = float(np.mean(magnitude))
        scene_entropy = self.compute_scene_entropy(curr_gray)
        
        # Update state
        self.prev_gray = curr_gray
        
        return FrameMotionState(
            magnitude=magnitude,
            angle=angle,
            motion_energy=motion_energy,
            mean_velocity=mean_velocity,
            scene_entropy=scene_entropy
        )
    
    def extract_window_features(
        self,
        frames: np.ndarray,
        window_index: int = 0,
        start_frame: int = 0,
        start_timestamp: float = 0.0
    ) -> BehavioralFeatureVector:
        """
        Extract behavioral features from a temporal window of frames.
        
        This is the main extraction method for LSTM-VAE input.
        
        Args:
            frames: Array of frames (T, H, W) or (T, H, W, C)
            window_index: Index of this window in sequence
            start_frame: Global frame index of window start
            start_timestamp: Timestamp of window start
            
        Returns:
            BehavioralFeatureVector with all extracted features
        """
        T = frames.shape[0]
        
        # Collect frame-level statistics
        motion_energies = []
        velocities = []
        scene_entropies = []
        direction_consistencies = []
        direction_entropies = []
        dominant_directions = []
        spatial_coherences = []
        
        prev_velocity = None
        accelerations = []
        
        idle_count = 0
        active_count = 0
        prev_activity_state = None
        activity_transitions = 0
        
        # Reset for this window
        self.reset()
        
        valid_frames = 0
        
        for i in range(T):
            frame = frames[i]
            
            # Preprocess
            gray = self.preprocess_frame(frame)
            
            # Extract motion state
            motion_state = self.extract_frame_motion_state(gray)
            
            if motion_state is None:
                continue
            
            valid_frames += 1
            
            # Collect statistics
            motion_energies.append(motion_state.motion_energy)
            velocities.append(motion_state.mean_velocity)
            scene_entropies.append(motion_state.scene_entropy)
            
            # Directional analysis
            consistency = self.compute_direction_consistency(
                motion_state.angle, motion_state.magnitude
            )
            direction_consistencies.append(consistency)
            
            _, dir_entropy, dom_dir = self.compute_direction_histogram(
                motion_state.angle, motion_state.magnitude
            )
            direction_entropies.append(dir_entropy)
            dominant_directions.append(dom_dir)
            
            # Spatial coherence
            coherence = self.compute_spatial_coherence(
                motion_state.magnitude, motion_state.angle
            )
            spatial_coherences.append(coherence)
            
            # Acceleration
            if prev_velocity is not None:
                acceleration = motion_state.mean_velocity - prev_velocity
                accelerations.append(acceleration)
            prev_velocity = motion_state.mean_velocity
            
            # Activity state
            current_activity = motion_state.mean_velocity > self.motion_threshold
            if motion_state.mean_velocity < self.idle_threshold:
                idle_count += 1
            elif current_activity:
                active_count += 1
            
            if prev_activity_state is not None and current_activity != prev_activity_state:
                activity_transitions += 1
            prev_activity_state = current_activity
        
        # Handle edge case of no valid frames
        if valid_frames < 2:
            return BehavioralFeatureVector(
                features=np.zeros(len(BEHAVIORAL_FEATURES)),
                window_index=window_index,
                start_frame=start_frame,
                end_frame=start_frame + T - 1,
                start_timestamp=start_timestamp,
                end_timestamp=start_timestamp + T / 10.0,
                confidence=0.0,
                valid_frames=valid_frames
            )
        
        # Convert to numpy
        motion_energies = np.array(motion_energies)
        velocities = np.array(velocities)
        scene_entropies = np.array(scene_entropies)
        direction_consistencies = np.array(direction_consistencies)
        direction_entropies = np.array(direction_entropies)
        dominant_directions = np.array(dominant_directions)
        spatial_coherences = np.array(spatial_coherences)
        accelerations = np.array(accelerations) if accelerations else np.array([0.0])
        
        # =================================================================
        # COMPUTE FINAL FEATURES
        # =================================================================
        
        features = []
        
        # Motion Energy Features
        features.append(np.mean(motion_energies))           # motion_energy
        features.append(np.std(motion_energies))            # motion_energy_std
        features.append(np.max(motion_energies))            # peak_motion
        
        # Optical Flow Features
        features.append(np.mean(velocities))                # flow_magnitude_mean
        features.append(np.std(velocities))                 # flow_magnitude_std
        features.append(np.max(velocities))                 # flow_magnitude_max
        features.append(np.var(velocities))                 # flow_variance
        
        # Directional Features
        features.append(np.mean(direction_consistencies))   # direction_consistency
        features.append(np.mean(direction_entropies))       # direction_entropy
        features.append(np.mean(dominant_directions))       # dominant_direction
        
        # Direction change rate
        if len(dominant_directions) > 1:
            dir_changes = np.abs(np.diff(dominant_directions))
            features.append(np.mean(dir_changes))           # direction_change_rate
        else:
            features.append(0.0)
        
        # Scene Analysis Features
        features.append(np.mean(scene_entropies))           # scene_entropy
        
        # Scene entropy change
        if len(scene_entropies) > 1:
            entropy_changes = np.abs(np.diff(scene_entropies))
            features.append(np.mean(entropy_changes))       # scene_entropy_change
        else:
            features.append(0.0)
        
        features.append(np.mean(spatial_coherences))        # spatial_coherence
        
        # Velocity Features
        features.append(np.mean(velocities))                # velocity_mean
        features.append(np.std(velocities))                 # velocity_std
        features.append(np.var(velocities))                 # velocity_variance
        features.append(np.mean(accelerations))             # acceleration_mean
        
        # Activity Pattern Features
        total_frames = max(valid_frames, 1)
        features.append(idle_count / total_frames)          # idle_ratio
        features.append(active_count / total_frames)        # active_ratio
        features.append(activity_transitions / total_frames) # activity_transitions
        
        # Temporal Features
        if len(motion_energies) > 1:
            temporal_gradient = np.mean(np.abs(np.diff(motion_energies)))
        else:
            temporal_gradient = 0.0
        features.append(temporal_gradient)                  # temporal_gradient
        
        # Temporal stability (inverse of coefficient of variation)
        cv = np.std(motion_energies) / (np.mean(motion_energies) + 1e-10)
        features.append(1.0 / (1.0 + cv))                   # temporal_stability
        
        # Movement complexity (based on entropy of motion patterns)
        velocity_hist, _ = np.histogram(velocities, bins=10, density=True)
        velocity_hist = velocity_hist[velocity_hist > 0]
        if len(velocity_hist) > 0:
            complexity = -np.sum(velocity_hist * np.log(velocity_hist + 1e-10))
            complexity = complexity / np.log(10)  # Normalize
        else:
            complexity = 0.0
        features.append(complexity)                         # movement_complexity
        
        # Create feature vector
        feature_array = np.array(features, dtype=np.float32)
        
        # Compute confidence based on valid frame ratio
        confidence = valid_frames / T
        
        return BehavioralFeatureVector(
            features=feature_array,
            window_index=window_index,
            start_frame=start_frame,
            end_frame=start_frame + T - 1,
            start_timestamp=start_timestamp,
            end_timestamp=start_timestamp + T / 10.0,
            confidence=confidence,
            valid_frames=valid_frames
        )
    
    def extract_from_video(
        self,
        video_path: str,
        window_size: int = 30,
        stride: int = 10,
        max_windows: Optional[int] = None,
        progress_callback=None
    ) -> List[BehavioralFeatureVector]:
        """
        Extract behavioral features from a video file.
        
        Args:
            video_path: Path to video file
            window_size: Number of frames per window
            stride: Stride between windows
            max_windows: Maximum windows to extract
            progress_callback: Optional progress callback
            
        Returns:
            List of BehavioralFeatureVector objects
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
        
        # Buffer for window frames
        frame_buffer = []
        frame_indices = []
        
        frame_idx = 0
        window_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            frame_indices.append(frame_idx)
            
            # Process window when buffer is full
            if len(frame_buffer) >= window_size:
                frames_array = np.stack(frame_buffer[:window_size])
                
                features = self.extract_window_features(
                    frames=frames_array,
                    window_index=window_idx,
                    start_frame=frame_indices[0],
                    start_timestamp=frame_indices[0] / fps
                )
                
                features_list.append(features)
                window_idx += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(window_idx)
                
                # Check max windows
                if max_windows and window_idx >= max_windows:
                    break
                
                # Slide window
                frame_buffer = frame_buffer[stride:]
                frame_indices = frame_indices[stride:]
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(features_list)} feature windows")
        
        return features_list
    
    def features_to_matrix(
        self,
        features_list: List[BehavioralFeatureVector]
    ) -> np.ndarray:
        """
        Convert list of feature vectors to matrix.
        
        Returns:
            Feature matrix of shape (N, F)
        """
        if not features_list:
            return np.array([])
        
        return np.stack([f.features for f in features_list])
    
    def features_to_sequence(
        self,
        features_list: List[BehavioralFeatureVector],
        sequence_length: int = 10
    ) -> np.ndarray:
        """
        Convert features to sequence format for LSTM-VAE.
        
        Returns:
            Sequence tensor of shape (N, T, F)
        """
        matrix = self.features_to_matrix(features_list)
        
        if len(matrix) < sequence_length:
            # Pad if needed
            padding = np.zeros((sequence_length - len(matrix), matrix.shape[1]))
            matrix = np.vstack([padding, matrix])
        
        # Create sliding sequences
        sequences = []
        for i in range(len(matrix) - sequence_length + 1):
            seq = matrix[i:i + sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)


# =============================================================================
# SYNTHETIC DATA GENERATION (FOR DEMO/TESTING)
# =============================================================================

def create_synthetic_normal_data(
    num_samples: int = 500,
    feature_dim: int = len(BEHAVIORAL_FEATURES),
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic normal behavioral data.
    
    Creates realistic-looking behavioral patterns that simulate
    normal border surveillance activity.
    """
    np.random.seed(seed)
    
    # Base patterns for each feature group
    data = []
    
    for i in range(num_samples):
        sample = []
        
        # Motion Energy Features (typically low for normal)
        sample.append(np.random.exponential(0.5))           # motion_energy
        sample.append(np.random.exponential(0.2))           # motion_energy_std
        sample.append(np.random.exponential(0.8))           # peak_motion
        
        # Optical Flow Features (moderate, consistent)
        sample.append(np.random.normal(0.3, 0.1))           # flow_magnitude_mean
        sample.append(np.random.exponential(0.1))           # flow_magnitude_std
        sample.append(np.random.exponential(0.5))           # flow_magnitude_max
        sample.append(np.random.exponential(0.05))          # flow_variance
        
        # Directional Features (high consistency for normal)
        sample.append(np.random.beta(5, 2))                 # direction_consistency
        sample.append(np.random.beta(2, 5))                 # direction_entropy
        sample.append(np.random.uniform(0, 1))              # dominant_direction
        sample.append(np.random.exponential(0.1))           # direction_change_rate
        
        # Scene Analysis Features
        sample.append(np.random.beta(5, 2))                 # scene_entropy
        sample.append(np.random.exponential(0.05))          # scene_entropy_change
        sample.append(np.random.beta(5, 2))                 # spatial_coherence
        
        # Velocity Features
        sample.append(np.random.normal(0.3, 0.1))           # velocity_mean
        sample.append(np.random.exponential(0.1))           # velocity_std
        sample.append(np.random.exponential(0.05))          # velocity_variance
        sample.append(np.random.normal(0, 0.05))            # acceleration_mean
        
        # Activity Pattern Features
        sample.append(np.random.beta(5, 2))                 # idle_ratio
        sample.append(np.random.beta(2, 5))                 # active_ratio
        sample.append(np.random.exponential(0.1))           # activity_transitions
        
        # Temporal Features
        sample.append(np.random.exponential(0.1))           # temporal_gradient
        sample.append(np.random.beta(5, 2))                 # temporal_stability
        sample.append(np.random.beta(3, 3))                 # movement_complexity
        
        data.append(sample)
    
    return np.array(data)[:, :feature_dim]


def create_synthetic_drift_data(
    num_samples: int = 200,
    feature_dim: int = len(BEHAVIORAL_FEATURES),
    drift_rate: float = 0.02,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic data with gradual behavioral drift.
    
    Simulates the gradual emergence of threatening behavior patterns
    in border surveillance scenarios.
    """
    np.random.seed(seed)
    
    # Start with normal data
    base_data = create_synthetic_normal_data(num_samples, feature_dim, seed)
    
    # Apply gradual drift
    drift_data = base_data.copy()
    
    for i in range(num_samples):
        drift_factor = 1 + (i * drift_rate)
        
        # Increase motion energy (more activity)
        drift_data[i, 0] *= drift_factor
        drift_data[i, 2] *= drift_factor
        
        # Decrease direction consistency (less organized movement)
        drift_data[i, 7] /= drift_factor
        
        # Increase direction entropy (more chaotic)
        drift_data[i, 8] = min(1.0, drift_data[i, 8] * drift_factor)
        
        # Decrease temporal stability
        drift_data[i, 21] /= drift_factor
        
        # Increase movement complexity
        drift_data[i, 22] = min(1.0, drift_data[i, 22] * drift_factor)
        
        # Decrease idle ratio, increase active ratio
        drift_data[i, 17] /= drift_factor
        drift_data[i, 18] = min(1.0, drift_data[i, 18] * drift_factor)
    
    return drift_data


def get_feature_names() -> List[str]:
    """Return list of all behavioral feature names."""
    return BEHAVIORAL_FEATURES.copy()


def get_feature_descriptions() -> Dict[str, str]:
    """Return descriptions for each feature."""
    return {
        'motion_energy': 'Total motion activity in the window',
        'motion_energy_std': 'Variance of motion over the window',
        'peak_motion': 'Maximum motion intensity observed',
        'flow_magnitude_mean': 'Average movement speed across pixels',
        'flow_magnitude_std': 'Consistency of movement speeds',
        'flow_magnitude_max': 'Peak movement speed observed',
        'flow_variance': 'Spatial variance in movement patterns',
        'direction_consistency': 'How coordinated/aligned is movement',
        'direction_entropy': 'Diversity of movement directions',
        'dominant_direction': 'Primary movement direction',
        'direction_change_rate': 'How fast movement direction changes',
        'scene_entropy': 'Visual complexity/texture of scene',
        'scene_entropy_change': 'Rate of scene appearance change',
        'spatial_coherence': 'Spatial organization of motion',
        'velocity_mean': 'Average velocity magnitude',
        'velocity_std': 'Velocity consistency',
        'velocity_variance': 'Spread of velocity distribution',
        'acceleration_mean': 'Average acceleration',
        'idle_ratio': 'Proportion of low-motion frames',
        'active_ratio': 'Proportion of high-motion frames',
        'activity_transitions': 'Frequency of idle/active changes',
        'temporal_gradient': 'Rate of change over time',
        'temporal_stability': 'Consistency of patterns over time',
        'movement_complexity': 'Complexity of movement patterns',
    }
