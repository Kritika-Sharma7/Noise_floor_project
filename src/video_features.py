"""
NOISE FLOOR - Real Video Feature Extraction Module
====================================================
Optical flow-based behavioral feature extraction from surveillance video.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

DESIGN PHILOSOPHY:
------------------
"The intelligence pipeline is feature-driven, not data-source-driven."

This module extracts the SAME 24 behavioral features from real video that
are used in synthetic data, ensuring the LSTM-VAE model can process both
without modification.

FEATURE EXTRACTION PIPELINE:
----------------------------
1. Frame acquisition (fixed interval sampling)
2. Preprocessing (resize, grayscale, denoise)
3. Optical flow computation (Farneback method)
4. Motion statistics aggregation
5. Directional analysis
6. Scene entropy computation
7. Temporal pattern analysis

PUBLIC DATASET USAGE:
---------------------
Public surveillance datasets (e.g., UCSD Anomaly Detection) are used as
proxy for border CCTV footage. Labels are intentionally ignored - we train
only on NORMAL behavior patterns.

TECHNOLOGY READINESS LEVEL: TRL-4
Lab-validated prototype for decision-support intelligence.
This is NOT an autonomous system - AI assists operators, it does NOT replace them.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Generator, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import deque
from scipy import stats
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import feature names from behavioral_features for consistency
try:
    from src.behavioral_features import BEHAVIORAL_FEATURES
except ImportError:
    BEHAVIORAL_FEATURES = [
        'motion_energy', 'motion_energy_std', 'peak_motion',
        'flow_magnitude_mean', 'flow_magnitude_std', 'flow_magnitude_max', 'flow_variance',
        'direction_consistency', 'direction_entropy', 'dominant_direction', 'direction_change_rate',
        'scene_entropy', 'scene_entropy_change', 'spatial_coherence',
        'velocity_mean', 'velocity_std', 'velocity_variance', 'acceleration_mean',
        'idle_ratio', 'active_ratio', 'activity_transitions',
        'temporal_gradient', 'temporal_stability', 'movement_complexity',
    ]


@dataclass
class VideoMetadata:
    """Metadata about the video source."""
    path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    source_type: str = "surveillance"  # surveillance, ucsd, synthetic


@dataclass
class FrameFeatures:
    """Features extracted from a single frame or frame pair."""
    frame_index: int
    timestamp: float
    
    # Raw optical flow statistics
    flow_magnitude: np.ndarray
    flow_angle: np.ndarray
    
    # Computed metrics
    motion_energy: float
    mean_velocity: float
    scene_entropy: float
    
    # Valid flag
    is_valid: bool = True


class OpticalFlowExtractor:
    """
    Farneback optical flow extraction with robust parameters.
    
    Optical flow captures HOW things move, not WHAT they are.
    This is the foundation of behavioral feature extraction.
    """
    
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        
        self.prev_gray = None
        
    def reset(self):
        """Reset state for new video."""
        self.prev_gray = None
    
    def compute(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow between current and previous frame.
        
        Returns:
            Tuple of (magnitude, angle) arrays
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray, dtype=np.float32), np.zeros_like(gray, dtype=np.float32)
        
        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, 0
        )
        
        # Convert to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        self.prev_gray = gray
        
        return magnitude, angle


class RealVideoFeatureExtractor:
    """
    Extracts 24 behavioral features from real surveillance video.
    
    CRITICAL: Output format matches synthetic data exactly.
    This allows the LSTM-VAE to process real and synthetic data identically.
    
    "The intelligence pipeline is feature-driven, not data-source-driven."
    """
    
    def __init__(
        self,
        frame_size: Tuple[int, int] = (224, 224),
        sample_interval: int = 5,          # Sample every Nth frame
        temporal_window: int = 10,          # Frames for temporal features
        motion_threshold: float = 1.0,      # Threshold for "active" motion
        idle_threshold: float = 0.5,        # Threshold for "idle" motion
    ):
        self.frame_size = frame_size
        self.sample_interval = sample_interval
        self.temporal_window = temporal_window
        self.motion_threshold = motion_threshold
        self.idle_threshold = idle_threshold
        
        self.flow_extractor = OpticalFlowExtractor()
        
        # Temporal buffers
        self.motion_history: deque = deque(maxlen=temporal_window)
        self.direction_history: deque = deque(maxlen=temporal_window)
        self.entropy_history: deque = deque(maxlen=temporal_window)
        self.velocity_history: deque = deque(maxlen=temporal_window)
        
        self.frame_count = 0
        self.prev_features = None
        
        logger.info(f"RealVideoFeatureExtractor initialized")
        logger.info(f"  Frame size: {frame_size}")
        logger.info(f"  Sample interval: every {sample_interval} frames")
        logger.info(f"  Feature dimension: {len(BEHAVIORAL_FEATURES)}")
    
    def reset(self):
        """Reset for new video."""
        self.flow_extractor.reset()
        self.motion_history.clear()
        self.direction_history.clear()
        self.entropy_history.clear()
        self.velocity_history.clear()
        self.frame_count = 0
        self.prev_features = None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for feature extraction."""
        # Resize
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        return frame
    
    def compute_scene_entropy(self, frame: np.ndarray) -> float:
        """Compute scene entropy as measure of visual complexity."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize to 0-1 range (max entropy for 256 bins is 8)
        return entropy / 8.0
    
    def compute_direction_histogram(self, angle: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        """Compute weighted direction histogram."""
        # 8 direction bins
        n_bins = 8
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        
        # Weight by magnitude
        hist = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (angle >= bin_edges[i]) & (angle < bin_edges[i + 1])
            hist[i] = np.sum(magnitude[mask])
        
        # Normalize
        total = hist.sum()
        if total > 0:
            hist /= total
        
        return hist
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 24 behavioral features from a single frame.
        
        Returns:
            np.ndarray of shape (24,) matching BEHAVIORAL_FEATURES order
        """
        self.frame_count += 1
        
        # Preprocess
        frame = self.preprocess_frame(frame)
        
        # Compute optical flow
        magnitude, angle = self.flow_extractor.compute(frame)
        
        # Compute scene entropy
        scene_entropy = self.compute_scene_entropy(frame)
        
        # === MOTION ENERGY FEATURES ===
        motion_energy = float(np.mean(magnitude))
        motion_energy_std = float(np.std(magnitude))
        peak_motion = float(np.max(magnitude))
        
        # Update motion history
        self.motion_history.append(motion_energy)
        
        # === OPTICAL FLOW FEATURES ===
        flow_magnitude_mean = motion_energy  # Same as motion_energy
        flow_magnitude_std = motion_energy_std
        flow_magnitude_max = peak_motion
        flow_variance = float(np.var(magnitude))
        
        # === DIRECTIONAL FEATURES ===
        dir_hist = self.compute_direction_histogram(angle, magnitude)
        self.direction_history.append(dir_hist)
        
        # Direction consistency: how concentrated is movement direction
        direction_consistency = float(np.max(dir_hist) - np.mean(dir_hist))
        
        # Direction entropy: diversity of movement directions
        dir_hist_nonzero = dir_hist[dir_hist > 0]
        if len(dir_hist_nonzero) > 0:
            direction_entropy = float(-np.sum(dir_hist_nonzero * np.log2(dir_hist_nonzero + 1e-10)))
        else:
            direction_entropy = 0.0
        
        # Dominant direction (normalized to 0-1)
        dominant_direction = float(np.argmax(dir_hist) / 8.0)
        
        # Direction change rate
        if len(self.direction_history) >= 2:
            prev_dominant = np.argmax(self.direction_history[-2])
            curr_dominant = np.argmax(dir_hist)
            direction_change_rate = float(abs(curr_dominant - prev_dominant) / 4.0)
        else:
            direction_change_rate = 0.0
        
        # === SCENE ANALYSIS FEATURES ===
        self.entropy_history.append(scene_entropy)
        
        # Scene entropy change
        if len(self.entropy_history) >= 2:
            scene_entropy_change = float(abs(scene_entropy - self.entropy_history[-2]))
        else:
            scene_entropy_change = 0.0
        
        # Spatial coherence: how uniform is motion across the frame
        # Higher coherence = more uniform motion
        if motion_energy > 0.1:
            spatial_coherence = float(1.0 - (motion_energy_std / (motion_energy + 1e-6)))
        else:
            spatial_coherence = 1.0
        spatial_coherence = max(0.0, min(1.0, spatial_coherence))
        
        # === VELOCITY FEATURES ===
        velocity_mean = motion_energy
        velocity_std = motion_energy_std
        velocity_variance = flow_variance
        
        self.velocity_history.append(velocity_mean)
        
        # Acceleration (change in velocity)
        if len(self.velocity_history) >= 2:
            acceleration_mean = float(abs(velocity_mean - self.velocity_history[-2]))
        else:
            acceleration_mean = 0.0
        
        # === ACTIVITY PATTERN FEATURES ===
        # Count pixels below idle threshold
        total_pixels = magnitude.size
        idle_pixels = np.sum(magnitude < self.idle_threshold)
        active_pixels = np.sum(magnitude > self.motion_threshold)
        
        idle_ratio = float(idle_pixels / total_pixels)
        active_ratio = float(active_pixels / total_pixels)
        
        # Activity transitions: changes between idle and active states
        if len(self.motion_history) >= 2:
            prev_active = self.motion_history[-2] > self.motion_threshold
            curr_active = motion_energy > self.motion_threshold
            activity_transitions = float(1.0 if prev_active != curr_active else 0.0)
        else:
            activity_transitions = 0.0
        
        # === TEMPORAL FEATURES ===
        # Temporal gradient: rate of change
        if len(self.motion_history) >= 3:
            recent = list(self.motion_history)[-3:]
            temporal_gradient = float(np.polyfit(range(len(recent)), recent, 1)[0])
        else:
            temporal_gradient = 0.0
        
        # Temporal stability: consistency of motion over time
        if len(self.motion_history) >= 5:
            recent = list(self.motion_history)[-5:]
            temporal_stability = float(1.0 - min(1.0, np.std(recent) / (np.mean(recent) + 1e-6)))
        else:
            temporal_stability = 1.0
        temporal_stability = max(0.0, min(1.0, temporal_stability))
        
        # Movement complexity: based on direction entropy and flow variance
        movement_complexity = float((direction_entropy / 3.0 + flow_variance / 10.0) / 2.0)
        movement_complexity = max(0.0, min(1.0, movement_complexity))
        
        # === ASSEMBLE FEATURE VECTOR ===
        # Order MUST match BEHAVIORAL_FEATURES
        features = np.array([
            motion_energy,
            motion_energy_std,
            peak_motion,
            flow_magnitude_mean,
            flow_magnitude_std,
            flow_magnitude_max,
            flow_variance,
            direction_consistency,
            direction_entropy,
            dominant_direction,
            direction_change_rate,
            scene_entropy,
            scene_entropy_change,
            spatial_coherence,
            velocity_mean,
            velocity_std,
            velocity_variance,
            acceleration_mean,
            idle_ratio,
            active_ratio,
            activity_transitions,
            temporal_gradient,
            temporal_stability,
            movement_complexity,
        ], dtype=np.float32)
        
        self.prev_features = features
        
        return features
    
    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ) -> Tuple[np.ndarray, VideoMetadata]:
        """
        Extract features from a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            start_frame: Starting frame index
            
        Returns:
            Tuple of (features array, video metadata)
        """
        self.reset()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = VideoMetadata(
            path=video_path,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=total_frames / fps if fps > 0 else 0,
            source_type="surveillance"
        )
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"  Total frames: {total_frames}, FPS: {fps:.1f}")
        logger.info(f"  Sampling every {self.sample_interval} frames")
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        features_list = []
        frame_idx = start_frame
        sampled_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample at interval
            if frame_idx % self.sample_interval == 0:
                features = self.extract_features(frame)
                features_list.append(features)
                sampled_count += 1
                
                if max_frames and sampled_count >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        features_array = np.array(features_list)
        logger.info(f"  Extracted {len(features_array)} feature vectors")
        
        return features_array, metadata
    
    def extract_from_frames(
        self,
        frame_generator: Generator[np.ndarray, None, None],
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract features from a frame generator.
        
        Useful for UCSD dataset which provides individual frames.
        """
        self.reset()
        
        features_list = []
        for i, frame in enumerate(frame_generator):
            if max_frames and i >= max_frames:
                break
            
            if i % self.sample_interval == 0:
                features = self.extract_features(frame)
                features_list.append(features)
        
        return np.array(features_list)


class UCSDDatasetLoader:
    """
    Loader for UCSD Anomaly Detection Dataset.
    
    PUBLIC DATASET USAGE:
    "Public surveillance dataset used as proxy for border CCTV footage."
    
    LABELS ARE INTENTIONALLY IGNORED.
    We train only on NORMAL behavior - the system learns what is normal
    and flags deviations, not specific anomaly types.
    
    Dataset structure:
    - UCSDped1/Train/Train001/*.tif (normal sequences)
    - UCSDped1/Test/Test001/*.tif (contains anomalies - labels ignored)
    """
    
    def __init__(self, dataset_path: str, subset: str = "ped1"):
        """
        Initialize UCSD dataset loader.
        
        Args:
            dataset_path: Path to UCSD dataset root (e.g., "./data/UCSD")
            subset: "ped1" or "ped2"
        """
        self.dataset_path = Path(dataset_path)
        self.subset = subset.lower()
        
        # Determine paths
        if self.subset == "ped1":
            self.train_path = self.dataset_path / "UCSDped1" / "Train"
            self.test_path = self.dataset_path / "UCSDped1" / "Test"
        elif self.subset == "ped2":
            self.train_path = self.dataset_path / "UCSDped2" / "Train"
            self.test_path = self.dataset_path / "UCSDped2" / "Test"
        else:
            raise ValueError(f"Unknown subset: {subset}. Use 'ped1' or 'ped2'")
        
        logger.info(f"UCSDDatasetLoader initialized")
        logger.info(f"  Subset: {subset}")
        logger.info(f"  Train path: {self.train_path}")
        logger.info(f"  NOTE: Labels intentionally ignored - unsupervised learning only")
    
    def _load_sequence(self, sequence_path: Path) -> Generator[np.ndarray, None, None]:
        """Load frames from a sequence directory."""
        # Find all .tif files
        frame_files = sorted(glob.glob(str(sequence_path / "*.tif")))
        
        if not frame_files:
            # Try .png
            frame_files = sorted(glob.glob(str(sequence_path / "*.png")))
        
        for frame_path in frame_files:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                yield frame
    
    def get_train_sequences(self) -> List[Path]:
        """Get list of training sequence directories."""
        if not self.train_path.exists():
            logger.warning(f"Train path does not exist: {self.train_path}")
            return []
        
        sequences = sorted([d for d in self.train_path.iterdir() if d.is_dir()])
        return sequences
    
    def get_test_sequences(self) -> List[Path]:
        """Get list of test sequence directories."""
        if not self.test_path.exists():
            logger.warning(f"Test path does not exist: {self.test_path}")
            return []
        
        sequences = sorted([d for d in self.test_path.iterdir() if d.is_dir()])
        return sequences
    
    def load_train_frames(
        self,
        max_sequences: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Load all training frames as a generator.
        
        Training data contains ONLY normal behavior.
        """
        sequences = self.get_train_sequences()
        
        if max_sequences:
            sequences = sequences[:max_sequences]
        
        logger.info(f"Loading {len(sequences)} training sequences (NORMAL behavior only)")
        
        for seq_path in sequences:
            for frame in self._load_sequence(seq_path):
                yield frame
    
    def load_test_frames(
        self,
        max_sequences: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Load test frames as a generator.
        
        NOTE: Test data contains anomalies, but we IGNORE labels.
        The system should detect drift without supervision.
        """
        sequences = self.get_test_sequences()
        
        if max_sequences:
            sequences = sequences[:max_sequences]
        
        logger.info(f"Loading {len(sequences)} test sequences")
        logger.info("  NOTE: Anomaly labels intentionally ignored")
        
        for seq_path in sequences:
            for frame in self._load_sequence(seq_path):
                yield frame
    
    def is_available(self) -> bool:
        """Check if dataset is available."""
        return self.train_path.exists()


def extract_features_from_ucsd(
    dataset_path: str,
    subset: str = "ped1",
    feature_extractor: Optional[RealVideoFeatureExtractor] = None,
    max_train_sequences: int = 10,
    max_test_sequences: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from UCSD dataset.
    
    Returns:
        Tuple of (train_features, test_features)
    """
    loader = UCSDDatasetLoader(dataset_path, subset)
    
    if not loader.is_available():
        raise FileNotFoundError(
            f"UCSD dataset not found at {dataset_path}. "
            f"Download from: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm"
        )
    
    if feature_extractor is None:
        feature_extractor = RealVideoFeatureExtractor(sample_interval=2)
    
    # Extract training features (NORMAL only)
    logger.info("Extracting training features (NORMAL behavior)...")
    train_features = feature_extractor.extract_from_frames(
        loader.load_train_frames(max_train_sequences),
        max_frames=2000
    )
    
    # Extract test features (contains anomalies - labels ignored)
    logger.info("Extracting test features (labels ignored)...")
    feature_extractor.reset()
    test_features = feature_extractor.extract_from_frames(
        loader.load_test_frames(max_test_sequences),
        max_frames=1000
    )
    
    return train_features, test_features


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_video_extraction(video_path: str):
    """Demonstrate feature extraction from a video file."""
    extractor = RealVideoFeatureExtractor()
    
    try:
        features, metadata = extractor.extract_from_video(video_path, max_frames=100)
        
        print(f"\n{'='*60}")
        print("REAL VIDEO FEATURE EXTRACTION")
        print(f"{'='*60}")
        print(f"Video: {metadata.path}")
        print(f"Resolution: {metadata.width}x{metadata.height}")
        print(f"Duration: {metadata.duration_seconds:.1f}s")
        print(f"Features extracted: {len(features)}")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"\nFeature statistics:")
        print(f"  Mean: {np.mean(features, axis=0)[:5]}...")
        print(f"  Std:  {np.std(features, axis=0)[:5]}...")
        
        return features, metadata
        
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        return None, None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        demo_video_extraction(video_path)
    else:
        print("Usage: python video_features.py <video_path>")
        print("\nThis module extracts 24 behavioral features from surveillance video")
        print("using optical flow analysis. Output format matches synthetic data exactly.")
