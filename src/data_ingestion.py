"""
NOISE FLOOR - Data Ingestion Module
=====================================
Border & Perimeter Surveillance Data Ingestion Pipeline

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

Supports two data ingestion modes:
- Mode A: Surveillance Video (PRIMARY) - UCSD, Avenue, UMN datasets
- Mode B: Sensor Time-Series (SECONDARY) - CSV-based radar/vibration data

GRAY-BOX DESIGN:
----------------
All data sources are processed into a unified temporal feature format
suitable for the LSTM-VAE normality learning model.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Generator, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data source types for border surveillance."""
    VIDEO_UCSD = "ucsd"              # UCSD Anomaly Detection Dataset
    VIDEO_AVENUE = "avenue"          # Avenue Dataset
    VIDEO_UMN = "umn"                # UMN Crowd Activity Dataset
    VIDEO_CUSTOM = "custom_video"    # Custom surveillance video
    SENSOR_CSV = "sensor_csv"        # CSV-based sensor data
    SYNTHETIC = "synthetic"          # Synthetic data for testing


@dataclass
class SurveillanceMetadata:
    """
    Metadata for surveillance data stream.
    Provides context for the normality model.
    """
    source_type: DataSource
    source_path: str
    camera_zone: str = "perimeter"           # fence, road, open_area, checkpoint
    patrol_context: str = "static"           # static, mobile, mixed
    expected_fps: float = 30.0
    resolution: Tuple[int, int] = (224, 224)
    start_timestamp: float = 0.0
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'source_type': self.source_type.value,
            'source_path': self.source_path,
            'camera_zone': self.camera_zone,
            'patrol_context': self.patrol_context,
            'expected_fps': self.expected_fps,
            'resolution': self.resolution,
            'start_timestamp': self.start_timestamp,
            'description': self.description
        }


@dataclass
class TemporalWindow:
    """
    A single temporal window of data with context.
    This is the basic unit passed to the LSTM-VAE model.
    """
    # Core temporal data
    frames: np.ndarray                       # Shape: (T, H, W) or (T, H, W, C)
    features: Optional[np.ndarray] = None    # Shape: (T, F) extracted features
    
    # Temporal metadata
    window_index: int = 0
    start_frame: int = 0
    end_frame: int = 0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    
    # Context augmentation (computed in augmentation step)
    context: Dict = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return self.end_timestamp - self.start_timestamp
    
    @property
    def num_frames(self) -> int:
        return self.frames.shape[0] if self.frames is not None else 0


class DataIngestionBase(ABC):
    """
    Abstract base class for data ingestion.
    All data sources implement this interface.
    """
    
    @abstractmethod
    def stream_windows(
        self, 
        window_size: int = 30,      # ~2-5 seconds at 10-15 fps
        stride: int = 10
    ) -> Generator[TemporalWindow, None, None]:
        """Yield temporal windows of data."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> SurveillanceMetadata:
        """Return metadata about the data source."""
        pass
    
    @abstractmethod
    def get_total_frames(self) -> int:
        """Return total number of frames available."""
        pass


class VideoIngestion(DataIngestionBase):
    """
    Video data ingestion for surveillance footage.
    
    Supports:
    - UCSD Anomaly Detection Dataset
    - Avenue Dataset
    - UMN Crowd Activity Dataset
    - Custom surveillance video files
    
    DESIGN NOTES:
    -------------
    - Processes video at reduced resolution for efficiency
    - Extracts grayscale for motion analysis (color not needed)
    - Yields overlapping temporal windows for continuous analysis
    """
    
    def __init__(
        self,
        video_path: str,
        source_type: DataSource = DataSource.VIDEO_CUSTOM,
        target_fps: float = 10.0,           # Downsample for efficiency
        frame_size: Tuple[int, int] = (224, 224),
        camera_zone: str = "perimeter",
        patrol_context: str = "static",
        grayscale: bool = True
    ):
        self.video_path = Path(video_path)
        self.source_type = source_type
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.camera_zone = camera_zone
        self.patrol_context = patrol_context
        self.grayscale = grayscale
        
        # Video capture will be initialized on first access
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[SurveillanceMetadata] = None
        self._total_frames: Optional[int] = None
        self._native_fps: Optional[float] = None
        
        logger.info(f"VideoIngestion initialized: {video_path}")
        logger.info(f"  Camera zone: {camera_zone}, Context: {patrol_context}")
    
    def _ensure_capture(self):
        """Lazily initialize video capture."""
        if self._cap is None:
            if not self.video_path.exists():
                raise FileNotFoundError(f"Video not found: {self.video_path}")
            
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            
            self._native_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"  Native FPS: {self._native_fps}, Frames: {self._total_frames}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame."""
        # Resize
        resized = cv2.resize(frame, self.frame_size)
        
        # Convert to grayscale if needed
        if self.grayscale and len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        return resized
    
    def stream_windows(
        self, 
        window_size: int = 30,
        stride: int = 10
    ) -> Generator[TemporalWindow, None, None]:
        """
        Stream temporal windows from video.
        
        Args:
            window_size: Number of frames per window (~2-5 seconds)
            stride: Frames to advance between windows
            
        Yields:
            TemporalWindow objects containing frame sequences
        """
        self._ensure_capture()
        
        # Calculate frame skip for target FPS
        frame_skip = max(1, int(self._native_fps / self.target_fps))
        
        # Buffer for current window
        frame_buffer = []
        frame_indices = []
        
        frame_idx = 0
        window_idx = 0
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            
            # Skip frames for target FPS
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Process and add to buffer
            processed = self._process_frame(frame)
            frame_buffer.append(processed)
            frame_indices.append(frame_idx)
            
            # Yield window when buffer is full
            if len(frame_buffer) >= window_size:
                frames_array = np.stack(frame_buffer[:window_size])
                
                yield TemporalWindow(
                    frames=frames_array,
                    window_index=window_idx,
                    start_frame=frame_indices[0],
                    end_frame=frame_indices[window_size - 1],
                    start_timestamp=frame_indices[0] / self._native_fps,
                    end_timestamp=frame_indices[window_size - 1] / self._native_fps,
                )
                
                # Slide window
                frame_buffer = frame_buffer[stride:]
                frame_indices = frame_indices[stride:]
                window_idx += 1
            
            frame_idx += 1
        
        # Yield remaining frames if any
        if len(frame_buffer) >= window_size // 2:
            frames_array = np.stack(frame_buffer)
            # Pad if needed
            if len(frames_array) < window_size:
                pad_size = window_size - len(frames_array)
                padding = np.repeat(frames_array[-1:], pad_size, axis=0)
                frames_array = np.concatenate([frames_array, padding])
            
            yield TemporalWindow(
                frames=frames_array[:window_size],
                window_index=window_idx,
                start_frame=frame_indices[0],
                end_frame=frame_indices[-1],
                start_timestamp=frame_indices[0] / self._native_fps,
                end_timestamp=frame_indices[-1] / self._native_fps,
            )
    
    def get_metadata(self) -> SurveillanceMetadata:
        """Return video metadata."""
        if self._metadata is None:
            self._ensure_capture()
            self._metadata = SurveillanceMetadata(
                source_type=self.source_type,
                source_path=str(self.video_path),
                camera_zone=self.camera_zone,
                patrol_context=self.patrol_context,
                expected_fps=self.target_fps,
                resolution=self.frame_size,
                description=f"Surveillance video from {self.video_path.name}"
            )
        return self._metadata
    
    def get_total_frames(self) -> int:
        """Return total frame count."""
        self._ensure_capture()
        return self._total_frames
    
    def close(self):
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SensorCSVIngestion(DataIngestionBase):
    """
    Sensor time-series data ingestion from CSV files.
    
    Supports radar, vibration, acoustic, and other sensor modalities
    commonly used in perimeter security systems.
    
    Expected CSV format:
    - timestamp column (or index)
    - Multiple sensor reading columns
    - Optional metadata columns
    """
    
    def __init__(
        self,
        csv_path: str,
        timestamp_col: str = "timestamp",
        feature_cols: Optional[List[str]] = None,
        sample_rate: float = 10.0,          # Hz
        camera_zone: str = "perimeter",
        patrol_context: str = "static"
    ):
        self.csv_path = Path(csv_path)
        self.timestamp_col = timestamp_col
        self.feature_cols = feature_cols
        self.sample_rate = sample_rate
        self.camera_zone = camera_zone
        self.patrol_context = patrol_context
        
        self._data: Optional[pd.DataFrame] = None
        self._metadata: Optional[SurveillanceMetadata] = None
        
        logger.info(f"SensorCSVIngestion initialized: {csv_path}")
    
    def _load_data(self):
        """Load CSV data."""
        if self._data is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {self.csv_path}")
            
            self._data = pd.read_csv(self.csv_path)
            
            # Determine feature columns
            if self.feature_cols is None:
                self.feature_cols = [
                    col for col in self._data.columns 
                    if col != self.timestamp_col
                ]
            
            logger.info(f"  Loaded {len(self._data)} samples, {len(self.feature_cols)} features")
    
    def stream_windows(
        self, 
        window_size: int = 30,
        stride: int = 10
    ) -> Generator[TemporalWindow, None, None]:
        """
        Stream temporal windows from sensor data.
        """
        self._load_data()
        
        num_samples = len(self._data)
        window_idx = 0
        
        for start_idx in range(0, num_samples - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            # Extract feature window
            window_data = self._data.iloc[start_idx:end_idx][self.feature_cols].values
            
            # Get timestamps
            if self.timestamp_col in self._data.columns:
                start_ts = self._data.iloc[start_idx][self.timestamp_col]
                end_ts = self._data.iloc[end_idx - 1][self.timestamp_col]
            else:
                start_ts = start_idx / self.sample_rate
                end_ts = (end_idx - 1) / self.sample_rate
            
            yield TemporalWindow(
                frames=window_data,  # For sensors, frames = feature sequences
                features=window_data,
                window_index=window_idx,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                start_timestamp=float(start_ts),
                end_timestamp=float(end_ts),
            )
            
            window_idx += 1
    
    def get_metadata(self) -> SurveillanceMetadata:
        """Return sensor metadata."""
        if self._metadata is None:
            self._load_data()
            self._metadata = SurveillanceMetadata(
                source_type=DataSource.SENSOR_CSV,
                source_path=str(self.csv_path),
                camera_zone=self.camera_zone,
                patrol_context=self.patrol_context,
                expected_fps=self.sample_rate,
                resolution=(1, len(self.feature_cols)),
                description=f"Sensor data from {self.csv_path.name}"
            )
        return self._metadata
    
    def get_total_frames(self) -> int:
        """Return total sample count."""
        self._load_data()
        return len(self._data)


class SyntheticDataGenerator(DataIngestionBase):
    """
    Synthetic data generator for testing and demonstration.
    
    Generates realistic behavioral patterns that simulate:
    - Normal perimeter patrol patterns
    - Gradual behavioral drift (potential threats)
    - Various environmental conditions
    
    This is designed for border surveillance and high-security perimeters
    where threats emerge gradually.
    """
    
    def __init__(
        self,
        num_frames: int = 1000,
        feature_dim: int = 15,
        drift_start: Optional[int] = None,   # Frame where drift begins
        drift_rate: float = 0.02,            # Rate of behavioral change
        noise_level: float = 0.1,
        camera_zone: str = "perimeter",
        patrol_context: str = "static",
        seed: int = 42
    ):
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.drift_start = drift_start or int(num_frames * 0.4)
        self.drift_rate = drift_rate
        self.noise_level = noise_level
        self.camera_zone = camera_zone
        self.patrol_context = patrol_context
        self.seed = seed
        
        np.random.seed(seed)
        
        # Generate all data upfront
        self._data = self._generate_surveillance_data()
        
        logger.info(f"SyntheticDataGenerator: {num_frames} frames, drift at {self.drift_start}")
    
    def _generate_surveillance_data(self) -> np.ndarray:
        """
        Generate synthetic behavioral feature time series.
        
        Features simulate border surveillance metrics:
        - Motion energy (overall activity)
        - Optical flow variance (movement consistency)
        - Directional consistency (coordinated vs random movement)
        - Scene entropy (visual complexity)
        - Velocity variance (speed consistency)
        - Idle/active ratio (movement patterns)
        """
        t = np.arange(self.num_frames)
        features = []
        
        # Feature names for reference (matches extraction module)
        feature_names = [
            'motion_energy',
            'flow_magnitude_mean',
            'flow_magnitude_std',
            'flow_variance',
            'direction_consistency',
            'direction_entropy',
            'scene_entropy',
            'velocity_mean',
            'velocity_std',
            'velocity_variance',
            'idle_ratio',
            'active_ratio',
            'temporal_gradient',
            'spatial_coherence',
            'movement_complexity'
        ]
        
        for i, name in enumerate(feature_names[:self.feature_dim]):
            # Base signal with slight variation
            base_freq = 0.01 + 0.005 * np.sin(2 * np.pi * i / self.feature_dim)
            signal = np.sin(2 * np.pi * base_freq * t + i * 0.5)
            
            # Add temporal autocorrelation (realistic sensor behavior)
            signal = np.convolve(signal, np.ones(5)/5, mode='same')
            
            # Add gradual drift after drift_start
            drift_mask = t >= self.drift_start
            drift_amount = np.zeros_like(t, dtype=float)
            drift_indices = t[drift_mask] - self.drift_start
            
            # Different features drift at different rates
            feature_drift_rate = self.drift_rate * (1 + 0.5 * np.sin(i))
            drift_amount[drift_mask] = drift_indices * feature_drift_rate
            
            # Some features drift up, some down
            drift_direction = 1 if i % 2 == 0 else -1
            signal += drift_direction * drift_amount
            
            # Add noise
            noise = np.random.normal(0, self.noise_level, self.num_frames)
            signal += noise
            
            # Normalize to reasonable range
            signal = (signal - signal.mean()) / (signal.std() + 1e-6)
            
            features.append(signal)
        
        return np.column_stack(features)
    
    def stream_windows(
        self, 
        window_size: int = 30,
        stride: int = 10
    ) -> Generator[TemporalWindow, None, None]:
        """Stream temporal windows of synthetic data."""
        num_frames = len(self._data)
        window_idx = 0
        
        for start_idx in range(0, num_frames - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            window_data = self._data[start_idx:end_idx]
            
            yield TemporalWindow(
                frames=window_data,
                features=window_data,
                window_index=window_idx,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                start_timestamp=start_idx / 10.0,
                end_timestamp=(end_idx - 1) / 10.0,
            )
            
            window_idx += 1
    
    def get_metadata(self) -> SurveillanceMetadata:
        """Return synthetic data metadata."""
        return SurveillanceMetadata(
            source_type=DataSource.SYNTHETIC,
            source_path="synthetic",
            camera_zone=self.camera_zone,
            patrol_context=self.patrol_context,
            expected_fps=10.0,
            resolution=(1, self.feature_dim),
            description="Synthetic surveillance behavioral data"
        )
    
    def get_total_frames(self) -> int:
        """Return total frame count."""
        return self.num_frames
    
    def get_raw_data(self) -> np.ndarray:
        """Return raw feature data for direct access."""
        return self._data.copy()
    
    def get_drift_start(self) -> int:
        """Return the frame where drift begins."""
        return self.drift_start


class DatasetLoader:
    """
    Unified dataset loader for common surveillance datasets.
    
    Supports automatic detection and loading of:
    - UCSD Anomaly Detection Dataset
    - Avenue Dataset  
    - UMN Crowd Activity Dataset
    """
    
    DATASET_CONFIGS = {
        'ucsd': {
            'source_type': DataSource.VIDEO_UCSD,
            'description': 'UCSD Anomaly Detection Dataset',
            'train_pattern': 'Train*/Train*.avi',
            'test_pattern': 'Test*/Test*.avi',
        },
        'avenue': {
            'source_type': DataSource.VIDEO_AVENUE,
            'description': 'CUHK Avenue Dataset',
            'train_pattern': 'training_videos/*.avi',
            'test_pattern': 'testing_videos/*.avi',
        },
        'umn': {
            'source_type': DataSource.VIDEO_UMN,
            'description': 'UMN Crowd Activity Dataset',
            'train_pattern': 'normal/*.avi',
            'test_pattern': 'abnormal/*.avi',
        }
    }
    
    @classmethod
    def load(
        cls,
        dataset_path: str,
        dataset_type: str = 'auto',
        split: str = 'train',
        **kwargs
    ) -> List[VideoIngestion]:
        """
        Load a surveillance dataset.
        
        Args:
            dataset_path: Path to dataset root
            dataset_type: 'ucsd', 'avenue', 'umn', or 'auto'
            split: 'train' or 'test'
            **kwargs: Additional arguments for VideoIngestion
            
        Returns:
            List of VideoIngestion objects
        """
        dataset_path = Path(dataset_path)
        
        if dataset_type == 'auto':
            dataset_type = cls._detect_dataset_type(dataset_path)
        
        if dataset_type not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        config = cls.DATASET_CONFIGS[dataset_type]
        pattern = config['train_pattern'] if split == 'train' else config['test_pattern']
        
        video_files = list(dataset_path.glob(pattern))
        
        if not video_files:
            raise FileNotFoundError(
                f"No videos found in {dataset_path} matching {pattern}"
            )
        
        logger.info(f"Loading {len(video_files)} videos from {dataset_type} {split} set")
        
        ingestion_list = []
        for video_file in sorted(video_files):
            ingestion_list.append(
                VideoIngestion(
                    video_path=str(video_file),
                    source_type=config['source_type'],
                    **kwargs
                )
            )
        
        return ingestion_list
    
    @classmethod
    def _detect_dataset_type(cls, path: Path) -> str:
        """Auto-detect dataset type from directory structure."""
        path_str = str(path).lower()
        
        if 'ucsd' in path_str:
            return 'ucsd'
        elif 'avenue' in path_str:
            return 'avenue'
        elif 'umn' in path_str:
            return 'umn'
        else:
            # Check directory contents
            if (path / 'Train001').exists() or (path / 'Train01').exists():
                return 'ucsd'
            elif (path / 'training_videos').exists():
                return 'avenue'
            else:
                raise ValueError(
                    "Cannot auto-detect dataset type. "
                    "Specify dataset_type='ucsd', 'avenue', or 'umn'"
                )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ingestion(
    source: str,
    source_type: str = 'auto',
    **kwargs
) -> DataIngestionBase:
    """
    Create appropriate data ingestion object.
    
    Args:
        source: Path to video, CSV, or 'synthetic'
        source_type: 'video', 'csv', 'synthetic', or 'auto'
        **kwargs: Additional arguments
        
    Returns:
        DataIngestionBase instance
    """
    if source == 'synthetic':
        return SyntheticDataGenerator(**kwargs)
    
    source_path = Path(source)
    
    if source_type == 'auto':
        if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            source_type = 'video'
        elif source_path.suffix.lower() in ['.csv', '.tsv']:
            source_type = 'csv'
        else:
            raise ValueError(f"Cannot determine source type for: {source}")
    
    if source_type == 'video':
        return VideoIngestion(source, **kwargs)
    elif source_type == 'csv':
        return SensorCSVIngestion(source, **kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")
