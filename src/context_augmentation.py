"""
NOISE FLOOR - Context Augmentation Module
==========================================
Scene & Context Awareness for Border Surveillance

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

Context augmentation adds situational awareness to behavioral features:
- Time of day (day/night) affects normal behavior patterns
- Visibility conditions impact detection reliability
- Camera zone defines expected activity levels
- Patrol context determines baseline expectations

GRAY-BOX DESIGN:
----------------
Normal behavior is CONTEXT-AWARE, not static. A gathering at noon
differs from one at midnight. This module provides the contextual
signals that allow the LSTM-VAE to learn context-specific normality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeOfDay(Enum):
    """Time of day classification for surveillance context."""
    DAWN = "dawn"           # 05:00 - 07:00
    DAY = "day"             # 07:00 - 18:00
    DUSK = "dusk"           # 18:00 - 20:00
    NIGHT = "night"         # 20:00 - 05:00
    
    @classmethod
    def from_hour(cls, hour: int) -> 'TimeOfDay':
        """Determine time of day from hour (0-23)."""
        if 5 <= hour < 7:
            return cls.DAWN
        elif 7 <= hour < 18:
            return cls.DAY
        elif 18 <= hour < 20:
            return cls.DUSK
        else:
            return cls.NIGHT
    
    @property
    def activity_modifier(self) -> float:
        """Expected activity level modifier."""
        modifiers = {
            TimeOfDay.DAWN: 0.6,    # Low activity
            TimeOfDay.DAY: 1.0,     # Normal activity
            TimeOfDay.DUSK: 0.7,    # Decreasing activity
            TimeOfDay.NIGHT: 0.3,   # Minimal activity
        }
        return modifiers[self]
    
    @property
    def alert_sensitivity(self) -> float:
        """Sensitivity multiplier for alerts."""
        sensitivities = {
            TimeOfDay.DAWN: 1.2,    # Higher sensitivity
            TimeOfDay.DAY: 1.0,     # Normal sensitivity
            TimeOfDay.DUSK: 1.3,    # Higher sensitivity
            TimeOfDay.NIGHT: 1.5,   # Highest sensitivity
        }
        return sensitivities[self]


class CameraZone(Enum):
    """Camera zone classification for border surveillance."""
    FENCE = "fence"                 # Fence line monitoring
    ROAD = "road"                   # Access road surveillance
    OPEN_AREA = "open_area"         # Open terrain monitoring
    CHECKPOINT = "checkpoint"       # Entry/exit points
    WATER_CROSSING = "water"        # River/water boundaries
    VEGETATION = "vegetation"       # Forested/vegetated areas
    
    @property
    def expected_activity(self) -> float:
        """Expected baseline activity level (0-1)."""
        levels = {
            CameraZone.FENCE: 0.2,          # Minimal activity expected
            CameraZone.ROAD: 0.7,           # Regular traffic
            CameraZone.OPEN_AREA: 0.3,      # Occasional activity
            CameraZone.CHECKPOINT: 0.9,     # High legitimate activity
            CameraZone.WATER_CROSSING: 0.1, # Very low activity
            CameraZone.VEGETATION: 0.2,     # Low activity
        }
        return levels[self]
    
    @property
    def alert_threshold_modifier(self) -> float:
        """Modifier for alert thresholds (lower = more sensitive)."""
        modifiers = {
            CameraZone.FENCE: 0.7,          # More sensitive
            CameraZone.ROAD: 1.2,           # Less sensitive (more normal activity)
            CameraZone.OPEN_AREA: 0.8,      # Moderate sensitivity
            CameraZone.CHECKPOINT: 1.5,     # Much less sensitive (high activity zone)
            CameraZone.WATER_CROSSING: 0.5, # Very sensitive
            CameraZone.VEGETATION: 0.6,     # High sensitivity
        }
        return modifiers[self]


class PatrolContext(Enum):
    """Patrol status context."""
    STATIC = "static"               # Fixed camera, no patrols
    PATROL_NEARBY = "patrol_nearby" # Active patrol in zone
    PATROL_AWAY = "patrol_away"     # Patrol elsewhere
    ALERT_MODE = "alert_mode"       # Heightened alert status
    
    @property
    def baseline_modifier(self) -> float:
        """Modifier for baseline expectations."""
        modifiers = {
            PatrolContext.STATIC: 1.0,
            PatrolContext.PATROL_NEARBY: 1.3,  # More activity expected
            PatrolContext.PATROL_AWAY: 0.9,    # Slightly less activity
            PatrolContext.ALERT_MODE: 0.5,     # Much stricter baseline
        }
        return modifiers[self]


class VisibilityLevel(Enum):
    """Visibility conditions classification."""
    EXCELLENT = "excellent"     # Clear conditions
    GOOD = "good"               # Minor haze/clouds
    MODERATE = "moderate"       # Fog/dust/rain
    POOR = "poor"               # Heavy fog/storm
    MINIMAL = "minimal"         # Near-zero visibility
    
    @property
    def confidence_modifier(self) -> float:
        """Detection confidence modifier."""
        modifiers = {
            VisibilityLevel.EXCELLENT: 1.0,
            VisibilityLevel.GOOD: 0.95,
            VisibilityLevel.MODERATE: 0.8,
            VisibilityLevel.POOR: 0.6,
            VisibilityLevel.MINIMAL: 0.3,
        }
        return modifiers[self]


@dataclass
class SceneContext:
    """
    Complete scene context for a temporal window.
    Provides all contextual signals for context-aware normality.
    """
    # Temporal context
    time_of_day: TimeOfDay = TimeOfDay.DAY
    hour: int = 12
    minute: int = 0
    
    # Location context
    camera_zone: CameraZone = CameraZone.OPEN_AREA
    camera_id: str = "CAM_001"
    
    # Environmental context
    visibility: VisibilityLevel = VisibilityLevel.GOOD
    brightness: float = 0.5             # 0-1 normalized
    contrast: float = 0.5               # 0-1 normalized
    
    # Operational context
    patrol_context: PatrolContext = PatrolContext.STATIC
    alert_level: int = 0                # 0 = normal, 1-5 = elevated
    
    # Computed modifiers
    activity_modifier: float = 1.0
    sensitivity_modifier: float = 1.0
    confidence_modifier: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """
        Convert context to feature vector for model input.
        
        Returns:
            Context feature vector (12 dimensions)
        """
        # One-hot encode time of day
        tod_vec = np.zeros(4)
        tod_vec[list(TimeOfDay).index(self.time_of_day)] = 1.0
        
        # One-hot encode camera zone
        zone_vec = np.zeros(6)
        zone_vec[list(CameraZone).index(self.camera_zone)] = 1.0
        
        # Continuous features
        continuous = np.array([
            self.brightness,
            self.contrast,
            self.visibility.confidence_modifier,
            self.activity_modifier,
            self.sensitivity_modifier,
            (self.hour * 60 + self.minute) / 1440.0,  # Normalized time
        ])
        
        return np.concatenate([tod_vec, zone_vec, continuous])
    
    @property
    def vector_dim(self) -> int:
        """Return dimension of context vector."""
        return 16  # 4 + 6 + 6
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'time_of_day': self.time_of_day.value,
            'hour': self.hour,
            'minute': self.minute,
            'camera_zone': self.camera_zone.value,
            'camera_id': self.camera_id,
            'visibility': self.visibility.value,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'patrol_context': self.patrol_context.value,
            'alert_level': self.alert_level,
            'activity_modifier': self.activity_modifier,
            'sensitivity_modifier': self.sensitivity_modifier,
            'confidence_modifier': self.confidence_modifier,
        }


class ContextAugmenter:
    """
    Augments temporal windows with scene context.
    
    DESIGN PHILOSOPHY:
    ------------------
    Context is not labeled data - it's auxiliary signals that help
    the model understand WHEN and WHERE behavior is occurring.
    
    The same motion pattern might be:
    - Normal at noon on a road
    - Suspicious at midnight near a fence
    
    Context-aware normality enables this discrimination.
    """
    
    def __init__(
        self,
        default_camera_zone: CameraZone = CameraZone.OPEN_AREA,
        default_patrol_context: PatrolContext = PatrolContext.STATIC,
        base_timestamp: float = 0.0,          # Unix timestamp for start
        simulation_speed: float = 1.0,        # Time multiplier for demo
    ):
        self.default_camera_zone = default_camera_zone
        self.default_patrol_context = default_patrol_context
        self.base_timestamp = base_timestamp
        self.simulation_speed = simulation_speed
        
        # If no base timestamp, use current time
        if self.base_timestamp == 0.0:
            self.base_timestamp = datetime.now().timestamp()
        
        logger.info(f"ContextAugmenter initialized")
        logger.info(f"  Default zone: {default_camera_zone.value}")
        logger.info(f"  Patrol context: {default_patrol_context.value}")
    
    def estimate_visibility(
        self, 
        frames: np.ndarray
    ) -> Tuple[VisibilityLevel, float, float]:
        """
        Estimate visibility conditions from frame statistics.
        
        Uses brightness and contrast as proxies for visibility.
        Real systems would use dedicated sensors.
        
        Args:
            frames: Array of frames (T, H, W) or (T, H, W, C)
            
        Returns:
            (visibility_level, brightness, contrast)
        """
        if frames is None or len(frames) == 0:
            return VisibilityLevel.GOOD, 0.5, 0.5
        
        # Use middle frame for estimate
        mid_idx = len(frames) // 2
        frame = frames[mid_idx]
        
        # Handle color frames
        if len(frame.shape) == 3:
            frame = frame.mean(axis=-1)
        
        # Normalize to 0-1
        frame = frame.astype(float)
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        # Calculate brightness (mean intensity)
        brightness = float(np.mean(frame))
        
        # Calculate contrast (standard deviation)
        contrast = float(np.std(frame))
        
        # Map to visibility level
        if contrast > 0.25 and 0.2 < brightness < 0.8:
            visibility = VisibilityLevel.EXCELLENT
        elif contrast > 0.15 and 0.15 < brightness < 0.85:
            visibility = VisibilityLevel.GOOD
        elif contrast > 0.10:
            visibility = VisibilityLevel.MODERATE
        elif contrast > 0.05:
            visibility = VisibilityLevel.POOR
        else:
            visibility = VisibilityLevel.MINIMAL
        
        return visibility, brightness, contrast
    
    def get_time_context(
        self, 
        timestamp: float
    ) -> Tuple[TimeOfDay, int, int]:
        """
        Determine time context from timestamp.
        
        Args:
            timestamp: Unix timestamp or simulation time
            
        Returns:
            (time_of_day, hour, minute)
        """
        # Convert to datetime
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        minute = dt.minute
        
        time_of_day = TimeOfDay.from_hour(hour)
        
        return time_of_day, hour, minute
    
    def augment(
        self,
        frames: np.ndarray,
        start_timestamp: float,
        camera_zone: Optional[CameraZone] = None,
        patrol_context: Optional[PatrolContext] = None,
        camera_id: str = "CAM_001",
        alert_level: int = 0
    ) -> SceneContext:
        """
        Generate complete scene context for a temporal window.
        
        Args:
            frames: Frame array (T, H, W) or (T, H, W, C)
            start_timestamp: Window start time
            camera_zone: Camera zone (uses default if None)
            patrol_context: Patrol context (uses default if None)
            camera_id: Camera identifier
            alert_level: Current alert level (0-5)
            
        Returns:
            SceneContext with all contextual signals
        """
        # Use defaults if not specified
        if camera_zone is None:
            camera_zone = self.default_camera_zone
        if patrol_context is None:
            patrol_context = self.default_patrol_context
        
        # Calculate effective timestamp
        effective_ts = self.base_timestamp + (start_timestamp * self.simulation_speed)
        
        # Get time context
        time_of_day, hour, minute = self.get_time_context(effective_ts)
        
        # Estimate visibility from frames
        visibility, brightness, contrast = self.estimate_visibility(frames)
        
        # Calculate modifiers
        activity_modifier = (
            time_of_day.activity_modifier *
            camera_zone.expected_activity *
            patrol_context.baseline_modifier
        )
        
        sensitivity_modifier = (
            time_of_day.alert_sensitivity /
            camera_zone.alert_threshold_modifier
        )
        
        if alert_level > 0:
            sensitivity_modifier *= (1 + 0.2 * alert_level)
        
        confidence_modifier = visibility.confidence_modifier
        
        return SceneContext(
            time_of_day=time_of_day,
            hour=hour,
            minute=minute,
            camera_zone=camera_zone,
            camera_id=camera_id,
            visibility=visibility,
            brightness=brightness,
            contrast=contrast,
            patrol_context=patrol_context,
            alert_level=alert_level,
            activity_modifier=activity_modifier,
            sensitivity_modifier=sensitivity_modifier,
            confidence_modifier=confidence_modifier,
        )
    
    def augment_batch(
        self,
        windows: List,              # List of TemporalWindow
        camera_zone: Optional[CameraZone] = None,
        patrol_context: Optional[PatrolContext] = None,
        camera_id: str = "CAM_001"
    ) -> List:
        """
        Augment a batch of temporal windows with context.
        
        Args:
            windows: List of TemporalWindow objects
            camera_zone: Camera zone for all windows
            patrol_context: Patrol context for all windows
            camera_id: Camera identifier
            
        Returns:
            List of TemporalWindow with context added
        """
        augmented = []
        
        for window in windows:
            context = self.augment(
                frames=window.frames,
                start_timestamp=window.start_timestamp,
                camera_zone=camera_zone,
                patrol_context=patrol_context,
                camera_id=camera_id,
            )
            
            # Add context to window
            window.context = context.to_dict()
            augmented.append(window)
        
        return augmented


class ContextualFeatureBuilder:
    """
    Builds context-aware feature vectors for the LSTM-VAE model.
    
    Combines behavioral features with context features to create
    the complete input representation.
    """
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
    
    def build(
        self,
        behavioral_features: np.ndarray,    # (T, F_behavior)
        context: SceneContext
    ) -> np.ndarray:
        """
        Build complete feature vector with context.
        
        Args:
            behavioral_features: Behavioral feature sequence
            context: Scene context
            
        Returns:
            Combined feature sequence (T, F_behavior + F_context)
        """
        if not self.include_context:
            return behavioral_features
        
        # Get context vector
        context_vec = context.to_vector()  # (F_context,)
        
        # Expand context to match temporal dimension
        T = behavioral_features.shape[0]
        context_expanded = np.tile(context_vec, (T, 1))  # (T, F_context)
        
        # Concatenate
        combined = np.concatenate([behavioral_features, context_expanded], axis=1)
        
        return combined
    
    def get_feature_names(
        self,
        behavioral_feature_names: List[str]
    ) -> List[str]:
        """
        Get names for all features including context.
        """
        context_names = [
            'ctx_dawn', 'ctx_day', 'ctx_dusk', 'ctx_night',
            'ctx_fence', 'ctx_road', 'ctx_open', 'ctx_checkpoint', 
            'ctx_water', 'ctx_vegetation',
            'ctx_brightness', 'ctx_contrast', 'ctx_visibility',
            'ctx_activity_mod', 'ctx_sensitivity_mod', 'ctx_time_norm'
        ]
        
        if self.include_context:
            return behavioral_feature_names + context_names
        return behavioral_feature_names


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_context_for_demo(
    num_windows: int,
    camera_zone: str = "perimeter",
    simulate_day_night: bool = True,
    start_hour: int = 8
) -> List[SceneContext]:
    """
    Create context sequence for demonstration.
    
    Simulates time progression through a surveillance period.
    """
    augmenter = ContextAugmenter()
    
    # Map zone string to enum
    zone_map = {
        'perimeter': CameraZone.FENCE,
        'fence': CameraZone.FENCE,
        'road': CameraZone.ROAD,
        'open': CameraZone.OPEN_AREA,
        'checkpoint': CameraZone.CHECKPOINT,
    }
    zone = zone_map.get(camera_zone, CameraZone.OPEN_AREA)
    
    contexts = []
    
    # Simulate 1 minute per window
    for i in range(num_windows):
        # Calculate time
        minutes = start_hour * 60 + i
        if simulate_day_night:
            minutes = minutes % 1440  # Wrap at midnight
        
        hour = (minutes // 60) % 24
        minute = minutes % 60
        
        # Create timestamp
        dt = datetime.now().replace(hour=hour, minute=minute)
        timestamp = dt.timestamp()
        
        # Generate context
        context = augmenter.augment(
            frames=None,
            start_timestamp=timestamp,
            camera_zone=zone,
        )
        
        contexts.append(context)
    
    return contexts
