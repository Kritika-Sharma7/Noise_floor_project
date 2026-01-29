"""
NOISE FLOOR - Confidence-Based Risk Zone Classification
=========================================================
Graduated threat assessment for border surveillance operations.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

DESIGN PHILOSOPHY:
------------------
Traditional alarm systems use binary alerts: NORMAL or ALARM.

Problems with binary alerts:
1. Alert fatigue from false positives
2. Late detection (waiting for "certain" anomaly)
3. No early warning capability
4. No context for decision-making

CONFIDENCE-BASED RISK ZONES:
----------------------------
🟢 NORMAL   - Stable behavior within expected bounds
🟡 WATCH    - Weak but persistent drift detected
🟠 WARNING  - Confirmed behavioral drift requiring attention
🔴 CRITICAL - High-confidence threat pattern detected

KEY PRINCIPLE:
"Defense systems manage CONFIDENCE, not panic."

THRESHOLD DESIGN:
-----------------
Thresholds are:
- Adaptive (based on recent baseline)
- Trend-based (consider direction of change)
- Smoothed (avoid oscillation between zones)
- Hysteresis-enabled (prevent rapid zone flipping)
"""

import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskZone(Enum):
    """
    Risk zone levels with associated operational semantics.
    
    Each zone maps to specific operator actions and
    resource allocation decisions.
    """
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"
    
    def __str__(self):
        return self.value.upper()
    
    @property
    def level(self) -> int:
        """Numeric level for comparison."""
        levels = {
            RiskZone.NORMAL: 0,
            RiskZone.WATCH: 1,
            RiskZone.WARNING: 2,
            RiskZone.CRITICAL: 3,
        }
        return levels[self]
    
    @property
    def color(self) -> str:
        """Color code for visualization."""
        colors = {
            RiskZone.NORMAL: "#10B981",     # Green
            RiskZone.WATCH: "#F59E0B",      # Yellow/Amber
            RiskZone.WARNING: "#F97316",    # Orange
            RiskZone.CRITICAL: "#EF4444",   # Red
        }
        return colors[self]
    
    @property
    def icon(self) -> str:
        """Emoji icon for display."""
        icons = {
            RiskZone.NORMAL: "🟢",
            RiskZone.WATCH: "🟡",
            RiskZone.WARNING: "🟠",
            RiskZone.CRITICAL: "🔴",
        }
        return icons[self]
    
    @property
    def action(self) -> str:
        """Recommended operator action."""
        actions = {
            RiskZone.NORMAL: "Continue standard monitoring protocols",
            RiskZone.WATCH: "Increase observation frequency, prepare response assets",
            RiskZone.WARNING: "Alert response team, initiate verification procedures",
            RiskZone.CRITICAL: "Immediate action required, dispatch response units",
        }
        return actions[self]
    
    @property
    def priority(self) -> str:
        """Priority level for logging/dispatch."""
        priorities = {
            RiskZone.NORMAL: "routine",
            RiskZone.WATCH: "elevated",
            RiskZone.WARNING: "high",
            RiskZone.CRITICAL: "urgent",
        }
        return priorities[self]


@dataclass
class ZoneState:
    """
    Complete risk zone state with all relevant context.
    """
    # Current zone
    zone: RiskZone = RiskZone.NORMAL
    
    # Input metrics
    threat_deviation_index: float = 0.0
    z_score: float = 0.0
    trend_slope: float = 0.0
    trend_persistence: int = 0
    
    # Zone stability
    duration_in_zone: int = 0              # Frames in current zone
    previous_zone: Optional[RiskZone] = None
    zone_change_frame: Optional[int] = None
    
    # Confidence metrics
    classification_confidence: float = 1.0  # How confident in this zone
    stability_score: float = 1.0            # How stable is the classification
    
    # Thresholds (for transparency)
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Trend assessment
    escalating: bool = False               # Moving to higher zone
    de_escalating: bool = False            # Moving to lower zone
    
    # Temporal info
    frame_index: int = 0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'zone': str(self.zone),
            'zone_value': self.zone.value,
            'icon': self.zone.icon,
            'color': self.zone.color,
            'action': self.zone.action,
            'priority': self.zone.priority,
            'threat_deviation_index': self.threat_deviation_index,
            'z_score': self.z_score,
            'duration_in_zone': self.duration_in_zone,
            'classification_confidence': self.classification_confidence,
            'stability_score': self.stability_score,
            'escalating': self.escalating,
            'de_escalating': self.de_escalating,
            'thresholds': self.thresholds,
            'frame_index': self.frame_index,
        }


@dataclass
class AdaptiveThresholds:
    """
    Adaptive thresholds that adjust based on baseline and recent history.
    """
    # Base thresholds (in standard deviations)
    normal_upper: float = 1.5      # Upper bound for NORMAL zone
    watch_upper: float = 2.5       # Upper bound for WATCH zone
    warning_upper: float = 3.5     # Upper bound for WARNING zone
    # Above warning_upper = CRITICAL
    
    # Trend modifiers
    trend_escalation_factor: float = 0.8   # Lower thresholds if trend is up
    trend_deescalation_factor: float = 1.2 # Raise thresholds if trend is down
    
    # Persistence modifiers
    persistence_factor: float = 0.95       # Lower thresholds with persistence
    
    # Hysteresis (prevents zone flipping)
    hysteresis_up: float = 0.3     # Extra needed to escalate
    hysteresis_down: float = 0.5   # Extra needed to de-escalate
    
    # Minimum durations (frames) to confirm zone
    min_duration_watch: int = 5
    min_duration_warning: int = 10
    min_duration_critical: int = 3


class RiskZoneClassifier:
    """
    Confidence-based risk zone classifier with adaptive thresholds.
    
    CLASSIFICATION PIPELINE:
    -----------------------
    1. Receive TDI and supporting metrics from Drift Intelligence
    2. Apply adaptive thresholds based on context
    3. Apply hysteresis to prevent zone flipping
    4. Require minimum duration for zone confirmation
    5. Output zone classification with confidence
    
    THRESHOLD ADAPTATION:
    --------------------
    - Thresholds adjust based on trend direction
    - Persistent deviation lowers thresholds
    - Time-of-day can modify sensitivity (via context)
    - Recent baseline variance affects thresholds
    
    This system is designed for border surveillance and high-security 
    perimeters where threats emerge gradually.
    """
    
    def __init__(
        self,
        # Base thresholds (Z-score based)
        normal_threshold: float = 1.5,
        watch_threshold: float = 2.5,
        warning_threshold: float = 3.5,
        
        # Hysteresis parameters
        hysteresis_up: float = 0.3,
        hysteresis_down: float = 0.5,
        
        # Minimum zone durations
        min_duration: int = 5,
        
        # Smoothing
        smoothing_window: int = 10,
        
        # TDI thresholds (alternative to Z-score)
        tdi_normal: float = 20.0,
        tdi_watch: float = 40.0,
        tdi_warning: float = 60.0,
    ):
        # Z-score thresholds
        self.thresholds = AdaptiveThresholds(
            normal_upper=normal_threshold,
            watch_upper=watch_threshold,
            warning_upper=warning_threshold,
            hysteresis_up=hysteresis_up,
            hysteresis_down=hysteresis_down,
            min_duration_watch=min_duration,
            min_duration_warning=min_duration * 2,
            min_duration_critical=min_duration // 2,
        )
        
        # TDI thresholds
        self.tdi_thresholds = {
            'normal': tdi_normal,
            'watch': tdi_watch,
            'warning': tdi_warning,
        }
        
        # State
        self.current_zone = RiskZone.NORMAL
        self.zone_duration = 0
        self.zone_history = deque(maxlen=1000)
        self.score_history = deque(maxlen=smoothing_window)
        
        self.frame_count = 0
        self.last_zone_change = 0
        
        # Transition tracking
        self.zone_transitions: List[Tuple[int, RiskZone, RiskZone]] = []
        
        logger.info(f"RiskZoneClassifier initialized")
        logger.info(f"  Thresholds (Z): normal<{normal_threshold}, watch<{watch_threshold}, warning<{warning_threshold}")
        logger.info(f"  Thresholds (TDI): normal<{tdi_normal}, watch<{tdi_watch}, warning<{tdi_warning}")
    
    def _get_adaptive_thresholds(
        self,
        trend_slope: float,
        trend_persistence: int,
        sensitivity_modifier: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute adaptive thresholds based on current context.
        
        Thresholds are LOWERED (more sensitive) when:
        - Trend is increasing (escalating threat)
        - Deviation has persisted for long time
        - Sensitivity modifier is high (e.g., night time)
        
        Thresholds are RAISED (less sensitive) when:
        - Trend is decreasing (de-escalating)
        - Deviation is new/sporadic
        """
        base = self.thresholds
        
        # Start with base thresholds
        normal = base.normal_upper
        watch = base.watch_upper
        warning = base.warning_upper
        
        # Trend modifier
        if trend_slope > 0.1:
            # Escalating - lower thresholds (more sensitive)
            modifier = base.trend_escalation_factor
        elif trend_slope < -0.1:
            # De-escalating - raise thresholds (less sensitive)
            modifier = base.trend_deescalation_factor
        else:
            modifier = 1.0
        
        normal *= modifier
        watch *= modifier
        warning *= modifier
        
        # Persistence modifier - lower thresholds with sustained deviation
        if trend_persistence > 10:
            persistence_mod = base.persistence_factor ** (trend_persistence / 10)
            normal *= persistence_mod
            watch *= persistence_mod
            warning *= persistence_mod
        
        # Apply sensitivity modifier (from context)
        normal /= sensitivity_modifier
        watch /= sensitivity_modifier
        warning /= sensitivity_modifier
        
        return {
            'normal': normal,
            'watch': watch,
            'warning': warning,
        }
    
    def _apply_hysteresis(
        self,
        raw_zone: RiskZone,
        score: float,
        thresholds: Dict[str, float]
    ) -> RiskZone:
        """
        Apply hysteresis to prevent zone flipping.
        
        To ESCALATE (move to higher zone): Score must exceed threshold + hysteresis_up
        To DE-ESCALATE (move to lower zone): Score must drop below threshold - hysteresis_down
        """
        current_level = self.current_zone.level
        raw_level = raw_zone.level
        
        if raw_level > current_level:
            # Attempting to escalate - apply upward hysteresis
            # Need extra margin to escalate
            if raw_level == 1:  # NORMAL -> WATCH
                needed = thresholds['normal'] + self.thresholds.hysteresis_up
            elif raw_level == 2:  # WATCH -> WARNING
                needed = thresholds['watch'] + self.thresholds.hysteresis_up
            else:  # WARNING -> CRITICAL
                needed = thresholds['warning'] + self.thresholds.hysteresis_up
            
            if score >= needed:
                return raw_zone
            else:
                return self.current_zone
        
        elif raw_level < current_level:
            # Attempting to de-escalate - apply downward hysteresis
            # Need to drop well below threshold
            if current_level == 1:  # WATCH -> NORMAL
                needed = thresholds['normal'] - self.thresholds.hysteresis_down
            elif current_level == 2:  # WARNING -> WATCH
                needed = thresholds['watch'] - self.thresholds.hysteresis_down
            else:  # CRITICAL -> WARNING
                needed = thresholds['warning'] - self.thresholds.hysteresis_down
            
            if score <= needed:
                return raw_zone
            else:
                return self.current_zone
        
        return raw_zone
    
    def _check_duration_requirement(self, proposed_zone: RiskZone) -> bool:
        """
        Check if minimum duration requirement is met for zone change.
        
        Some zones require sustained deviation before confirmation.
        This prevents brief spikes from triggering alerts.
        """
        if proposed_zone == self.current_zone:
            return True  # Staying in same zone - always OK
        
        # Check pending duration
        if proposed_zone == RiskZone.WATCH:
            return self.zone_duration >= self.thresholds.min_duration_watch
        elif proposed_zone == RiskZone.WARNING:
            return self.zone_duration >= self.thresholds.min_duration_warning
        elif proposed_zone == RiskZone.CRITICAL:
            return self.zone_duration >= self.thresholds.min_duration_critical
        
        return True  # De-escalation doesn't require duration
    
    def classify(
        self,
        threat_deviation_index: float,
        z_score: float = 0.0,
        trend_slope: float = 0.0,
        trend_persistence: int = 0,
        sensitivity_modifier: float = 1.0,
        frame_index: int = 0,
        timestamp: float = 0.0,
        use_tdi: bool = True
    ) -> ZoneState:
        """
        Classify current state into risk zone.
        
        PRIMARY CLASSIFICATION METHOD.
        
        Args:
            threat_deviation_index: TDI from drift intelligence (0-100)
            z_score: Z-score from drift intelligence
            trend_slope: Rate of change
            trend_persistence: Frames of sustained deviation
            sensitivity_modifier: Context-based sensitivity adjustment
            frame_index: Current frame index
            timestamp: Current timestamp
            use_tdi: Whether to use TDI or Z-score for classification
            
        Returns:
            ZoneState with complete classification
        """
        self.frame_count += 1
        
        # Choose primary score
        if use_tdi:
            score = threat_deviation_index
            thresholds = self.tdi_thresholds
        else:
            score = z_score
            thresholds = self._get_adaptive_thresholds(
                trend_slope, trend_persistence, sensitivity_modifier
            )
        
        # Add to history for smoothing
        self.score_history.append(score)
        
        # Use smoothed score for classification
        smoothed_score = np.mean(list(self.score_history))
        
        # ===== RAW ZONE CLASSIFICATION =====
        if use_tdi:
            if smoothed_score < thresholds['normal']:
                raw_zone = RiskZone.NORMAL
            elif smoothed_score < thresholds['watch']:
                raw_zone = RiskZone.WATCH
            elif smoothed_score < thresholds['warning']:
                raw_zone = RiskZone.WARNING
            else:
                raw_zone = RiskZone.CRITICAL
        else:
            if smoothed_score < thresholds['normal']:
                raw_zone = RiskZone.NORMAL
            elif smoothed_score < thresholds['watch']:
                raw_zone = RiskZone.WATCH
            elif smoothed_score < thresholds['warning']:
                raw_zone = RiskZone.WARNING
            else:
                raw_zone = RiskZone.CRITICAL
        
        # ===== APPLY HYSTERESIS =====
        proposed_zone = self._apply_hysteresis(raw_zone, smoothed_score, thresholds)
        
        # ===== CHECK DURATION REQUIREMENT =====
        # Track pending zone
        if proposed_zone != self.current_zone:
            # We're in a transition period - increment duration
            self.zone_duration += 1
            
            if self._check_duration_requirement(proposed_zone):
                # Duration met - confirm zone change
                previous_zone = self.current_zone
                self.current_zone = proposed_zone
                self.zone_duration = 0
                self.last_zone_change = frame_index
                
                # Record transition
                self.zone_transitions.append((frame_index, previous_zone, proposed_zone))
                
                logger.info(
                    f"Zone change: {previous_zone.icon} {previous_zone} → "
                    f"{proposed_zone.icon} {proposed_zone} at frame {frame_index}"
                )
        else:
            self.zone_duration += 1
        
        # ===== COMPUTE CONFIDENCE =====
        # Confidence is higher when:
        # - Score is far from threshold boundaries
        # - Zone has been stable for a while
        
        # Distance from nearest threshold
        if self.current_zone == RiskZone.NORMAL:
            threshold_dist = thresholds['normal'] - smoothed_score
        elif self.current_zone == RiskZone.WATCH:
            threshold_dist = min(
                smoothed_score - thresholds['normal'],
                thresholds['watch'] - smoothed_score
            )
        elif self.current_zone == RiskZone.WARNING:
            threshold_dist = min(
                smoothed_score - thresholds['watch'],
                thresholds['warning'] - smoothed_score
            )
        else:  # CRITICAL
            threshold_dist = smoothed_score - thresholds['warning']
        
        classification_confidence = min(1.0, abs(threshold_dist) / 1.0)
        
        # Stability score based on duration
        stability_score = min(1.0, self.zone_duration / 30.0)
        
        # ===== DETERMINE ESCALATION STATE =====
        escalating = raw_zone.level > self.current_zone.level
        de_escalating = raw_zone.level < self.current_zone.level
        
        # ===== CREATE OUTPUT =====
        state = ZoneState(
            zone=self.current_zone,
            threat_deviation_index=threat_deviation_index,
            z_score=z_score,
            trend_slope=trend_slope,
            trend_persistence=trend_persistence,
            duration_in_zone=self.zone_duration,
            previous_zone=self.zone_transitions[-1][1] if self.zone_transitions else None,
            zone_change_frame=self.last_zone_change if self.zone_transitions else None,
            classification_confidence=classification_confidence,
            stability_score=stability_score,
            thresholds=thresholds,
            escalating=escalating,
            de_escalating=de_escalating,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        
        # Store in history
        self.zone_history.append(state)
        
        return state
    
    def get_zone_summary(self) -> Dict:
        """Get summary of zone transitions and current state."""
        if not self.zone_history:
            return {'current_zone': 'unknown', 'transitions': 0}
        
        return {
            'current_zone': str(self.current_zone),
            'current_zone_icon': self.current_zone.icon,
            'current_zone_color': self.current_zone.color,
            'current_zone_action': self.current_zone.action,
            'duration_in_zone': self.zone_duration,
            'total_frames': self.frame_count,
            'num_transitions': len(self.zone_transitions),
            'transitions': [
                {
                    'frame': t[0],
                    'from': str(t[1]),
                    'to': str(t[2]),
                    'from_icon': t[1].icon,
                    'to_icon': t[2].icon,
                }
                for t in self.zone_transitions[-10:]  # Last 10 transitions
            ],
            'time_in_zones': self._compute_time_in_zones(),
        }
    
    def _compute_time_in_zones(self) -> Dict[str, float]:
        """Compute percentage of time in each zone."""
        if not self.zone_history:
            return {}
        
        counts = {zone: 0 for zone in RiskZone}
        for state in self.zone_history:
            counts[state.zone] += 1
        
        total = len(self.zone_history)
        return {
            str(zone): count / total * 100
            for zone, count in counts.items()
        }
    
    def reset(self):
        """Reset classifier state."""
        self.current_zone = RiskZone.NORMAL
        self.zone_duration = 0
        self.zone_history.clear()
        self.score_history.clear()
        self.zone_transitions = []
        self.frame_count = 0
        self.last_zone_change = 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_zone_classifier(
    sensitivity: str = "balanced"
) -> RiskZoneClassifier:
    """
    Create a risk zone classifier with preset sensitivity.
    
    Args:
        sensitivity: "low", "balanced", or "high"
        
    Returns:
        Configured RiskZoneClassifier
    """
    presets = {
        'low': {
            'normal_threshold': 2.0,
            'watch_threshold': 3.0,
            'warning_threshold': 4.0,
            'tdi_normal': 30.0,
            'tdi_watch': 50.0,
            'tdi_warning': 70.0,
        },
        'balanced': {
            'normal_threshold': 1.5,
            'watch_threshold': 2.5,
            'warning_threshold': 3.5,
            'tdi_normal': 20.0,
            'tdi_watch': 40.0,
            'tdi_warning': 60.0,
        },
        'high': {
            'normal_threshold': 1.0,
            'watch_threshold': 2.0,
            'warning_threshold': 3.0,
            'tdi_normal': 15.0,
            'tdi_watch': 30.0,
            'tdi_warning': 50.0,
        },
    }
    
    params = presets.get(sensitivity, presets['balanced'])
    
    return RiskZoneClassifier(**params)
