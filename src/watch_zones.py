"""
NOISE FLOOR - Watch Zones Module
==================================
Confidence-based alert levels for operational monitoring.

GRAY-BOX EXPLANATION:
---------------------
Traditional alarm systems use binary alerts: NORMAL or ALARM.

Problems with binary alerts:
1. Alert fatigue from false positives
2. Late detection (waiting for "certain" anomaly)
3. No early warning capability
4. No context for decision-making

WATCH ZONES provide graduated confidence levels:
- 🟢 NORMAL: Behavior within expected range
- 🟡 WATCH: Early weak drift - worth monitoring
- 🟠 WARNING: Sustained drift - prepare response
- 🔴 ALERT: Strong deviation - action needed

WHY THIS DESIGN:
1. Reduces false positive impact (watch ≠ alarm)
2. Enables PREVENTIVE action before incidents
3. Provides context for operators
4. Matches real-world decision processes
"""

import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Zone(Enum):
    """
    Alert zone levels with associated semantics.
    
    Each zone has:
    - Name for display
    - Color for visualization
    - Action recommendation
    """
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    ALERT = "alert"
    
    def __str__(self):
        return self.value.upper()
    
    @property
    def color(self) -> str:
        """Return color code for visualization."""
        colors = {
            Zone.NORMAL: "#00C851",   # Green
            Zone.WATCH: "#FFBB33",    # Yellow
            Zone.WARNING: "#FF8800",  # Orange
            Zone.ALERT: "#FF4444",    # Red
        }
        return colors[self]
    
    @property
    def icon(self) -> str:
        """Return emoji icon for display."""
        icons = {
            Zone.NORMAL: "🟢",
            Zone.WATCH: "🟡",
            Zone.WARNING: "🟠",
            Zone.ALERT: "🔴",
        }
        return icons[self]
    
    @property
    def action(self) -> str:
        """Return recommended action."""
        actions = {
            Zone.NORMAL: "Continue normal monitoring",
            Zone.WATCH: "Increase observation frequency",
            Zone.WARNING: "Prepare response resources",
            Zone.ALERT: "Immediate investigation required",
        }
        return actions[self]


@dataclass
class ZoneState:
    """
    Current zone state with all relevant information.
    """
    zone: Zone
    drift_score: float
    confidence: float           # How confident we are in this zone
    duration: int               # Frames in current zone
    trend: str                  # "stable", "escalating", "de-escalating"
    thresholds: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'zone': str(self.zone),
            'icon': self.zone.icon,
            'color': self.zone.color,
            'drift_score': self.drift_score,
            'confidence': self.confidence,
            'duration': self.duration,
            'trend': self.trend,
            'action': self.zone.action,
        }


class WatchZoneClassifier:
    """
    Classifies drift scores into watch zones.
    
    THRESHOLD DESIGN:
    -----------------
    Thresholds are based on standard deviations from baseline:
    - NORMAL: score < 1.5σ (within normal variation)
    - WATCH: 1.5σ ≤ score < 2.0σ (starting to deviate)
    - WARNING: 2.0σ ≤ score < 2.5σ (significant deviation)
    - ALERT: score ≥ 2.5σ (strong deviation)
    
    These thresholds can be tuned based on:
    - Operational requirements
    - False positive tolerance
    - Response capability
    """
    
    def __init__(
        self,
        normal_threshold: float = 1.5,
        watch_threshold: float = 2.0,
        warning_threshold: float = 2.5,
        alert_threshold: float = 3.0,
        min_zone_duration: int = 10,
        hysteresis: float = 0.1
    ):
        """
        Initialize zone classifier.
        
        Args:
            normal_threshold: Upper bound for normal zone
            watch_threshold: Upper bound for watch zone
            warning_threshold: Upper bound for warning zone
            alert_threshold: Threshold for alert zone
            min_zone_duration: Minimum frames to confirm zone change
            hysteresis: Buffer to prevent zone flickering
        """
        self.thresholds = {
            'normal': normal_threshold,
            'watch': watch_threshold,
            'warning': warning_threshold,
            'alert': alert_threshold,
        }
        self.min_zone_duration = min_zone_duration
        self.hysteresis = hysteresis
        
        # State
        self.current_zone = Zone.NORMAL
        self.zone_duration = 0
        self.pending_zone = None
        self.pending_duration = 0
        self.history = []
        
        logger.info(f"WatchZoneClassifier initialized with thresholds: {self.thresholds}")
    
    def reset(self):
        """Reset classifier state."""
        self.current_zone = Zone.NORMAL
        self.zone_duration = 0
        self.pending_zone = None
        self.pending_duration = 0
        self.history = []
    
    def _score_to_zone(self, drift_score: float) -> Zone:
        """
        Map drift score to zone without hysteresis.
        
        ZONE LOGIC:
        -----------
        score < normal_threshold → NORMAL
        normal_threshold ≤ score < watch_threshold → WATCH
        watch_threshold ≤ score < warning_threshold → WARNING
        score ≥ warning_threshold → ALERT
        """
        if drift_score < self.thresholds['normal']:
            return Zone.NORMAL
        elif drift_score < self.thresholds['watch']:
            return Zone.WATCH
        elif drift_score < self.thresholds['warning']:
            return Zone.WARNING
        else:
            return Zone.ALERT
    
    def _apply_hysteresis(
        self,
        raw_zone: Zone,
        drift_score: float
    ) -> Zone:
        """
        Apply hysteresis to prevent zone flickering.
        
        HYSTERESIS EXPLANATION:
        -----------------------
        Without hysteresis, if score hovers around a threshold,
        the zone would rapidly switch back and forth.
        
        Hysteresis adds a buffer: to exit a zone, score must
        move further than the threshold by the hysteresis amount.
        
        Example with hysteresis=0.1:
        - Enter WATCH at score ≥ 1.5
        - Exit WATCH only when score < 1.4
        """
        current_level = list(Zone).index(self.current_zone)
        raw_level = list(Zone).index(raw_zone)
        
        # Escalating (to higher alert level)
        if raw_level > current_level:
            return raw_zone
        
        # De-escalating (to lower alert level) - apply hysteresis
        elif raw_level < current_level:
            # Check if score is significantly below current zone threshold
            current_zone_name = self.current_zone.value
            if current_zone_name == 'alert':
                threshold = self.thresholds['warning']
            elif current_zone_name == 'warning':
                threshold = self.thresholds['watch']
            elif current_zone_name == 'watch':
                threshold = self.thresholds['normal']
            else:
                threshold = 0
            
            if drift_score < threshold - self.hysteresis:
                return raw_zone
            else:
                return self.current_zone
        
        return raw_zone
    
    def _compute_confidence(self, drift_score: float, zone: Zone) -> float:
        """
        Compute confidence in the current zone classification.
        
        CONFIDENCE EXPLANATION:
        -----------------------
        Confidence indicates how "deep" we are in the current zone.
        
        - Score near threshold → low confidence
        - Score well into zone → high confidence
        
        IMPORTANT: Confidence is NOT probability of threat.
        It reflects "strength within zone", capped at 95%.
        No real system claims 100% certainty.
        """
        MAX_CONFIDENCE = 0.95  # Never claim 100% certainty
        
        if zone == Zone.NORMAL:
            # Higher confidence the lower the score
            conf = 1.0 - (drift_score / self.thresholds['normal']) if self.thresholds['normal'] > 0 else 1.0
            return min(MAX_CONFIDENCE, max(0.0, conf))
        
        elif zone == Zone.WATCH:
            # Confidence based on position within watch zone
            zone_range = self.thresholds['watch'] - self.thresholds['normal']
            if zone_range <= 0:
                return 0.5
            position = (drift_score - self.thresholds['normal']) / zone_range
            return min(MAX_CONFIDENCE, max(0.0, position))
        
        elif zone == Zone.WARNING:
            # Confidence based on position within warning zone
            zone_range = self.thresholds['warning'] - self.thresholds['watch']
            if zone_range <= 0:
                return 0.5
            position = (drift_score - self.thresholds['watch']) / zone_range
            return min(MAX_CONFIDENCE, max(0.0, position))
        
        else:  # ALERT
            # Higher score = higher confidence, but capped
            excess = drift_score - self.thresholds['warning']
            conf = 0.5 + min(0.45, excess / 4)  # Max out at 95%
            return min(MAX_CONFIDENCE, max(0.0, conf))
    
    def _get_trend(self) -> str:
        """
        Determine zone trend from recent history.
        
        TREND OPTIONS:
        - "escalating": Moving toward higher alert zones
        - "de-escalating": Moving toward lower alert zones
        - "stable": Staying in same zone
        """
        if len(self.history) < 5:
            return "stable"
        
        recent_zones = [h.zone for h in self.history[-10:]]
        zone_levels = [list(Zone).index(z) for z in recent_zones]
        
        # Simple trend: compare first half to second half
        first_half = np.mean(zone_levels[:len(zone_levels)//2])
        second_half = np.mean(zone_levels[len(zone_levels)//2:])
        
        diff = second_half - first_half
        if diff > 0.3:
            return "escalating"
        elif diff < -0.3:
            return "de-escalating"
        else:
            return "stable"
    
    def classify(self, drift_score: float) -> ZoneState:
        """
        Classify drift score into a zone with persistence.
        
        PERSISTENCE LOGIC:
        ------------------
        To prevent rapid zone switching, we require:
        1. Score must indicate new zone
        2. Must persist for min_zone_duration frames
        3. Then zone officially changes
        
        This ensures we don't overreact to momentary fluctuations.
        """
        # Get raw zone from score
        raw_zone = self._score_to_zone(drift_score)
        
        # Apply hysteresis
        target_zone = self._apply_hysteresis(raw_zone, drift_score)
        
        # Handle zone persistence
        if target_zone == self.current_zone:
            # Still in same zone
            self.zone_duration += 1
            self.pending_zone = None
            self.pending_duration = 0
        
        elif target_zone == self.pending_zone:
            # Confirming pending zone change
            self.pending_duration += 1
            if self.pending_duration >= self.min_zone_duration:
                # Confirm zone change
                logger.info(f"Zone change: {self.current_zone} → {target_zone}")
                self.current_zone = target_zone
                self.zone_duration = self.pending_duration
                self.pending_zone = None
                self.pending_duration = 0
        
        else:
            # New pending zone
            self.pending_zone = target_zone
            self.pending_duration = 1
        
        # Compute confidence and trend
        confidence = self._compute_confidence(drift_score, self.current_zone)
        trend = self._get_trend()
        
        # Create state
        state = ZoneState(
            zone=self.current_zone,
            drift_score=drift_score,
            confidence=confidence,
            duration=self.zone_duration,
            trend=trend,
            thresholds=self.thresholds.copy(),
        )
        
        self.history.append(state)
        
        return state
    
    def get_zone_statistics(self) -> Dict:
        """
        Get statistics about time spent in each zone.
        
        Useful for reports and analysis.
        """
        if not self.history:
            return {}
        
        zone_counts = {zone: 0 for zone in Zone}
        for state in self.history:
            zone_counts[state.zone] += 1
        
        total = len(self.history)
        zone_percentages = {
            zone.value: count / total * 100
            for zone, count in zone_counts.items()
        }
        
        return {
            'total_frames': total,
            'zone_counts': {z.value: c for z, c in zone_counts.items()},
            'zone_percentages': zone_percentages,
            'current_zone': str(self.current_zone),
            'max_zone_reached': str(max(zone_counts.keys(), key=lambda z: list(Zone).index(z))),
        }


class AdaptiveWatchZones(WatchZoneClassifier):
    """
    Watch zones with adaptive thresholds.
    
    ADAPTIVE EXPLANATION:
    ---------------------
    In long-term monitoring, what's "normal" may slowly change.
    Adaptive thresholds adjust to slow environmental changes
    while still detecting rapid behavioral drift.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.001,
        min_samples: int = 500,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        self.score_history = []
    
    def classify(self, drift_score: float) -> ZoneState:
        self.score_history.append(drift_score)
        
        # Adapt thresholds after enough samples
        if len(self.score_history) >= self.min_samples:
            self._adapt_thresholds()
        
        return super().classify(drift_score)
    
    def _adapt_thresholds(self):
        """Slowly adapt thresholds based on observed score distribution."""
        recent_scores = np.array(self.score_history[-1000:])
        
        # Only adapt if mostly in normal zone
        normal_count = sum(1 for s in recent_scores if s < self.thresholds['normal'])
        if normal_count / len(recent_scores) > 0.7:
            # Compute new thresholds based on percentiles
            new_normal = np.percentile(recent_scores, 85)
            new_watch = np.percentile(recent_scores, 90)
            new_warning = np.percentile(recent_scores, 95)
            
            # Slow adaptation
            self.thresholds['normal'] = (
                (1 - self.adaptation_rate) * self.thresholds['normal'] +
                self.adaptation_rate * new_normal
            )
            self.thresholds['watch'] = (
                (1 - self.adaptation_rate) * self.thresholds['watch'] +
                self.adaptation_rate * new_watch
            )
            self.thresholds['warning'] = (
                (1 - self.adaptation_rate) * self.thresholds['warning'] +
                self.adaptation_rate * new_warning
            )


def create_zone_visualization_data(states: List[ZoneState]) -> Dict:
    """
    Create data structure for zone visualization.
    
    Returns data suitable for plotting zone transitions over time.
    """
    return {
        'frames': list(range(len(states))),
        'drift_scores': [s.drift_score for s in states],
        'zones': [s.zone.value for s in states],
        'colors': [s.zone.color for s in states],
        'confidences': [s.confidence for s in states],
        'thresholds': states[0].thresholds if states else {},
    }


if __name__ == "__main__":
    # Test watch zones
    print("Testing Watch Zones Module")
    print("=" * 50)
    
    # Create synthetic drift scores (simulating gradual drift)
    np.random.seed(42)
    
    # Normal period
    normal_scores = np.random.normal(0.5, 0.3, 100)
    
    # Gradual escalation
    escalation = np.linspace(0.5, 3.5, 150) + np.random.normal(0, 0.2, 150)
    
    # De-escalation
    de_escalation = np.linspace(3.5, 1.0, 50) + np.random.normal(0, 0.2, 50)
    
    all_scores = np.concatenate([normal_scores, escalation, de_escalation])
    
    # Process with zone classifier
    classifier = WatchZoneClassifier()
    states = []
    
    for score in all_scores:
        state = classifier.classify(score)
        states.append(state)
    
    # Show results
    print("\nZone Transitions:")
    current_zone = None
    for i, state in enumerate(states):
        if state.zone != current_zone:
            print(f"  Frame {i}: {state.zone.icon} {state.zone} "
                  f"(score={state.drift_score:.2f}, confidence={state.confidence:.2f})")
            current_zone = state.zone
    
    # Statistics
    stats = classifier.get_zone_statistics()
    print(f"\nZone Statistics:")
    for zone, pct in stats['zone_percentages'].items():
        print(f"  {zone.upper()}: {pct:.1f}%")
    
    print("\nWatch zones module ready!")
