"""
NOISE FLOOR - Drift Detection Module
=====================================
Temporal analysis for detecting gradual behavioral drift.

GRAY-BOX EXPLANATION:
---------------------
This is the CORE innovation of NOISE FLOOR.

TRADITIONAL ANOMALY DETECTION:
- Asks: "Is this frame abnormal?"
- Triggers on single high-error frames
- High false positive rate
- Misses gradual changes

NOISE FLOOR DRIFT DETECTION:
- Asks: "Is behavior gradually changing over time?"
- Aggregates errors over sliding windows
- Uses EWMA to track trends
- Detects SLOW, HIDDEN deviations

KEY CONCEPTS:
1. SLIDING WINDOW: Aggregate multiple frames to reduce noise
2. EWMA (Exponentially Weighted Moving Average): Track trends over time
3. TREND SLOPE: Measure the RATE of change
4. DRIFT SCORE: Combination of deviation magnitude and trend

WHY THIS MATTERS FOR DEFENSE:
- Insider threats: Gradual behavioral shift
- Equipment degradation: Slow performance decline
- Surveillance: Crowd behavior changes before incidents
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftState:
    """
    Current state of drift detection.
    Tracks all relevant metrics for explainability.
    """
    # Raw normality score (from autoencoder)
    raw_score: float = 0.0
    
    # Smoothed score (EWMA)
    smoothed_score: float = 0.0
    
    # Trend metrics
    trend_slope: float = 0.0          # Rate of change
    trend_direction: str = "stable"    # "increasing", "decreasing", "stable"
    
    # Aggregated metrics
    window_mean: float = 0.0          # Mean over sliding window
    window_std: float = 0.0           # Std over sliding window
    window_max: float = 0.0           # Max over sliding window
    
    # Final drift score
    drift_score: float = 0.0          # Combined metric
    
    # Temporal info
    frame_index: int = 0
    timestamp: float = 0.0
    
    # History for visualization
    score_history: List[float] = field(default_factory=list)


class DriftDetector:
    """
    Detects gradual behavioral drift from normality scores.
    
    ALGORITHM OVERVIEW:
    -------------------
    1. Receive normality scores from autoencoder
    2. Apply EWMA smoothing to reduce noise
    3. Maintain sliding window of recent scores
    4. Compute trend slope using linear regression
    5. Combine magnitude and trend into drift score
    
    DRIFT SCORE FORMULA:
    drift_score = α * smoothed_magnitude + β * trend_slope + γ * window_variance
    
    Where:
    - smoothed_magnitude: How far from normal (EWMA smoothed)
    - trend_slope: How fast it's changing
    - window_variance: How unstable the behavior is
    """
    
    def __init__(
        self,
        window_size: int = 30,
        ewma_alpha: float = 0.1,
        trend_window: int = 50,
        baseline_frames: int = 200,
        magnitude_weight: float = 0.5,
        trend_weight: float = 0.3,
        variance_weight: float = 0.2
    ):
        """
        Initialize drift detector.
        
        Args:
            window_size: Frames in sliding window for aggregation
            ewma_alpha: EWMA smoothing factor (lower = smoother)
            trend_window: Frames for trend calculation
            baseline_frames: Frames to establish baseline
            magnitude_weight: Weight for score magnitude in drift score
            trend_weight: Weight for trend slope in drift score
            variance_weight: Weight for variance in drift score
        """
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.trend_window = trend_window
        self.baseline_frames = baseline_frames
        
        # Drift score weights
        self.magnitude_weight = magnitude_weight
        self.trend_weight = trend_weight
        self.variance_weight = variance_weight
        
        # State
        self.reset()
        
        logger.info(f"DriftDetector initialized: window={window_size}, alpha={ewma_alpha}")
    
    def reset(self):
        """Reset detector state."""
        # Buffer must be large enough for both trend calculation AND baseline
        buffer_size = max(self.trend_window, self.baseline_frames)
        self.score_buffer = deque(maxlen=buffer_size)
        self.ewma_value = None
        self.frame_count = 0
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        self.baseline_established = False
        self.baseline_scores = []  # Store baseline scores separately (frozen)
        self.history = []
    
    def _update_ewma(self, score: float) -> float:
        """
        Update Exponentially Weighted Moving Average.
        
        EWMA EXPLANATION:
        -----------------
        EWMA gives more weight to recent observations while
        maintaining memory of past values.
        
        Formula: EWMA_t = α * score_t + (1-α) * EWMA_{t-1}
        
        Where α (alpha) controls the trade-off:
        - α close to 1: Responsive but noisy
        - α close to 0: Smooth but slow to react
        
        We use α = 0.1 by default for slow, stable tracking.
        """
        if self.ewma_value is None:
            self.ewma_value = score
        else:
            self.ewma_value = self.ewma_alpha * score + (1 - self.ewma_alpha) * self.ewma_value
        
        return self.ewma_value
    
    def _compute_trend_slope(self) -> float:
        """
        Compute trend slope using linear regression.
        
        TREND SLOPE EXPLANATION:
        ------------------------
        We fit a line to recent scores and measure its slope.
        
        Positive slope = scores are increasing = drift toward abnormal
        Negative slope = scores are decreasing = returning to normal
        Zero slope = stable behavior
        
        We use normalized time to make slope comparable across scales.
        """
        if len(self.score_buffer) < 10:
            return 0.0
        
        scores = np.array(list(self.score_buffer))
        n = len(scores)
        
        # Time indices (normalized)
        t = np.arange(n) / n
        
        # Linear regression: slope = cov(t, scores) / var(t)
        t_mean = t.mean()
        s_mean = scores.mean()
        
        numerator = np.sum((t - t_mean) * (scores - s_mean))
        denominator = np.sum((t - t_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        return float(slope)
    
    def _establish_baseline(self):
        """
        Establish baseline statistics from initial frames.
        
        BASELINE EXPLANATION:
        ---------------------
        The first N frames are used to establish what "normal"
        looks like in terms of score distribution.
        
        This baseline is used to:
        1. Normalize drift scores
        2. Set initial thresholds
        3. Provide context for deviation
        """
        if len(self.score_buffer) >= self.baseline_frames:
            scores = np.array(list(self.score_buffer))[:self.baseline_frames]
            self.baseline_mean = float(np.mean(scores))
            self.baseline_std = float(np.std(scores))
            if self.baseline_std < 0.01:
                self.baseline_std = 0.01  # Prevent division by zero
            self.baseline_established = True
            logger.info(f"Baseline established: mean={self.baseline_mean:.4f}, std={self.baseline_std:.4f}")
    
    def _compute_drift_score(
        self,
        smoothed_score: float,
        trend_slope: float,
        window_scores: np.ndarray
    ) -> float:
        """
        Compute combined drift score.
        
        DRIFT SCORE EXPLANATION:
        ------------------------
        The drift score combines multiple signals:
        
        1. MAGNITUDE: How far is current behavior from baseline?
           - Normalized by baseline statistics
           - Captures "how abnormal"
        
        2. TREND: Is the deviation getting worse?
           - Positive trend = concerning
           - Captures "direction of change"
        
        3. VARIANCE: Is behavior becoming unstable?
           - High variance = inconsistent behavior
           - Captures "uncertainty"
        
        Final score is weighted combination, designed to:
        - Be close to 0 for normal, stable behavior
        - Increase gradually as drift develops
        - Spike only when multiple signals agree
        
        CRITICAL: Baseline is FROZEN after establishment.
        It does NOT adapt to new data (which might contain threats).
        """
        # CRITICAL: During baseline period, always return 0 (normal)
        # This prevents false positives during learning phase
        if not self.baseline_established:
            return 0.0
        
        # Magnitude component (normalized deviation from FROZEN baseline)
        magnitude = (smoothed_score - self.baseline_mean) / self.baseline_std
        
        # Trend component (scaled to similar range)
        trend = trend_slope * 10  # Scale factor for trend
        
        # Variance component
        if len(window_scores) > 1:
            variance = np.std(window_scores) / (self.baseline_std + 0.01)
        else:
            variance = 0.0
        
        # Combine with weights
        drift_score = (
            self.magnitude_weight * max(0, magnitude) +  # Only positive deviation
            self.trend_weight * max(0, trend) +          # Only increasing trend
            self.variance_weight * variance
        )
        
        return float(drift_score)
    
    def update(
        self,
        normality_score: float,
        frame_index: int = 0,
        timestamp: float = 0.0
    ) -> DriftState:
        """
        Process a new normality score and return drift state.
        
        This is the main entry point for drift detection.
        
        Args:
            normality_score: Raw score from autoencoder
            frame_index: Current frame number
            timestamp: Current timestamp
            
        Returns:
            DriftState with all computed metrics
        """
        self.frame_count += 1
        
        # Add to buffer
        self.score_buffer.append(normality_score)
        
        # Establish baseline if not done
        if not self.baseline_established and self.frame_count >= self.baseline_frames:
            self._establish_baseline()
        
        # Update EWMA
        smoothed_score = self._update_ewma(normality_score)
        
        # Compute trend
        trend_slope = self._compute_trend_slope()
        
        # Get trend direction
        if trend_slope > 0.01:
            trend_direction = "increasing"
        elif trend_slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Window statistics
        window_scores = np.array(list(self.score_buffer)[-self.window_size:])
        window_mean = float(np.mean(window_scores))
        window_std = float(np.std(window_scores))
        window_max = float(np.max(window_scores))
        
        # Compute drift score
        drift_score = self._compute_drift_score(
            smoothed_score, trend_slope, window_scores
        )
        
        # Create state
        state = DriftState(
            raw_score=normality_score,
            smoothed_score=smoothed_score,
            trend_slope=trend_slope,
            trend_direction=trend_direction,
            window_mean=window_mean,
            window_std=window_std,
            window_max=window_max,
            drift_score=drift_score,
            frame_index=frame_index,
            timestamp=timestamp,
            score_history=list(self.score_buffer)
        )
        
        self.history.append(state)
        
        return state
    
    def process_batch(
        self,
        normality_scores: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[DriftState]:
        """
        Process a batch of normality scores.
        
        Useful for offline analysis of recorded video.
        """
        if timestamps is None:
            timestamps = np.arange(len(normality_scores)) / 30.0
        
        states = []
        for i, (score, ts) in enumerate(zip(normality_scores, timestamps)):
            state = self.update(score, frame_index=i, timestamp=ts)
            states.append(state)
        
        return states
    
    def get_drift_summary(self) -> Dict:
        """
        Get summary statistics of detected drift.
        
        Useful for reports and analysis.
        """
        if not self.history:
            return {}
        
        drift_scores = [s.drift_score for s in self.history]
        smoothed_scores = [s.smoothed_score for s in self.history]
        
        return {
            'total_frames': len(self.history),
            'drift_score_mean': float(np.mean(drift_scores)),
            'drift_score_max': float(np.max(drift_scores)),
            'drift_score_final': drift_scores[-1] if drift_scores else 0,
            'smoothed_score_trend': self.history[-1].trend_direction if self.history else "unknown",
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
        }


class AdaptiveDriftDetector(DriftDetector):
    """
    Drift detector with adaptive thresholds.
    
    ADAPTATION EXPLANATION:
    -----------------------
    Environments change over time (lighting, weather, schedules).
    Adaptive detection slowly updates its baseline to account for
    normal environmental changes, while still detecting abnormal drift.
    
    Key: Adaptation rate is MUCH slower than drift detection rate.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.001,
        min_adaptation_interval: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.min_adaptation_interval = min_adaptation_interval
        self.last_adaptation = 0
    
    def _maybe_adapt_baseline(self, current_score: float):
        """
        Slowly adapt baseline if conditions are stable.
        
        Only adapts if:
        1. Enough frames since last adaptation
        2. Current behavior is relatively normal
        3. No active drift detected
        """
        if self.frame_count - self.last_adaptation < self.min_adaptation_interval:
            return
        
        # Only adapt if score is within 1.5 std of baseline
        if self.baseline_established:
            deviation = abs(current_score - self.baseline_mean) / self.baseline_std
            if deviation < 1.5:
                # Slow adaptation
                self.baseline_mean = (
                    (1 - self.adaptation_rate) * self.baseline_mean +
                    self.adaptation_rate * current_score
                )
                self.last_adaptation = self.frame_count
    
    def update(self, normality_score: float, **kwargs) -> DriftState:
        state = super().update(normality_score, **kwargs)
        self._maybe_adapt_baseline(normality_score)
        return state


class MultiScaleDriftDetector:
    """
    Detects drift at multiple time scales.
    
    WHY MULTI-SCALE?
    ----------------
    Different threats manifest at different time scales:
    - Short-term (seconds): Sudden behavioral change
    - Medium-term (minutes): Gradual shift
    - Long-term (hours): Slow degradation
    
    Multi-scale detection catches threats at all scales.
    """
    
    def __init__(
        self,
        scales: List[int] = [30, 100, 300],
        **kwargs
    ):
        self.scales = scales
        self.detectors = {
            scale: DriftDetector(window_size=scale, trend_window=scale * 2, **kwargs)
            for scale in scales
        }
    
    def reset(self):
        for detector in self.detectors.values():
            detector.reset()
    
    def update(self, normality_score: float, **kwargs) -> Dict[int, DriftState]:
        """Update all scale detectors and return states."""
        return {
            scale: detector.update(normality_score, **kwargs)
            for scale, detector in self.detectors.items()
        }
    
    def get_combined_drift_score(self, states: Dict[int, DriftState]) -> float:
        """
        Combine drift scores from all scales.
        
        Uses maximum across scales to catch drift at any timescale.
        """
        return max(state.drift_score for state in states.values())


if __name__ == "__main__":
    # Test drift detection
    print("Testing Drift Detection Module")
    print("=" * 50)
    
    # Create synthetic normality scores
    np.random.seed(42)
    
    # Normal period
    normal_scores = np.random.normal(0, 0.3, 200)
    
    # Gradual drift period
    drift_scores = np.linspace(0, 2, 150) + np.random.normal(0, 0.3, 150)
    
    # Combine
    all_scores = np.concatenate([normal_scores, drift_scores])
    
    print(f"Total scores: {len(all_scores)}")
    print(f"Normal period: frames 0-199")
    print(f"Drift period: frames 200-349")
    
    # Process with drift detector
    detector = DriftDetector(baseline_frames=100)
    states = detector.process_batch(all_scores)
    
    # Show results
    print(f"\nDrift Detection Results:")
    print(f"Baseline mean: {detector.baseline_mean:.4f}")
    print(f"Baseline std: {detector.baseline_std:.4f}")
    
    # Show drift scores at key points
    key_frames = [50, 150, 200, 250, 300, 349]
    for f in key_frames:
        s = states[f]
        print(f"\nFrame {f}:")
        print(f"  Raw score: {s.raw_score:.4f}")
        print(f"  Drift score: {s.drift_score:.4f}")
        print(f"  Trend: {s.trend_direction} (slope: {s.trend_slope:.4f})")
    
    print("\nDrift detection module ready!")
