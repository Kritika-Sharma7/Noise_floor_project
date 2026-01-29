"""
NOISE FLOOR - Drift Intelligence Layer
========================================
Advanced drift scoring and intelligence metrics for border surveillance.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

CRITICAL DESIGN PRINCIPLE:
--------------------------
Do NOT use raw reconstruction error directly for alerts!

This layer converts ML model outputs into DECISION INTELLIGENCE:
1. KL Divergence from baseline latent distribution
2. EWMA-smoothed deviation scores
3. Rolling Z-scores for statistical significance
4. Trend persistence scores (duration of deviation)

KEY PHILOSOPHY:
"Defense systems manage CONFIDENCE, not panic."

DRIFT INTELLIGENCE METRICS:
---------------------------
- Threat Deviation Index (TDI): 0-100 scale for operators
- Drift Trend: Increasing / Stable / Decreasing
- Trend Persistence: How long has deviation been sustained
- Statistical Confidence: How significant is the deviation
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftTrend(Enum):
    """Drift trend direction."""
    INCREASING = "increasing"       # ↑ Drift is growing
    STABLE = "stable"               # → Drift is steady
    DECREASING = "decreasing"       # ↓ Drift is reducing
    
    @property
    def symbol(self) -> str:
        """Arrow symbol for display."""
        symbols = {
            DriftTrend.INCREASING: "↑",
            DriftTrend.STABLE: "→",
            DriftTrend.DECREASING: "↓"
        }
        return symbols[self]


@dataclass
class DriftIntelligence:
    """
    Complete drift intelligence state for a single time point.
    
    This is the OUTPUT of the Drift Intelligence Layer,
    ready for consumption by the Risk Zone classifier and dashboard.
    """
    # ===== PRIMARY METRICS (for dashboard) =====
    
    # Threat Deviation Index (0-100)
    # This is the MAIN metric shown to operators
    threat_deviation_index: float = 0.0
    
    # Current risk zone (will be set by classifier)
    risk_zone: str = "normal"
    
    # Drift trend direction
    drift_trend: DriftTrend = DriftTrend.STABLE
    
    # When drift was first detected (if any)
    drift_onset_frame: Optional[int] = None
    drift_onset_timestamp: Optional[float] = None
    
    # Top contributing features
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    
    # Overall confidence in assessment
    confidence: float = 1.0
    
    # ===== DETAILED METRICS (for analysis) =====
    
    # Raw model scores
    raw_reconstruction_loss: float = 0.0
    raw_kl_divergence: float = 0.0
    
    # Statistical scores
    z_score: float = 0.0                # Standard deviations from baseline
    ewma_score: float = 0.0             # Smoothed deviation score
    rolling_z_score: float = 0.0        # Z-score over recent window
    
    # Trend metrics
    trend_slope: float = 0.0            # Rate of change
    trend_persistence: int = 0          # Frames of sustained deviation
    trend_confidence: float = 0.0       # Confidence in trend direction
    
    # Latent space metrics
    latent_kl_divergence: float = 0.0   # KL from baseline latent distribution
    latent_mahalanobis: float = 0.0     # Mahalanobis distance in latent space
    
    # Temporal info
    frame_index: int = 0
    timestamp: float = 0.0
    window_index: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'threat_deviation_index': self.threat_deviation_index,
            'risk_zone': self.risk_zone,
            'drift_trend': self.drift_trend.value,
            'drift_trend_symbol': self.drift_trend.symbol,
            'drift_onset_frame': self.drift_onset_frame,
            'confidence': self.confidence,
            'top_features': self.top_features,
            'z_score': self.z_score,
            'ewma_score': self.ewma_score,
            'trend_slope': self.trend_slope,
            'trend_persistence': self.trend_persistence,
            'frame_index': self.frame_index,
            'timestamp': self.timestamp,
        }


@dataclass
class BaselineStatistics:
    """Baseline statistics for drift detection."""
    # Raw score statistics
    score_mean: float = 0.0
    score_std: float = 1.0
    score_percentiles: Dict[int, float] = field(default_factory=dict)
    
    # Latent space baseline
    latent_mean: np.ndarray = field(default_factory=lambda: np.zeros(16))
    latent_cov: np.ndarray = field(default_factory=lambda: np.eye(16))
    latent_cov_inv: np.ndarray = field(default_factory=lambda: np.eye(16))
    
    # Feature-level baselines
    feature_means: np.ndarray = field(default_factory=lambda: np.zeros(24))
    feature_stds: np.ndarray = field(default_factory=lambda: np.ones(24))
    
    # Training info
    num_samples: int = 0
    is_established: bool = False


class DriftIntelligenceEngine:
    """
    Core engine for converting ML outputs into drift intelligence.
    
    PROCESSING PIPELINE:
    -------------------
    1. Receive raw model outputs (reconstruction loss, latent vectors)
    2. Compute KL divergence from baseline latent distribution
    3. Apply EWMA smoothing to reduce noise
    4. Calculate rolling Z-scores for statistical significance
    5. Track trend direction and persistence
    6. Compute Threat Deviation Index (0-100)
    7. Identify top contributing features
    
    OUTPUT:
    -------
    DriftIntelligence object with all metrics for decision-making.
    """
    
    def __init__(
        self,
        # Window parameters
        window_size: int = 30,              # Frames for aggregation
        trend_window: int = 50,             # Frames for trend calculation
        baseline_frames: int = 200,         # Frames for baseline establishment
        
        # EWMA parameters
        ewma_alpha: float = 0.1,            # Smoothing factor (lower = smoother)
        
        # Scoring weights
        reconstruction_weight: float = 0.4,
        kl_weight: float = 0.3,
        trend_weight: float = 0.2,
        persistence_weight: float = 0.1,
        
        # Thresholds
        deviation_threshold: float = 2.0,   # Z-score for "deviation"
        trend_threshold: float = 0.1,       # Slope for trend detection
        
        # Feature names for attribution
        feature_names: Optional[List[str]] = None
    ):
        self.window_size = window_size
        self.trend_window = trend_window
        self.baseline_frames = baseline_frames
        self.ewma_alpha = ewma_alpha
        
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.trend_weight = trend_weight
        self.persistence_weight = persistence_weight
        
        self.deviation_threshold = deviation_threshold
        self.trend_threshold = trend_threshold
        
        self.feature_names = feature_names or []
        
        # State
        self.reset()
        
        logger.info(f"DriftIntelligenceEngine initialized:")
        logger.info(f"  Window: {window_size}, Trend: {trend_window}")
        logger.info(f"  EWMA alpha: {ewma_alpha}")
    
    def reset(self):
        """Reset engine state."""
        # Score buffers
        buffer_size = max(self.trend_window, self.baseline_frames, 500)
        self.score_buffer = deque(maxlen=buffer_size)
        self.reconstruction_buffer = deque(maxlen=buffer_size)
        self.kl_buffer = deque(maxlen=buffer_size)
        self.latent_buffer = deque(maxlen=buffer_size)
        self.feature_buffer = deque(maxlen=buffer_size)
        
        # EWMA state
        self.ewma_value = None
        
        # Baseline
        self.baseline = BaselineStatistics()
        
        # Tracking
        self.frame_count = 0
        self.drift_onset = None
        self.current_trend = DriftTrend.STABLE
        self.trend_persistence = 0
        
        # History
        self.history: List[DriftIntelligence] = []
    
    def set_baseline(
        self,
        reconstruction_losses: np.ndarray,
        kl_divergences: np.ndarray,
        latent_means: np.ndarray,
        feature_values: Optional[np.ndarray] = None
    ):
        """
        Establish baseline from training/normal data.
        
        Call this after training the model on normal data.
        """
        # Combined score
        scores = (
            self.reconstruction_weight * reconstruction_losses +
            self.kl_weight * kl_divergences
        )
        
        # Score statistics
        self.baseline.score_mean = float(np.mean(scores))
        self.baseline.score_std = float(np.std(scores)) + 1e-6
        self.baseline.score_percentiles = {
            50: float(np.percentile(scores, 50)),
            75: float(np.percentile(scores, 75)),
            90: float(np.percentile(scores, 90)),
            95: float(np.percentile(scores, 95)),
            99: float(np.percentile(scores, 99)),
        }
        
        # Latent space statistics
        self.baseline.latent_mean = np.mean(latent_means, axis=0)
        self.baseline.latent_cov = np.cov(latent_means.T) + np.eye(latent_means.shape[1]) * 1e-6
        try:
            self.baseline.latent_cov_inv = np.linalg.inv(self.baseline.latent_cov)
        except np.linalg.LinAlgError:
            self.baseline.latent_cov_inv = np.eye(latent_means.shape[1])
        
        # Feature statistics
        if feature_values is not None:
            self.baseline.feature_means = np.mean(feature_values, axis=0)
            self.baseline.feature_stds = np.std(feature_values, axis=0) + 1e-6
        
        self.baseline.num_samples = len(scores)
        self.baseline.is_established = True
        
        # Initialize EWMA with baseline mean
        self.ewma_value = self.baseline.score_mean
        
        logger.info(f"Baseline established from {len(scores)} samples")
        logger.info(f"  Score mean: {self.baseline.score_mean:.4f}, std: {self.baseline.score_std:.4f}")
    
    def compute_kl_divergence(
        self,
        current_mean: np.ndarray,
        current_logvar: np.ndarray
    ) -> float:
        """
        Compute KL divergence between current latent distribution and baseline.
        
        This measures how "different" the current behavioral pattern is
        from the learned normal distribution.
        
        KL(q||p) where:
        - q = current distribution N(current_mean, current_var)
        - p = baseline distribution N(baseline_mean, baseline_cov)
        """
        if not self.baseline.is_established:
            return 0.0
        
        current_var = np.exp(current_logvar)
        
        # Simplified KL for diagonal covariances
        baseline_var = np.diag(self.baseline.latent_cov)
        
        kl = 0.5 * np.sum(
            np.log(baseline_var / (current_var + 1e-10)) +
            (current_var + (current_mean - self.baseline.latent_mean)**2) / baseline_var -
            1
        )
        
        return float(np.clip(kl, 0, 100))
    
    def compute_mahalanobis_distance(
        self,
        latent_mean: np.ndarray
    ) -> float:
        """
        Compute Mahalanobis distance in latent space.
        
        This measures how many "standard deviations" the current point
        is from the baseline center, accounting for correlations.
        """
        if not self.baseline.is_established:
            return 0.0
        
        diff = latent_mean - self.baseline.latent_mean
        
        try:
            dist = np.sqrt(diff @ self.baseline.latent_cov_inv @ diff)
        except:
            dist = np.linalg.norm(diff)
        
        return float(dist)
    
    def update_ewma(self, score: float) -> float:
        """
        Update Exponentially Weighted Moving Average.
        
        EWMA provides noise-resistant smoothing:
        - High alpha = more responsive, more noise
        - Low alpha = smoother, slower response
        """
        if self.ewma_value is None:
            self.ewma_value = score
        else:
            self.ewma_value = self.ewma_alpha * score + (1 - self.ewma_alpha) * self.ewma_value
        
        return self.ewma_value
    
    def compute_rolling_z_score(self) -> float:
        """
        Compute rolling Z-score over recent window.
        
        This provides statistical significance testing:
        - |Z| > 2: Significant deviation
        - |Z| > 3: Highly significant deviation
        """
        if len(self.score_buffer) < self.window_size:
            return 0.0
        
        recent_scores = list(self.score_buffer)[-self.window_size:]
        window_mean = np.mean(recent_scores)
        
        if not self.baseline.is_established:
            return 0.0
        
        z_score = (window_mean - self.baseline.score_mean) / self.baseline.score_std
        
        return float(z_score)
    
    def compute_trend(self) -> Tuple[float, DriftTrend, float]:
        """
        Compute trend direction and slope.
        
        Uses linear regression on recent scores to determine
        if deviation is increasing, stable, or decreasing.
        
        Returns:
            slope: Rate of change
            trend: Trend direction
            confidence: R² of the fit
        """
        if len(self.score_buffer) < self.trend_window // 2:
            return 0.0, DriftTrend.STABLE, 0.0
        
        n = min(len(self.score_buffer), self.trend_window)
        recent_scores = list(self.score_buffer)[-n:]
        
        # Linear regression
        x = np.arange(n)
        y = np.array(recent_scores)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if abs(slope) < self.trend_threshold:
            trend = DriftTrend.STABLE
        elif slope > 0:
            trend = DriftTrend.INCREASING
        else:
            trend = DriftTrend.DECREASING
        
        confidence = r_value ** 2  # R² value
        
        return float(slope), trend, float(confidence)
    
    def compute_feature_attribution(
        self,
        current_features: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Identify top contributing features to drift.
        
        Compares current features to baseline and ranks by deviation.
        
        Returns:
            List of (feature_name, z_score) tuples, sorted by |z_score|
        """
        if not self.baseline.is_established or len(self.feature_names) == 0:
            return []
        
        if len(current_features) != len(self.baseline.feature_means):
            return []
        
        # Compute Z-scores for each feature
        z_scores = (current_features - self.baseline.feature_means) / self.baseline.feature_stds
        
        # Create attribution list
        attributions = []
        for i, (name, z) in enumerate(zip(self.feature_names, z_scores)):
            if abs(z) > 1.0:  # Only include significant deviations
                attributions.append((name, float(z)))
        
        # Sort by absolute Z-score
        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return attributions[:5]  # Return top 5
    
    def compute_threat_deviation_index(
        self,
        z_score: float,
        ewma_score: float,
        trend_slope: float,
        persistence: int
    ) -> float:
        """
        Compute Threat Deviation Index (0-100 scale).
        
        This is the PRIMARY metric shown to operators.
        
        Components:
        - Base deviation (Z-score based)
        - Trend acceleration (slope based)
        - Persistence bonus (sustained deviation)
        
        The index is designed to:
        - Stay low during normal operations
        - Rise gradually with drift
        - Peak during sustained abnormal patterns
        """
        # Base score from Z-score (0-100 mapping)
        # Z=0 -> 0, Z=3 -> 50, Z=6 -> 80
        base_score = min(100, max(0, 50 * (1 - np.exp(-z_score / 3))))
        
        # Trend component (adds up to 20 points if increasing rapidly)
        trend_score = min(20, max(0, trend_slope * 100))
        
        # Persistence bonus (adds up to 30 points for sustained deviation)
        persistence_score = min(30, persistence * 0.5)
        
        # Combine with weights
        tdi = (
            self.reconstruction_weight * base_score +
            self.trend_weight * trend_score +
            self.persistence_weight * persistence_score
        )
        
        # Apply sigmoid-like smoothing to avoid jumps
        tdi = 100 * (1 - np.exp(-tdi / 50))
        
        return float(np.clip(tdi, 0, 100))
    
    def process(
        self,
        reconstruction_loss: float,
        kl_divergence: float,
        latent_mean: np.ndarray,
        latent_logvar: np.ndarray,
        features: Optional[np.ndarray] = None,
        frame_index: int = 0,
        timestamp: float = 0.0,
        window_index: int = 0
    ) -> DriftIntelligence:
        """
        Process a single observation and produce drift intelligence.
        
        This is the MAIN METHOD called for each temporal window.
        
        Args:
            reconstruction_loss: Raw reconstruction loss from model
            kl_divergence: KL divergence from model
            latent_mean: Latent space mean vector
            latent_logvar: Latent space log variance
            features: Current feature values (for attribution)
            frame_index: Current frame index
            timestamp: Current timestamp
            window_index: Current window index
            
        Returns:
            DriftIntelligence with all computed metrics
        """
        self.frame_count += 1
        
        # ===== STEP 1: Compute raw combined score =====
        raw_score = (
            self.reconstruction_weight * reconstruction_loss +
            self.kl_weight * kl_divergence
        )
        
        # ===== STEP 2: Update buffers =====
        self.score_buffer.append(raw_score)
        self.reconstruction_buffer.append(reconstruction_loss)
        self.kl_buffer.append(kl_divergence)
        self.latent_buffer.append(latent_mean)
        if features is not None:
            self.feature_buffer.append(features)
        
        # ===== STEP 3: Auto-establish baseline if not set =====
        if not self.baseline.is_established and len(self.score_buffer) >= self.baseline_frames:
            self._auto_establish_baseline()
        
        # ===== STEP 4: Compute intelligence metrics =====
        
        # EWMA smoothing
        ewma_score = self.update_ewma(raw_score)
        
        # Z-score from baseline
        if self.baseline.is_established:
            z_score = (raw_score - self.baseline.score_mean) / self.baseline.score_std
        else:
            z_score = 0.0
        
        # Rolling Z-score
        rolling_z = self.compute_rolling_z_score()
        
        # Trend analysis
        trend_slope, trend_direction, trend_confidence = self.compute_trend()
        
        # Latent space metrics
        latent_kl = self.compute_kl_divergence(latent_mean, latent_logvar)
        latent_mahal = self.compute_mahalanobis_distance(latent_mean)
        
        # ===== STEP 5: Track drift state =====
        
        # Check if currently in deviation
        is_deviating = z_score > self.deviation_threshold
        
        # Update persistence
        if is_deviating:
            self.trend_persistence += 1
            if self.drift_onset is None:
                self.drift_onset = (frame_index, timestamp)
        else:
            self.trend_persistence = max(0, self.trend_persistence - 1)
            if self.trend_persistence == 0:
                self.drift_onset = None
        
        # Update trend
        self.current_trend = trend_direction
        
        # ===== STEP 6: Compute TDI =====
        tdi = self.compute_threat_deviation_index(
            z_score, ewma_score, trend_slope, self.trend_persistence
        )
        
        # ===== STEP 7: Feature attribution =====
        if features is not None and len(self.feature_names) > 0:
            top_features = self.compute_feature_attribution(features)
        else:
            top_features = []
        
        # ===== STEP 8: Compute confidence =====
        # Confidence is lower when:
        # - Baseline not fully established
        # - High variance in recent scores
        if not self.baseline.is_established:
            confidence = len(self.score_buffer) / self.baseline_frames
        else:
            recent_std = np.std(list(self.score_buffer)[-self.window_size:]) if len(self.score_buffer) >= self.window_size else 1.0
            confidence = 1.0 / (1.0 + recent_std / self.baseline.score_std)
        
        # ===== STEP 9: Create output =====
        intelligence = DriftIntelligence(
            threat_deviation_index=tdi,
            drift_trend=trend_direction,
            drift_onset_frame=self.drift_onset[0] if self.drift_onset else None,
            drift_onset_timestamp=self.drift_onset[1] if self.drift_onset else None,
            top_features=top_features,
            confidence=confidence,
            raw_reconstruction_loss=reconstruction_loss,
            raw_kl_divergence=kl_divergence,
            z_score=z_score,
            ewma_score=ewma_score,
            rolling_z_score=rolling_z,
            trend_slope=trend_slope,
            trend_persistence=self.trend_persistence,
            trend_confidence=trend_confidence,
            latent_kl_divergence=latent_kl,
            latent_mahalanobis=latent_mahal,
            frame_index=frame_index,
            timestamp=timestamp,
            window_index=window_index,
        )
        
        # Store in history
        self.history.append(intelligence)
        
        return intelligence
    
    def _auto_establish_baseline(self):
        """Automatically establish baseline from buffered data."""
        recon_losses = np.array(list(self.reconstruction_buffer)[:self.baseline_frames])
        kl_divs = np.array(list(self.kl_buffer)[:self.baseline_frames])
        latent_means = np.array(list(self.latent_buffer)[:self.baseline_frames])
        
        feature_values = None
        if len(self.feature_buffer) >= self.baseline_frames:
            feature_values = np.array(list(self.feature_buffer)[:self.baseline_frames])
        
        self.set_baseline(recon_losses, kl_divs, latent_means, feature_values)
        
        logger.info(f"Baseline auto-established at frame {self.frame_count}")
    
    def process_batch(
        self,
        reconstruction_losses: np.ndarray,
        kl_divergences: np.ndarray,
        latent_means: np.ndarray,
        latent_logvars: np.ndarray,
        features: Optional[np.ndarray] = None,
        start_frame: int = 0
    ) -> List[DriftIntelligence]:
        """
        Process a batch of observations.
        
        Useful for offline analysis of recorded data.
        """
        results = []
        
        for i in range(len(reconstruction_losses)):
            feat = features[i] if features is not None else None
            
            intel = self.process(
                reconstruction_loss=reconstruction_losses[i],
                kl_divergence=kl_divergences[i],
                latent_mean=latent_means[i],
                latent_logvar=latent_logvars[i],
                features=feat,
                frame_index=start_frame + i,
                timestamp=i / 10.0,  # Assume 10 fps
                window_index=i
            )
            
            results.append(intel)
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary statistics from processing history."""
        if not self.history:
            return {}
        
        tdis = [h.threat_deviation_index for h in self.history]
        z_scores = [h.z_score for h in self.history]
        
        return {
            'total_frames': len(self.history),
            'baseline_established': self.baseline.is_established,
            'mean_tdi': float(np.mean(tdis)),
            'max_tdi': float(np.max(tdis)),
            'mean_z_score': float(np.mean(z_scores)),
            'max_z_score': float(np.max(z_scores)),
            'drift_detected': self.drift_onset is not None,
            'drift_onset_frame': self.drift_onset[0] if self.drift_onset else None,
            'current_trend': self.current_trend.value,
            'trend_persistence': self.trend_persistence,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_drift_engine(
    feature_names: Optional[List[str]] = None,
    baseline_frames: int = 100,
    ewma_alpha: float = 0.1
) -> DriftIntelligenceEngine:
    """
    Create a drift intelligence engine with sensible defaults.
    """
    return DriftIntelligenceEngine(
        window_size=30,
        trend_window=50,
        baseline_frames=baseline_frames,
        ewma_alpha=ewma_alpha,
        feature_names=feature_names or [],
    )
