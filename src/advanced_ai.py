"""
NOISE FLOOR - Advanced AI Intelligence Module
===============================================
Advanced AI/ML capabilities for defense-grade anomaly detection.

This module provides:
1. Confidence Calibration - Calibrated uncertainty estimation
2. Adaptive Thresholding - Context-aware dynamic thresholds
3. Anomaly Classification - Categorize detected anomalies
4. Prediction Module - Forecast future TDI values

DESIGN PHILOSOPHY:
------------------
"Intelligence is not just detection, it's understanding and prediction."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ANOMALY CLASSIFICATION
# =============================================================================

class AnomalyCategory(Enum):
    """Categories of detected anomalies for border surveillance."""
    NORMAL = "normal"
    LOITERING = "loitering"                    # Prolonged presence in area
    INTRUSION = "intrusion"                    # Boundary crossing
    CROWD_FORMATION = "crowd_formation"        # Unusual gathering
    ERRATIC_MOVEMENT = "erratic_movement"      # Abnormal motion patterns
    COORDINATED_ACTIVITY = "coordinated"       # Synchronized movement
    SPEED_ANOMALY = "speed_anomaly"            # Unusual velocity
    DIRECTION_ANOMALY = "direction_anomaly"    # Unusual direction pattern
    UNKNOWN = "unknown"                        # Unclassified anomaly


@dataclass
class AnomalyClassification:
    """Classification result for a detected anomaly."""
    primary_category: AnomalyCategory
    confidence: float                          # 0-1 confidence in classification
    secondary_categories: List[Tuple[AnomalyCategory, float]] = field(default_factory=list)
    reasoning: str = ""
    suggested_response: str = ""
    severity: int = 1                          # 1-5 severity scale
    
    def __repr__(self):
        return f"{self.primary_category.value} (conf={self.confidence:.2f}, severity={self.severity})"


class AnomalyClassifier:
    """
    Classifies detected anomalies into actionable categories.
    
    Uses feature attribution to determine the type of anomaly
    based on which behavioral features are most deviant.
    """
    
    # Feature patterns for each anomaly type
    CATEGORY_PATTERNS = {
        AnomalyCategory.LOITERING: {
            'primary_features': ['idle_ratio', 'velocity_mean', 'movement_complexity'],
            'pattern': 'high_idle',  # High idle ratio, low velocity
            'severity_base': 2,
        },
        AnomalyCategory.INTRUSION: {
            'primary_features': ['direction_consistency', 'velocity_mean', 'motion_energy'],
            'pattern': 'directed_motion',  # Consistent direction, high energy
            'severity_base': 4,
        },
        AnomalyCategory.CROWD_FORMATION: {
            'primary_features': ['spatial_coherence', 'activity_transitions', 'scene_entropy'],
            'pattern': 'clustering',  # High scene entropy, low coherence
            'severity_base': 3,
        },
        AnomalyCategory.ERRATIC_MOVEMENT: {
            'primary_features': ['direction_entropy', 'direction_change_rate', 'velocity_std'],
            'pattern': 'random_motion',  # High entropy, high variance
            'severity_base': 2,
        },
        AnomalyCategory.COORDINATED_ACTIVITY: {
            'primary_features': ['direction_consistency', 'temporal_stability', 'spatial_coherence'],
            'pattern': 'synchronized',  # High consistency, high stability
            'severity_base': 4,
        },
        AnomalyCategory.SPEED_ANOMALY: {
            'primary_features': ['velocity_mean', 'velocity_std', 'acceleration_mean'],
            'pattern': 'speed_deviation',
            'severity_base': 3,
        },
        AnomalyCategory.DIRECTION_ANOMALY: {
            'primary_features': ['dominant_direction', 'direction_consistency', 'direction_entropy'],
            'pattern': 'direction_deviation',
            'severity_base': 2,
        },
    }
    
    # Response suggestions
    RESPONSE_SUGGESTIONS = {
        AnomalyCategory.NORMAL: "Continue routine monitoring",
        AnomalyCategory.LOITERING: "Dispatch patrol to investigate stationary activity",
        AnomalyCategory.INTRUSION: "ALERT: Potential perimeter breach - immediate response required",
        AnomalyCategory.CROWD_FORMATION: "Monitor for crowd dispersal, prepare crowd control if needed",
        AnomalyCategory.ERRATIC_MOVEMENT: "Track subject, possible distress or evasion",
        AnomalyCategory.COORDINATED_ACTIVITY: "HIGH ALERT: Possible coordinated operation - notify command",
        AnomalyCategory.SPEED_ANOMALY: "Track high-speed movement, possible pursuit scenario",
        AnomalyCategory.DIRECTION_ANOMALY: "Monitor unusual movement pattern",
        AnomalyCategory.UNKNOWN: "Unclassified anomaly - manual review recommended",
    }
    
    def __init__(self):
        self.classification_history = deque(maxlen=100)
    
    def classify(
        self,
        features: np.ndarray,
        feature_names: List[str],
        z_scores: np.ndarray,
        tdi: float,
    ) -> AnomalyClassification:
        """
        Classify an anomaly based on feature patterns.
        
        Args:
            features: Current feature vector
            feature_names: Names of features
            z_scores: Z-scores for each feature
            tdi: Threat Deviation Index
        
        Returns:
            AnomalyClassification with category and details
        """
        if tdi < 20:
            return AnomalyClassification(
                primary_category=AnomalyCategory.NORMAL,
                confidence=1.0,
                reasoning="TDI within normal bounds",
                suggested_response=self.RESPONSE_SUGGESTIONS[AnomalyCategory.NORMAL],
                severity=0,
            )
        
        # Create feature-score mapping
        feature_z_map = dict(zip(feature_names, z_scores))
        
        # Score each category
        category_scores = {}
        
        for category, pattern_info in self.CATEGORY_PATTERNS.items():
            primary_features = pattern_info['primary_features']
            
            # Calculate match score
            relevant_z_scores = []
            for feat in primary_features:
                if feat in feature_z_map:
                    relevant_z_scores.append(abs(feature_z_map[feat]))
            
            if relevant_z_scores:
                # Higher Z-scores in relevant features = better match
                match_score = np.mean(relevant_z_scores) / 5.0  # Normalize to ~0-1
                category_scores[category] = min(match_score, 1.0)
            else:
                category_scores[category] = 0.0
        
        # Sort by score
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_categories or sorted_categories[0][1] < 0.2:
            primary = AnomalyCategory.UNKNOWN
            confidence = 0.3
        else:
            primary = sorted_categories[0][0]
            confidence = sorted_categories[0][1]
        
        # Get secondary categories
        secondary = [(cat, score) for cat, score in sorted_categories[1:4] if score > 0.1]
        
        # Calculate severity (1-5)
        base_severity = self.CATEGORY_PATTERNS.get(primary, {}).get('severity_base', 2)
        tdi_factor = min(tdi / 60, 1.5)  # Scale by TDI
        severity = min(5, max(1, int(base_severity * tdi_factor)))
        
        # Generate reasoning
        top_z_features = sorted(feature_z_map.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        reasoning = f"Classification based on elevated: {', '.join([f[0] for f in top_z_features])}"
        
        classification = AnomalyClassification(
            primary_category=primary,
            confidence=confidence,
            secondary_categories=secondary,
            reasoning=reasoning,
            suggested_response=self.RESPONSE_SUGGESTIONS.get(primary, "Manual review recommended"),
            severity=severity,
        )
        
        self.classification_history.append(classification)
        return classification
    
    def get_category_distribution(self, recent_n: int = 50) -> Dict[str, int]:
        """Get distribution of recent classifications."""
        recent = list(self.classification_history)[-recent_n:]
        distribution = {}
        for cls in recent:
            cat_name = cls.primary_category.value
            distribution[cat_name] = distribution.get(cat_name, 0) + 1
        return distribution


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

@dataclass
class CalibratedConfidence:
    """Calibrated confidence estimation."""
    raw_confidence: float              # Original confidence
    calibrated_confidence: float       # Calibrated confidence
    uncertainty: float                 # Epistemic uncertainty
    aleatoric_uncertainty: float       # Data uncertainty
    reliability: float                 # How reliable is this prediction
    calibration_method: str = "temperature_scaling"


class ConfidenceCalibrator:
    """
    Calibrates model confidence using temperature scaling and
    VAE reconstruction variance.
    
    "Overconfident models are dangerous in defense applications."
    """
    
    def __init__(
        self,
        temperature: float = 1.5,
        window_size: int = 100,
    ):
        self.temperature = temperature
        self.window_size = window_size
        
        # History for calibration
        self.prediction_history = deque(maxlen=window_size)
        self.reconstruction_variances = deque(maxlen=window_size)
        self.latent_variances = deque(maxlen=window_size)
        
        # Calibration statistics
        self.calibration_error = 0.0
        self._calibrated = False
    
    def calibrate(
        self,
        raw_confidence: float,
        reconstruction_variance: float = 0.0,
        latent_variance: float = 0.0,
        ensemble_agreement: float = 1.0,
    ) -> CalibratedConfidence:
        """
        Calibrate confidence using multiple uncertainty sources.
        
        Args:
            raw_confidence: Original model confidence (0-1)
            reconstruction_variance: VAE reconstruction variance
            latent_variance: Variance in latent space
            ensemble_agreement: Agreement between ensemble members (0-1)
        
        Returns:
            CalibratedConfidence with calibrated values
        """
        # Store for history
        self.reconstruction_variances.append(reconstruction_variance)
        self.latent_variances.append(latent_variance)
        
        # Temperature scaling
        # Higher temperature = lower confidence (more conservative)
        scaled_confidence = self._temperature_scale(raw_confidence)
        
        # Epistemic uncertainty (model uncertainty)
        # Based on ensemble disagreement and prediction variance
        epistemic = 1.0 - ensemble_agreement
        
        # Aleatoric uncertainty (data uncertainty)
        # Based on reconstruction and latent variance
        if self.reconstruction_variances:
            mean_recon_var = np.mean(list(self.reconstruction_variances))
            normalized_recon_var = min(reconstruction_variance / (mean_recon_var + 1e-6), 2.0)
        else:
            normalized_recon_var = 0.5
        
        aleatoric = min(normalized_recon_var * 0.5, 1.0)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic ** 2 + aleatoric ** 2)
        
        # Adjust calibrated confidence by uncertainty
        calibrated = scaled_confidence * (1.0 - total_uncertainty * 0.5)
        calibrated = np.clip(calibrated, 0.05, 0.95)  # Never 0 or 1
        
        # Reliability score
        reliability = 1.0 - total_uncertainty
        
        return CalibratedConfidence(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            uncertainty=total_uncertainty,
            aleatoric_uncertainty=aleatoric,
            reliability=reliability,
            calibration_method="temperature_scaling_with_uncertainty",
        )
    
    def _temperature_scale(self, confidence: float) -> float:
        """Apply temperature scaling to confidence."""
        # Convert to logit, scale, convert back
        eps = 1e-7
        confidence = np.clip(confidence, eps, 1 - eps)
        logit = np.log(confidence / (1 - confidence))
        scaled_logit = logit / self.temperature
        scaled_confidence = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(scaled_confidence)
    
    def update_temperature(self, predictions: List[float], actuals: List[bool]) -> None:
        """Update temperature based on calibration error."""
        if len(predictions) < 10:
            return
        
        # Calculate expected calibration error
        predictions = np.array(predictions)
        actuals = np.array(actuals, dtype=float)
        
        # Binned calibration
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        calibration_error = 0.0
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_conf = predictions[mask].mean()
                avg_acc = actuals[mask].mean()
                calibration_error += mask.sum() * abs(avg_conf - avg_acc)
        
        calibration_error /= len(predictions)
        self.calibration_error = calibration_error
        
        # Adjust temperature to reduce calibration error
        if calibration_error > 0.1:
            self.temperature *= 1.1  # Increase temperature (reduce confidence)
        elif calibration_error < 0.05:
            self.temperature *= 0.95  # Decrease temperature


# =============================================================================
# ADAPTIVE THRESHOLDING
# =============================================================================

class TimeOfDay(Enum):
    """Time periods for threshold adaptation."""
    DAWN = "dawn"          # 5-7 AM
    MORNING = "morning"    # 7-12 PM
    AFTERNOON = "afternoon"  # 12-5 PM
    EVENING = "evening"    # 5-8 PM
    NIGHT = "night"        # 8 PM - 5 AM


@dataclass
class AdaptiveThresholds:
    """Adaptive threshold values based on context."""
    watch_threshold: float
    warning_threshold: float
    critical_threshold: float
    
    # Context factors
    time_factor: float = 1.0
    activity_factor: float = 1.0
    weather_factor: float = 1.0
    
    # Effective thresholds (after applying factors)
    effective_watch: float = 0.0
    effective_warning: float = 0.0
    effective_critical: float = 0.0
    
    context_description: str = ""
    
    def __post_init__(self):
        combined_factor = self.time_factor * self.activity_factor * self.weather_factor
        self.effective_watch = self.watch_threshold * combined_factor
        self.effective_warning = self.warning_threshold * combined_factor
        self.effective_critical = self.critical_threshold * combined_factor


class AdaptiveThresholdManager:
    """
    Manages dynamic thresholds based on context.
    
    Context factors:
    - Time of day (higher sensitivity at night)
    - Activity level (baseline adjustment)
    - Weather conditions (if available)
    - Historical patterns
    """
    
    # Time-based sensitivity multipliers
    TIME_FACTORS = {
        TimeOfDay.DAWN: 1.1,      # Slightly higher sensitivity
        TimeOfDay.MORNING: 1.0,   # Normal
        TimeOfDay.AFTERNOON: 0.95,  # Slightly lower (more activity expected)
        TimeOfDay.EVENING: 1.0,   # Normal
        TimeOfDay.NIGHT: 1.2,     # Higher sensitivity
    }
    
    def __init__(
        self,
        base_watch: float = 20.0,
        base_warning: float = 40.0,
        base_critical: float = 60.0,
        enable_time_adaptation: bool = True,
        enable_activity_adaptation: bool = True,
    ):
        self.base_watch = base_watch
        self.base_warning = base_warning
        self.base_critical = base_critical
        
        self.enable_time_adaptation = enable_time_adaptation
        self.enable_activity_adaptation = enable_activity_adaptation
        
        # Activity baseline tracking
        self.activity_history = deque(maxlen=1000)
        self.hourly_baselines = {h: [] for h in range(24)}
        
    def get_time_of_day(self, hour: int = None) -> TimeOfDay:
        """Get current time period."""
        if hour is None:
            hour = datetime.now().hour
        
        if 5 <= hour < 7:
            return TimeOfDay.DAWN
        elif 7 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 20:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT
    
    def get_thresholds(
        self,
        current_activity: float = None,
        weather_condition: str = None,
        custom_factors: Dict[str, float] = None,
    ) -> AdaptiveThresholds:
        """
        Get current adaptive thresholds.
        
        Args:
            current_activity: Current activity level (for baseline comparison)
            weather_condition: Weather condition string (optional)
            custom_factors: Custom adjustment factors
        
        Returns:
            AdaptiveThresholds with context-adjusted values
        """
        # Time factor
        time_of_day = self.get_time_of_day()
        time_factor = self.TIME_FACTORS[time_of_day] if self.enable_time_adaptation else 1.0
        
        # Activity factor
        activity_factor = 1.0
        if self.enable_activity_adaptation and current_activity is not None:
            self.activity_history.append(current_activity)
            if len(self.activity_history) > 50:
                baseline_activity = np.mean(list(self.activity_history))
                if baseline_activity > 0:
                    ratio = current_activity / baseline_activity
                    # If activity is higher than usual, slightly lower thresholds
                    activity_factor = np.clip(1.0 / (ratio * 0.5 + 0.5), 0.8, 1.2)
        
        # Weather factor (placeholder - would integrate with weather API)
        weather_factor = 1.0
        if weather_condition:
            weather_factors = {
                'clear': 1.0,
                'rain': 0.9,      # Expect less activity, higher sensitivity
                'fog': 1.2,       # Poor visibility, higher sensitivity
                'snow': 0.85,     # Much less activity expected
            }
            weather_factor = weather_factors.get(weather_condition.lower(), 1.0)
        
        # Apply custom factors
        if custom_factors:
            time_factor *= custom_factors.get('time', 1.0)
            activity_factor *= custom_factors.get('activity', 1.0)
            weather_factor *= custom_factors.get('weather', 1.0)
        
        # Build context description
        context_parts = []
        if self.enable_time_adaptation:
            context_parts.append(f"{time_of_day.value.title()} mode")
        if activity_factor != 1.0:
            adj = "elevated" if activity_factor < 1.0 else "reduced"
            context_parts.append(f"activity {adj}")
        context_description = ", ".join(context_parts) if context_parts else "Standard"
        
        return AdaptiveThresholds(
            watch_threshold=self.base_watch,
            warning_threshold=self.base_warning,
            critical_threshold=self.base_critical,
            time_factor=time_factor,
            activity_factor=activity_factor,
            weather_factor=weather_factor,
            context_description=context_description,
        )
    
    def update_hourly_baseline(self, hour: int, activity: float) -> None:
        """Update hourly activity baseline."""
        self.hourly_baselines[hour].append(activity)
        # Keep last 7 days of data per hour
        if len(self.hourly_baselines[hour]) > 7:
            self.hourly_baselines[hour] = self.hourly_baselines[hour][-7:]


# =============================================================================
# PREDICTION MODULE
# =============================================================================

@dataclass
class TDIPrediction:
    """Prediction of future TDI values."""
    predicted_values: List[float]      # Predicted TDI for next N frames
    prediction_horizon: int            # Number of frames predicted
    confidence_intervals: List[Tuple[float, float]]  # 95% CI for each prediction
    trend_direction: str               # "rising", "stable", "falling"
    risk_forecast: str                 # Predicted risk level
    time_to_threshold: Optional[int]   # Frames until next threshold crossing
    
    def __repr__(self):
        return f"Prediction: {self.trend_direction}, next={self.predicted_values[0]:.1f} ({self.risk_forecast})"


class TDIPredictor:
    """
    Predicts future TDI values using trend analysis and time series forecasting.
    
    "Prediction enables proactive response, not reactive panic."
    """
    
    def __init__(
        self,
        history_window: int = 50,
        prediction_horizon: int = 30,
    ):
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        
        self.tdi_history = deque(maxlen=history_window)
        self.trend_history = deque(maxlen=history_window)
    
    def update(self, tdi: float) -> None:
        """Update history with new TDI value."""
        self.tdi_history.append(tdi)
    
    def predict(self) -> TDIPrediction:
        """
        Predict future TDI values.
        
        Uses exponential smoothing and trend extrapolation.
        """
        if len(self.tdi_history) < 5:
            # Not enough data
            return TDIPrediction(
                predicted_values=[0.0] * self.prediction_horizon,
                prediction_horizon=self.prediction_horizon,
                confidence_intervals=[(0, 0)] * self.prediction_horizon,
                trend_direction="stable",
                risk_forecast="insufficient_data",
                time_to_threshold=None,
            )
        
        history = np.array(list(self.tdi_history))
        
        # Calculate trend using linear regression
        x = np.arange(len(history))
        slope, intercept = np.polyfit(x, history, 1)
        
        # Exponential smoothing parameters
        alpha = 0.3  # Smoothing factor
        
        # Calculate smoothed value
        smoothed = history[-1]
        for i in range(len(history) - 2, -1, -1):
            smoothed = alpha * history[i] + (1 - alpha) * smoothed
        
        # Generate predictions
        predictions = []
        confidence_intervals = []
        
        current_value = history[-1]
        std_error = np.std(history) if len(history) > 1 else 1.0
        
        for i in range(1, self.prediction_horizon + 1):
            # Trend-adjusted prediction
            predicted = current_value + slope * i
            
            # Reversion to mean (slight pull toward historical mean)
            mean_reversion = 0.02
            hist_mean = np.mean(history)
            predicted = predicted * (1 - mean_reversion) + hist_mean * mean_reversion
            
            # Clip to valid range
            predicted = np.clip(predicted, 0, 100)
            predictions.append(float(predicted))
            
            # Confidence interval (widens with horizon)
            ci_width = std_error * 1.96 * np.sqrt(i)
            ci_low = max(0, predicted - ci_width)
            ci_high = min(100, predicted + ci_width)
            confidence_intervals.append((float(ci_low), float(ci_high)))
        
        # Determine trend direction
        if slope > 0.5:
            trend_direction = "rising"
        elif slope < -0.5:
            trend_direction = "falling"
        else:
            trend_direction = "stable"
        
        # Risk forecast
        max_predicted = max(predictions[:10])  # Look at next 10 frames
        if max_predicted >= 60:
            risk_forecast = "CRITICAL risk imminent"
        elif max_predicted >= 40:
            risk_forecast = "WARNING level expected"
        elif max_predicted >= 20:
            risk_forecast = "WATCH level possible"
        else:
            risk_forecast = "Remaining NORMAL"
        
        # Time to threshold crossing
        time_to_threshold = None
        current_zone_threshold = 20 if history[-1] < 20 else (40 if history[-1] < 40 else 60)
        for i, pred in enumerate(predictions):
            if pred >= current_zone_threshold and history[-1] < current_zone_threshold:
                time_to_threshold = i + 1
                break
        
        return TDIPrediction(
            predicted_values=predictions,
            prediction_horizon=self.prediction_horizon,
            confidence_intervals=confidence_intervals,
            trend_direction=trend_direction,
            risk_forecast=risk_forecast,
            time_to_threshold=time_to_threshold,
        )
    
    def get_trend_stats(self) -> Dict[str, float]:
        """Get trend statistics."""
        if len(self.tdi_history) < 5:
            return {'slope': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        history = np.array(list(self.tdi_history))
        x = np.arange(len(history))
        slope, _ = np.polyfit(x, history, 1)
        
        return {
            'slope': float(slope),
            'mean': float(np.mean(history)),
            'std': float(np.std(history)),
            'min': float(np.min(history)),
            'max': float(np.max(history)),
        }


# =============================================================================
# INTEGRATED ADVANCED AI ENGINE
# =============================================================================

class AdvancedAIEngine:
    """
    Integrated engine combining all advanced AI capabilities.
    
    Provides a unified interface for:
    - Anomaly classification
    - Confidence calibration
    - Adaptive thresholding
    - TDI prediction
    """
    
    def __init__(
        self,
        feature_names: List[str],
        base_thresholds: Tuple[float, float, float] = (20.0, 40.0, 60.0),
    ):
        self.feature_names = feature_names
        
        # Initialize components
        self.classifier = AnomalyClassifier()
        self.calibrator = ConfidenceCalibrator(temperature=1.5)
        self.threshold_manager = AdaptiveThresholdManager(
            base_watch=base_thresholds[0],
            base_warning=base_thresholds[1],
            base_critical=base_thresholds[2],
        )
        self.predictor = TDIPredictor(history_window=50, prediction_horizon=30)
        
        logger.info("AdvancedAIEngine initialized with all components")
    
    def process(
        self,
        features: np.ndarray,
        z_scores: np.ndarray,
        tdi: float,
        raw_confidence: float,
        reconstruction_variance: float = 0.0,
        latent_variance: float = 0.0,
        ensemble_agreement: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process a frame through all advanced AI components.
        
        Returns dict with:
        - classification: AnomalyClassification
        - calibrated_confidence: CalibratedConfidence
        - adaptive_thresholds: AdaptiveThresholds
        - prediction: TDIPrediction
        """
        # Update predictor
        self.predictor.update(tdi)
        
        # Classify anomaly
        classification = self.classifier.classify(
            features=features,
            feature_names=self.feature_names,
            z_scores=z_scores,
            tdi=tdi,
        )
        
        # Calibrate confidence
        calibrated = self.calibrator.calibrate(
            raw_confidence=raw_confidence,
            reconstruction_variance=reconstruction_variance,
            latent_variance=latent_variance,
            ensemble_agreement=ensemble_agreement,
        )
        
        # Get adaptive thresholds
        current_activity = float(np.mean(np.abs(z_scores))) if len(z_scores) > 0 else 0.0
        thresholds = self.threshold_manager.get_thresholds(
            current_activity=current_activity,
        )
        
        # Generate prediction
        prediction = self.predictor.predict()
        
        return {
            'classification': classification,
            'calibrated_confidence': calibrated,
            'adaptive_thresholds': thresholds,
            'prediction': prediction,
            'current_activity': current_activity,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            'classifier_history_size': len(self.classifier.classification_history),
            'calibrator_temperature': self.calibrator.temperature,
            'predictor_history_size': len(self.predictor.tdi_history),
            'threshold_context': self.threshold_manager.get_thresholds().context_description,
        }
