"""
NOISE FLOOR - Drift Attribution & Explainability (XAI)
=======================================================
Explainable AI for border surveillance drift detection.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

EXPLAINABILITY PHILOSOPHY:
--------------------------
Defense systems must be EXPLAINABLE to operators:
1. WHY is the system flagging this as drift?
2. WHICH features are contributing to the alert?
3. HOW confident is the system in its assessment?

This module provides:
- Top contributing features identification
- Natural language explanations
- Confidence metrics
- Attribution visualization data

NO DEEP MATH IN OUTPUTS:
------------------------
Operators don't need to understand KL divergence.
They need to know: "Motion dispersion increased by 40%"

ATTRIBUTION METHODS:
--------------------
1. Feature Z-score ranking (primary)
2. Reconstruction error decomposition
3. Latent space contribution analysis
4. Temporal pattern attribution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttributionType(Enum):
    """Types of drift attribution."""
    FEATURE_DEVIATION = "feature_deviation"     # Feature exceeds baseline
    PATTERN_CHANGE = "pattern_change"           # Temporal pattern changed
    CORRELATION_SHIFT = "correlation_shift"     # Feature relationships changed
    TREND_ACCELERATION = "trend_acceleration"   # Rate of change increased


@dataclass
class FeatureAttribution:
    """
    Attribution for a single feature's contribution to drift.
    """
    feature_name: str
    feature_index: int
    
    # Deviation metrics
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float                      # Standard deviations from baseline
    
    # Contribution to overall drift
    contribution_pct: float             # Percentage contribution to drift score
    
    # Direction
    direction: str                      # "increased" or "decreased"
    
    # Attribution type
    attribution_type: AttributionType = AttributionType.FEATURE_DEVIATION
    
    def to_natural_language(self) -> str:
        """Convert to operator-friendly explanation."""
        if abs(self.z_score) < 1.0:
            return None  # Not significant
        
        magnitude = "slightly" if abs(self.z_score) < 2 else "significantly" if abs(self.z_score) < 3 else "dramatically"
        
        # Generate human-readable feature name
        readable_name = self._humanize_feature_name()
        
        return f"{readable_name} has {magnitude} {self.direction}"
    
    def _humanize_feature_name(self) -> str:
        """Convert technical feature name to readable format."""
        name_map = {
            'motion_energy': 'Overall motion activity',
            'motion_energy_std': 'Motion variability',
            'peak_motion': 'Peak motion intensity',
            'flow_magnitude_mean': 'Average movement speed',
            'flow_magnitude_std': 'Speed consistency',
            'flow_magnitude_max': 'Maximum movement speed',
            'flow_variance': 'Movement dispersion',
            'direction_consistency': 'Directional coordination',
            'direction_entropy': 'Movement randomness',
            'dominant_direction': 'Primary movement direction',
            'direction_change_rate': 'Direction changes',
            'scene_entropy': 'Scene complexity',
            'scene_entropy_change': 'Scene change rate',
            'spatial_coherence': 'Spatial organization',
            'velocity_mean': 'Average velocity',
            'velocity_std': 'Velocity consistency',
            'velocity_variance': 'Velocity spread',
            'acceleration_mean': 'Average acceleration',
            'idle_ratio': 'Idle time proportion',
            'active_ratio': 'Active time proportion',
            'activity_transitions': 'Activity state changes',
            'temporal_gradient': 'Rate of change',
            'temporal_stability': 'Pattern stability',
            'movement_complexity': 'Movement complexity',
        }
        
        return name_map.get(self.feature_name, self.feature_name.replace('_', ' ').title())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'readable_name': self._humanize_feature_name(),
            'z_score': self.z_score,
            'direction': self.direction,
            'contribution_pct': self.contribution_pct,
            'explanation': self.to_natural_language(),
        }


@dataclass
class DriftExplanation:
    """
    Complete explanation for a drift detection event.
    """
    # Summary
    summary: str                                # One-line summary
    confidence_statement: str                   # Confidence in explanation
    
    # Top contributors
    top_features: List[FeatureAttribution]      # Ranked by contribution
    
    # Detailed explanations
    bullet_points: List[str]                    # Key observations
    
    # Recommended actions
    recommended_actions: List[str]
    
    # Metadata
    threat_deviation_index: float
    risk_zone: str
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'summary': self.summary,
            'confidence_statement': self.confidence_statement,
            'top_features': [f.to_dict() for f in self.top_features],
            'bullet_points': self.bullet_points,
            'recommended_actions': self.recommended_actions,
            'threat_deviation_index': self.threat_deviation_index,
            'risk_zone': self.risk_zone,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown format for display."""
        lines = [
            f"## Drift Analysis",
            f"",
            f"**Summary:** {self.summary}",
            f"",
            f"**Confidence:** {self.confidence_statement}",
            f"",
            f"### Key Observations",
        ]
        
        for bullet in self.bullet_points:
            lines.append(f"- {bullet}")
        
        lines.extend([
            f"",
            f"### Top Contributing Factors",
        ])
        
        for i, feat in enumerate(self.top_features[:5], 1):
            explanation = feat.to_natural_language()
            if explanation:
                lines.append(f"{i}. {explanation}")
        
        lines.extend([
            f"",
            f"### Recommended Actions",
        ])
        
        for action in self.recommended_actions:
            lines.append(f"- {action}")
        
        return "\n".join(lines)


class DriftAttributor:
    """
    Computes feature attributions for drift detection.
    
    ATTRIBUTION PIPELINE:
    --------------------
    1. Compare current features to baseline
    2. Compute Z-scores for each feature
    3. Rank by contribution to drift
    4. Generate natural language explanations
    5. Produce actionable insights
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        baseline_means: Optional[np.ndarray] = None,
        baseline_stds: Optional[np.ndarray] = None,
        significance_threshold: float = 1.5,    # Z-score for significance
    ):
        self.feature_names = feature_names or []
        self.baseline_means = baseline_means
        self.baseline_stds = baseline_stds
        self.significance_threshold = significance_threshold
        
        # Baseline can be set later
        self.is_baseline_set = baseline_means is not None
    
    def set_baseline(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """Set baseline statistics for attribution."""
        self.baseline_means = means
        self.baseline_stds = stds + 1e-6  # Prevent division by zero
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        self.is_baseline_set = True
        logger.info(f"Attribution baseline set with {len(means)} features")
    
    def compute_feature_attributions(
        self,
        current_features: np.ndarray,
        top_k: int = 5
    ) -> List[FeatureAttribution]:
        """
        Compute feature-level attributions.
        
        Args:
            current_features: Current feature vector
            top_k: Number of top contributors to return
            
        Returns:
            List of FeatureAttribution objects, sorted by |z_score|
        """
        if not self.is_baseline_set:
            return []
        
        # Ensure 1D
        if len(current_features.shape) > 1:
            current_features = current_features.flatten()
        
        # Compute Z-scores
        z_scores = (current_features - self.baseline_means) / self.baseline_stds
        
        # Compute absolute contributions
        abs_z = np.abs(z_scores)
        total_deviation = abs_z.sum() + 1e-6
        contributions = abs_z / total_deviation * 100
        
        # Create attributions
        attributions = []
        for i, (z, contrib) in enumerate(zip(z_scores, contributions)):
            # Get feature name
            if i < len(self.feature_names):
                name = self.feature_names[i]
            else:
                name = f"feature_{i}"
            
            # Determine direction
            direction = "increased" if z > 0 else "decreased"
            
            attr = FeatureAttribution(
                feature_name=name,
                feature_index=i,
                current_value=float(current_features[i]),
                baseline_mean=float(self.baseline_means[i]),
                baseline_std=float(self.baseline_stds[i]),
                z_score=float(z),
                contribution_pct=float(contrib),
                direction=direction,
            )
            
            attributions.append(attr)
        
        # Sort by absolute Z-score
        attributions.sort(key=lambda x: abs(x.z_score), reverse=True)
        
        # Filter to significant only
        significant = [a for a in attributions if abs(a.z_score) >= self.significance_threshold]
        
        return significant[:top_k]
    
    def generate_explanation(
        self,
        current_features: np.ndarray,
        threat_deviation_index: float,
        risk_zone: str,
        trend_direction: str = "stable",
        confidence: float = 1.0,
        timestamp: float = 0.0
    ) -> DriftExplanation:
        """
        Generate complete drift explanation.
        
        This is the MAIN METHOD for producing operator-facing explanations.
        """
        # Get top feature attributions
        top_features = self.compute_feature_attributions(current_features, top_k=5)
        
        # Generate summary
        summary = self._generate_summary(
            threat_deviation_index, risk_zone, top_features, trend_direction
        )
        
        # Generate confidence statement
        confidence_statement = self._generate_confidence_statement(confidence)
        
        # Generate bullet points
        bullet_points = self._generate_bullet_points(
            top_features, trend_direction, threat_deviation_index
        )
        
        # Generate recommended actions
        recommended_actions = self._generate_actions(risk_zone, top_features)
        
        return DriftExplanation(
            summary=summary,
            confidence_statement=confidence_statement,
            top_features=top_features,
            bullet_points=bullet_points,
            recommended_actions=recommended_actions,
            threat_deviation_index=threat_deviation_index,
            risk_zone=risk_zone,
            timestamp=timestamp,
        )
    
    def _generate_summary(
        self,
        tdi: float,
        zone: str,
        features: List[FeatureAttribution],
        trend: str
    ) -> str:
        """Generate one-line summary."""
        # Zone-based opening
        zone_phrases = {
            'normal': 'Behavioral patterns within normal parameters',
            'watch': 'Elevated behavioral deviation detected',
            'warning': 'Significant behavioral drift confirmed',
            'critical': 'Critical behavioral anomaly detected',
        }
        
        opening = zone_phrases.get(zone.lower(), 'Behavioral pattern analysis complete')
        
        # Add primary contributor if available
        if features and abs(features[0].z_score) >= 2.0:
            primary = features[0]
            contributor = f", primarily driven by {primary._humanize_feature_name().lower()}"
        else:
            contributor = ""
        
        # Add trend
        trend_phrases = {
            'increasing': ' (escalating)',
            'decreasing': ' (improving)',
            'stable': '',
        }
        trend_suffix = trend_phrases.get(trend, '')
        
        return f"{opening}{contributor}{trend_suffix}."
    
    def _generate_confidence_statement(self, confidence: float) -> str:
        """Generate confidence level statement."""
        if confidence >= 0.9:
            return "High confidence in this assessment based on strong signal clarity."
        elif confidence >= 0.7:
            return "Moderate confidence. Continue monitoring for pattern confirmation."
        elif confidence >= 0.5:
            return "Low confidence due to signal variability. Additional observation recommended."
        else:
            return "Very low confidence. Assessment may be unreliable due to insufficient data."
    
    def _generate_bullet_points(
        self,
        features: List[FeatureAttribution],
        trend: str,
        tdi: float
    ) -> List[str]:
        """Generate key observation bullet points."""
        bullets = []
        
        # TDI summary
        if tdi > 60:
            bullets.append(f"Threat Deviation Index at {tdi:.0f}% - significantly elevated")
        elif tdi > 30:
            bullets.append(f"Threat Deviation Index at {tdi:.0f}% - moderately elevated")
        else:
            bullets.append(f"Threat Deviation Index at {tdi:.0f}% - within acceptable range")
        
        # Trend observation
        if trend == 'increasing':
            bullets.append("Deviation trend is INCREASING - situation may be escalating")
        elif trend == 'decreasing':
            bullets.append("Deviation trend is DECREASING - situation may be improving")
        
        # Feature-specific observations
        for feat in features[:3]:
            explanation = feat.to_natural_language()
            if explanation:
                bullets.append(explanation)
        
        # Pattern observations
        motion_features = [f for f in features if 'motion' in f.feature_name.lower()]
        if motion_features and any(abs(f.z_score) > 2 for f in motion_features):
            bullets.append("Motion patterns show significant deviation from baseline")
        
        direction_features = [f for f in features if 'direction' in f.feature_name.lower()]
        if direction_features and any(abs(f.z_score) > 2 for f in direction_features):
            bullets.append("Movement coordination patterns have changed")
        
        return bullets
    
    def _generate_actions(
        self,
        zone: str,
        features: List[FeatureAttribution]
    ) -> List[str]:
        """Generate recommended actions based on zone and features."""
        actions = []
        
        zone_actions = {
            'normal': [
                "Continue standard monitoring protocols",
                "No immediate action required",
            ],
            'watch': [
                "Increase observation frequency for affected sector",
                "Prepare response assets for potential deployment",
                "Monitor for continued deviation",
            ],
            'warning': [
                "Alert response team to elevated status",
                "Initiate sector verification procedures",
                "Consider deploying additional observation resources",
                "Document current conditions for analysis",
            ],
            'critical': [
                "IMMEDIATE: Dispatch response units to sector",
                "Initiate full verification protocol",
                "Alert command center of critical status",
                "Preserve all sensor data for analysis",
                "Coordinate with adjacent sectors",
            ],
        }
        
        actions.extend(zone_actions.get(zone.lower(), zone_actions['normal']))
        
        # Feature-specific actions
        motion_deviation = any(
            'motion' in f.feature_name and abs(f.z_score) > 2 
            for f in features
        )
        if motion_deviation:
            actions.append("Review visual feed for unusual activity patterns")
        
        return actions


class ExplainabilityReport:
    """
    Generates comprehensive explainability reports for stakeholders.
    """
    
    def __init__(self, attributor: DriftAttributor):
        self.attributor = attributor
        self.explanations_history: List[DriftExplanation] = []
    
    def add_explanation(self, explanation: DriftExplanation):
        """Add explanation to history."""
        self.explanations_history.append(explanation)
    
    def generate_session_report(self) -> str:
        """Generate report for the monitoring session."""
        if not self.explanations_history:
            return "No drift events recorded in this session."
        
        # Count events by zone
        zone_counts = {}
        for exp in self.explanations_history:
            zone = exp.risk_zone
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        # Find top features across all events
        all_features = []
        for exp in self.explanations_history:
            all_features.extend(exp.top_features)
        
        feature_counts = {}
        for f in all_features:
            name = f.feature_name
            feature_counts[name] = feature_counts.get(name, 0) + 1
        
        top_recurring = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Build report
        lines = [
            "# NOISE FLOOR - Session Explainability Report",
            "",
            "## Summary",
            f"Total monitoring windows analyzed: {len(self.explanations_history)}",
            "",
            "## Zone Distribution",
        ]
        
        for zone, count in sorted(zone_counts.items()):
            pct = count / len(self.explanations_history) * 100
            lines.append(f"- {zone.upper()}: {count} ({pct:.1f}%)")
        
        lines.extend([
            "",
            "## Most Frequent Contributing Factors",
        ])
        
        for name, count in top_recurring:
            lines.append(f"- {name.replace('_', ' ').title()}: appeared in {count} events")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_attributor(
    feature_names: List[str],
    baseline_means: np.ndarray,
    baseline_stds: np.ndarray
) -> DriftAttributor:
    """
    Create a configured drift attributor.
    """
    return DriftAttributor(
        feature_names=feature_names,
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
    )


def get_default_feature_names() -> List[str]:
    """Return default behavioral feature names."""
    return [
        'motion_energy',
        'motion_energy_std',
        'peak_motion',
        'flow_magnitude_mean',
        'flow_magnitude_std',
        'flow_magnitude_max',
        'flow_variance',
        'direction_consistency',
        'direction_entropy',
        'dominant_direction',
        'direction_change_rate',
        'scene_entropy',
        'scene_entropy_change',
        'spatial_coherence',
        'velocity_mean',
        'velocity_std',
        'velocity_variance',
        'acceleration_mean',
        'idle_ratio',
        'active_ratio',
        'activity_transitions',
        'temporal_gradient',
        'temporal_stability',
        'movement_complexity',
    ]
