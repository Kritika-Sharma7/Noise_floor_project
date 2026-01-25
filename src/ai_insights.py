"""
NOISE FLOOR - AI Insights Module
=================================
OpenAI-powered intelligent analysis and explanations.

Features:
- Natural language drift explanations
- Graph summaries
- Incident narratives
- Conversational Q&A
"""

import os
import json
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
except ImportError:
    client = None
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. AI insights will be disabled.")


@dataclass
class DriftContext:
    """Context for drift analysis."""
    current_zone: str
    drift_score: float
    detection_delay: Optional[int]
    false_positive_rate: float
    peak_score: float
    drift_start_frame: int
    total_frames: int
    zone_transitions: List[Dict[str, Any]]
    trend: str  # 'increasing', 'decreasing', 'stable'


class AIInsightsEngine:
    """
    AI-powered insights engine for NOISE FLOOR.
    Generates natural language explanations and summaries.
    """
    
    def __init__(self, model: str = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.enabled = OPENAI_AVAILABLE
        
        if self.enabled:
            logger.info(f"AIInsightsEngine initialized with model: {self.model}")
        else:
            logger.warning("AIInsightsEngine running in fallback mode (no API key)")
    
    def _call_openai(self, messages: List[Dict], max_tokens: int = 300) -> Optional[str]:
        """Make an OpenAI API call with error handling."""
        if not self.enabled:
            return None
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def generate_drift_explanation(self, context: DriftContext) -> str:
        """
        Generate a natural language explanation of the current drift state.
        """
        if not self.enabled:
            return self._fallback_drift_explanation(context)
        
        system_prompt = """You are an AI analyst for NOISE FLOOR, a behavioral drift detection system used in security monitoring.
Your role is to explain drift detection results in clear, professional language suitable for security operators.
Be concise (2-3 sentences), actionable, and avoid technical jargon. Focus on what the operator needs to know and do."""

        user_prompt = f"""Analyze this drift detection state and provide a brief explanation:

Current Zone: {context.current_zone}
Drift Score: {context.drift_score:.2f}
Peak Score: {context.peak_score:.2f}
Detection Delay: {context.detection_delay if context.detection_delay else 'N/A'} frames
False Positive Rate: {context.false_positive_rate:.1f}%
Trend: {context.trend}
Frames Analyzed: {context.total_frames}
Drift Started: Frame {context.drift_start_frame}

Provide a 2-3 sentence explanation of what this means for the operator."""

        response = self._call_openai([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], max_tokens=150)
        
        return response or self._fallback_drift_explanation(context)
    
    def generate_graph_summary(self, context: DriftContext) -> str:
        """
        Generate a summary of what the drift score timeline graph shows.
        """
        if not self.enabled:
            return self._fallback_graph_summary(context)
        
        system_prompt = """You are an AI analyst for NOISE FLOOR, explaining data visualizations to security operators.
Write a brief, clear summary of what the drift score timeline graph reveals.
Be specific about patterns, transitions, and key moments. Keep it to 2-3 sentences."""

        # Calculate zone distribution
        zone_counts = {}
        for transition in context.zone_transitions:
            zone = transition.get('zone', 'UNKNOWN')
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        user_prompt = f"""Summarize this drift score timeline for an operator:

Timeline Overview:
- Total frames: {context.total_frames}
- Drift onset: Frame {context.drift_start_frame}
- Current zone: {context.current_zone}
- Peak drift score: {context.peak_score:.2f}
- Score trend: {context.trend}
- Detection delay: {context.detection_delay if context.detection_delay else 'N/A'} frames after drift started

Zone Transitions: {len(context.zone_transitions)} changes detected
Final State: {context.current_zone} zone

Write a 2-3 sentence summary of what this timeline shows."""

        response = self._call_openai([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], max_tokens=150)
        
        return response or self._fallback_graph_summary(context)
    
    def generate_zone_insight(self, zone: str, drift_score: float, confidence: float) -> str:
        """
        Generate a brief insight about the current zone status.
        """
        if not self.enabled:
            return self._fallback_zone_insight(zone, drift_score)
        
        system_prompt = """You are a concise AI assistant for a security monitoring system.
Generate a single sentence describing the current zone status and recommended vigilance level.
Be calm and professional - not alarmist."""

        user_prompt = f"""Current state:
- Zone: {zone}
- Drift Score: {drift_score:.2f}
- Confidence: {confidence:.0%}

Generate ONE sentence about what this means."""

        response = self._call_openai([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], max_tokens=60)
        
        return response or self._fallback_zone_insight(zone, drift_score)
    
    def generate_comparison_insight(self, results: Dict[str, Any], drift_start: int) -> str:
        """
        Generate insight comparing NOISE FLOOR to baseline methods.
        """
        if not self.enabled:
            return self._fallback_comparison_insight(results)
        
        system_prompt = """You are an AI analyst explaining method comparison results.
Highlight NOISE FLOOR's advantages in 2-3 sentences. Be objective but note meaningful differences."""

        # Format results for the prompt
        methods_summary = []
        for name, result in results.items():
            methods_summary.append(f"- {result.method_name}: Detection delay {result.detection_delay} frames, FP rate {result.false_positive_rate*100:.1f}%")
        
        user_prompt = f"""Compare these drift detection methods (drift started at frame {drift_start}):

{chr(10).join(methods_summary)}

Summarize NOISE FLOOR's performance vs traditional methods in 2-3 sentences."""

        response = self._call_openai([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], max_tokens=150)
        
        return response or self._fallback_comparison_insight(results)
    
    def generate_metrics_summary(self, detection_delay: int, fp_rate: float, 
                                  peak_score: float, drift_start: int) -> str:
        """
        Generate a summary of the detection performance metrics.
        """
        if not self.enabled:
            return self._fallback_metrics_summary(detection_delay, fp_rate)
        
        system_prompt = """You are an AI analyst summarizing detection performance metrics.
Provide a brief, professional assessment in 1-2 sentences. Focus on practical implications."""

        user_prompt = f"""Summarize these detection metrics:
- Detection Delay: {detection_delay} frames after drift onset
- False Positive Rate: {fp_rate:.1f}%
- Peak Drift Score: {peak_score:.1f}
- Drift Started: Frame {drift_start}

What do these metrics tell us about system performance? (1-2 sentences)"""

        response = self._call_openai([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], max_tokens=100)
        
        return response or self._fallback_metrics_summary(detection_delay, fp_rate)
    
    def chat_query(self, question: str, context: DriftContext) -> str:
        """
        Answer operator questions about the current monitoring session.
        """
        if not self.enabled:
            return "AI chat is unavailable. Please add your OpenAI API key to the .env file."
        
        system_prompt = """You are an AI assistant for NOISE FLOOR, a behavioral drift detection system.
Answer operator questions about the current monitoring session clearly and concisely.
You have access to the current drift detection context. Be helpful but brief."""

        context_str = f"""Current Session Context:
- Zone: {context.current_zone}
- Drift Score: {context.drift_score:.2f}
- Peak Score: {context.peak_score:.2f}
- Detection Delay: {context.detection_delay} frames
- False Positive Rate: {context.false_positive_rate:.1f}%
- Trend: {context.trend}
- Total Frames: {context.total_frames}
- Drift Started: Frame {context.drift_start_frame}"""

        response = self._call_openai([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_str}\n\nOperator Question: {question}"}
        ], max_tokens=200)
        
        return response or "Unable to process your question. Please try again."
    
    # =========================================================================
    # FALLBACK METHODS (when OpenAI is unavailable)
    # =========================================================================
    
    def _fallback_drift_explanation(self, context: DriftContext) -> str:
        """Fallback explanation without AI."""
        zone_descriptions = {
            'NORMAL': f"System operating normally with drift score of {context.drift_score:.2f}. No action required.",
            'WATCH': f"Minor deviation detected (score: {context.drift_score:.2f}). Elevated monitoring recommended.",
            'WARNING': f"Significant drift in progress (score: {context.drift_score:.2f}). Review behavioral patterns and prepare response protocols.",
            'ALERT': f"Critical drift threshold exceeded (score: {context.drift_score:.2f}). Immediate review of monitoring feed recommended."
        }
        return zone_descriptions.get(context.current_zone, "Status unknown.")
    
    def _fallback_graph_summary(self, context: DriftContext) -> str:
        """Fallback graph summary without AI."""
        if context.detection_delay is not None:
            return f"Timeline shows drift detection {context.detection_delay} frames after onset at frame {context.drift_start_frame}. Peak deviation reached {context.peak_score:.1f}. Current trend: {context.trend}."
        return f"Timeline shows behavioral patterns across {context.total_frames} frames with peak score of {context.peak_score:.1f}."
    
    def _fallback_zone_insight(self, zone: str, drift_score: float) -> str:
        """Fallback zone insight without AI."""
        insights = {
            'NORMAL': "Behavioral patterns within expected baseline parameters.",
            'WATCH': "Minor deviation from baseline detected; increased attention advised.",
            'WARNING': "Notable behavioral shift in progress; review recommended.",
            'ALERT': "Significant anomaly detected; immediate attention required."
        }
        return insights.get(zone, "Zone status under evaluation.")
    
    def _fallback_comparison_insight(self, results: Dict) -> str:
        """Fallback comparison insight without AI."""
        nf_result = results.get('noise_floor')
        if nf_result:
            return f"NOISE FLOOR detected drift with {nf_result.detection_delay} frame delay and {nf_result.false_positive_rate*100:.1f}% false positive rate, demonstrating temporal smoothing benefits."
        return "Comparison analysis complete. Review table for detailed metrics."
    
    def _fallback_metrics_summary(self, detection_delay: int, fp_rate: float) -> str:
        """Fallback metrics summary without AI."""
        quality = "excellent" if detection_delay < 20 and fp_rate < 5 else "good" if detection_delay < 50 else "acceptable"
        return f"Detection performance is {quality} with {detection_delay} frame delay and {fp_rate:.1f}% false positive rate."


# Singleton instance for easy import
insights_engine = AIInsightsEngine()


def get_drift_context(history: dict, drift_start: int, thresholds: dict) -> DriftContext:
    """
    Build a DriftContext from session history.
    """
    if not history.get('results'):
        return DriftContext(
            current_zone='NORMAL',
            drift_score=0.0,
            detection_delay=None,
            false_positive_rate=0.0,
            peak_score=0.0,
            drift_start_frame=drift_start,
            total_frames=0,
            zone_transitions=[],
            trend='stable'
        )
    
    results = history['results']
    drift_scores = history['drift_scores']
    zones = history['zones']
    
    # Calculate metrics
    latest = results[-1]
    current_zone = str(latest['zone']).upper()
    
    # Find first detection after drift start
    first_detection_idx = None
    for i in range(drift_start, len(zones)):
        if zones[i] in ['WATCH', 'WARNING', 'ALERT']:
            first_detection_idx = i
            break
    
    detection_delay = (first_detection_idx - drift_start) if first_detection_idx else None
    
    # False positive rate
    false_positives = sum(1 for i in range(min(drift_start, len(zones))) 
                         if zones[i] in ['WATCH', 'WARNING', 'ALERT'])
    fp_rate = (false_positives / drift_start * 100) if drift_start > 0 else 0
    
    # Zone transitions
    zone_transitions = []
    prev_zone = None
    for i, zone in enumerate(zones):
        if zone != prev_zone:
            zone_transitions.append({'frame': i, 'zone': zone})
            prev_zone = zone
    
    # Determine trend from last 30 frames
    if len(drift_scores) > 30:
        recent = drift_scores[-30:]
        if recent[-1] > recent[0] * 1.1:
            trend = 'increasing'
        elif recent[-1] < recent[0] * 0.9:
            trend = 'decreasing'
        else:
            trend = 'stable'
    else:
        trend = 'stable'
    
    return DriftContext(
        current_zone=current_zone,
        drift_score=latest['drift_score'],
        detection_delay=detection_delay,
        false_positive_rate=fp_rate,
        peak_score=max(drift_scores) if drift_scores else 0,
        drift_start_frame=drift_start,
        total_frames=len(results),
        zone_transitions=zone_transitions,
        trend=trend
    )
