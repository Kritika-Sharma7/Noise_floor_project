"""
NOISE FLOOR - Defense-Grade Intelligence Dashboard
===================================================
Border surveillance and high-security perimeter monitoring.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

TECHNOLOGY READINESS LEVEL: TRL-4
Lab-validated prototype for decision-support intelligence.
This is NOT an autonomous system - AI assists operators, it does NOT replace them.

SYSTEM PHILOSOPHY:
- "Defense systems manage CONFIDENCE, not panic."
- "AI assists operators, it does NOT replace them."
- "Baseline adaptation is human-gated."

OUTPUTS:
--------
• Threat Deviation Index (TDI): 0-100 scale
• Risk Zone: Normal / Watch / Warning / Critical
• Drift Trend: ↑ Rising / → Stable / ↓ Falling
• Drift Onset Timestamp
• Top Contributing Features
• Confidence Level

DATA MODES:
-----------
• synthetic   - Controlled testing with generated behavioral data
• real_video  - Real surveillance footage (UCSD dataset or custom)

Run with: streamlit run dashboard/app_intelligence.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
from config import DATA_MODE, UCSD_DATASET_PATH, UCSD_SUBSET, BASELINE_FREEZE_CONFIG

# Import new intelligence modules
from src.behavioral_features import BehavioralFeatureExtractor, BEHAVIORAL_FEATURES
from src.behavioral_features import create_synthetic_normal_data, create_synthetic_drift_data
from src.lstm_vae import TemporalNormalityLSTMVAE
from src.drift_intelligence import DriftIntelligenceEngine, DriftTrend
from src.risk_zones import RiskZoneClassifier, RiskZone
from src.explainability import DriftAttributor, ExplainabilityReport
from src.feedback_system import HumanInTheLoop, FeedbackType, OperatorAction
from src.context_augmentation import ContextAugmenter, SceneContext
from src.baseline_freeze import BaselineFreezeManager

# Import video feature extractor
try:
    from src.video_features import RealVideoFeatureExtractor, UCSDDatasetLoader
    VIDEO_FEATURES_AVAILABLE = True
except ImportError:
    VIDEO_FEATURES_AVAILABLE = False

# Fallback to old modules if new ones not available
try:
    from src.ai_insights import insights_engine, OPENAI_AVAILABLE
except ImportError:
    OPENAI_AVAILABLE = False
    insights_engine = None

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NOISE FLOOR - Defense Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DEFENSE DESIGN SYSTEM CSS
# =============================================================================
st.markdown("""
<style>
    :root {
        --bg-primary: #0a0d12;
        --bg-secondary: #0f1318;
        --bg-card: rgba(15, 19, 24, 0.9);
        --bg-glass: rgba(15, 19, 24, 0.7);
        
        --border-subtle: rgba(255, 255, 255, 0.05);
        --border-medium: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(34, 197, 94, 0.3);
        
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-tertiary: #64748b;
        --text-muted: #475569;
        
        --zone-normal: #22c55e;
        --zone-watch: #eab308;
        --zone-warning: #f97316;
        --zone-critical: #ef4444;
        
        --accent-primary: #22c55e;
        --accent-blue: #3b82f6;
    }

    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Command Bar */
    .command-bar {
        background: linear-gradient(90deg, rgba(10, 13, 18, 0.98), rgba(15, 19, 24, 0.98));
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .system-id {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .system-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--zone-normal), #16a34a);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
    }
    
    .system-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    .system-subtitle {
        font-size: 0.7rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .status-badge.normal {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: var(--zone-normal);
    }
    
    .status-badge.watch {
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.4);
        color: var(--zone-watch);
    }
    
    .status-badge.warning {
        background: rgba(249, 115, 22, 0.15);
        border: 1px solid rgba(249, 115, 22, 0.4);
        color: var(--zone-warning);
        animation: pulse-warning 2s ease-in-out infinite;
    }
    
    .status-badge.critical {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: var(--zone-critical);
        animation: pulse-critical 1s ease-in-out infinite;
    }
    
    @keyframes pulse-warning {
        0%, 100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.3); }
        50% { box-shadow: 0 0 0 8px rgba(249, 115, 22, 0); }
    }
    
    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    }
    
    /* Primary Intelligence Card (TDI) */
    .tdi-card {
        background: linear-gradient(135deg, rgba(15, 19, 24, 0.95), rgba(10, 13, 18, 0.98));
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .tdi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--zone-normal), var(--zone-watch), var(--zone-warning), var(--zone-critical));
    }
    
    .tdi-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    
    .tdi-value {
        font-size: 4rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
        margin-bottom: 8px;
    }
    
    .tdi-value.normal { color: var(--zone-normal); }
    .tdi-value.watch { color: var(--zone-watch); }
    .tdi-value.warning { color: var(--zone-warning); }
    .tdi-value.critical { color: var(--zone-critical); }
    
    .tdi-scale {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    /* Risk Zone Card */
    .zone-card {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 20px;
    }
    
    .zone-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    
    .zone-dot {
        width: 24px;
        height: 24px;
        border-radius: 50%;
    }
    
    .zone-dot.normal { background: var(--zone-normal); box-shadow: 0 0 12px var(--zone-normal); }
    .zone-dot.watch { background: var(--zone-watch); box-shadow: 0 0 12px var(--zone-watch); }
    .zone-dot.warning { background: var(--zone-warning); box-shadow: 0 0 12px var(--zone-warning); }
    .zone-dot.critical { background: var(--zone-critical); box-shadow: 0 0 12px var(--zone-critical); }
    
    .zone-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .zone-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    /* Trend Card */
    .trend-card {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .trend-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .trend-arrow {
        font-size: 3rem;
        line-height: 1;
    }
    
    .trend-arrow.rising { color: var(--zone-critical); }
    .trend-arrow.stable { color: var(--zone-watch); }
    .trend-arrow.falling { color: var(--zone-normal); }
    
    .trend-text {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 8px;
    }
    
    /* Confidence Card */
    .confidence-card {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .confidence-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--accent-blue);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Feature Attribution */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 20px;
    }
    
    .feature-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    
    .feature-name {
        flex: 1;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .feature-bar-container {
        width: 120px;
        height: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .feature-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    .feature-score {
        width: 50px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        text-align: right;
    }
    
    /* Onset Timestamp */
    .onset-card {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 20px;
    }
    
    .onset-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .onset-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--zone-warning);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .onset-elapsed {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 4px;
    }
    
    /* Explanation Card */
    .explanation-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), var(--bg-card));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 20px;
    }
    
    .explanation-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
    
    .explanation-icon {
        font-size: 1.2rem;
    }
    
    .explanation-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--accent-blue);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .explanation-text {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Feedback Panel */
    .feedback-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 20px;
    }
    
    .feedback-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }
    
    .feedback-btn {
        display: block;
        width: 100%;
        padding: 10px 16px;
        margin-bottom: 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .feedback-btn.confirm {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: var(--zone-normal);
    }
    
    .feedback-btn.benign {
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.3);
        color: var(--zone-watch);
    }
    
    .feedback-btn.investigate {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: var(--zone-critical);
    }
    
    /* Section Header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 16px 0;
    }
    
    .section-icon {
        font-size: 1.2rem;
    }
    
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Streamlit Button Override */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), #16a34a) !important;
        border: none !important;
        color: #0a0d12 !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-primary), var(--bg-secondary));
        border-right: 1px solid var(--border-subtle);
    }
    
    .sidebar-section {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .sidebar-section-title {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }
    
    /* Timeline Chart Container */
    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'initialized': False,
        'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'lstm_vae': None,
        'drift_engine': None,
        'zone_classifier': None,
        'attributor': None,
        'feedback_system': None,
        'baseline_means': None,
        'baseline_stds': None,
        'history': {
            'tdi': [],
            'zones': [],
            'trends': [],
            'confidences': [],
            'timestamps': [],
            'features': [],
            'explanations': [],
        },
        'drift_onset_frame': None,
        'drift_onset_time': None,
        'last_trained': 'Not trained',
        'current_alert_id': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
@st.cache_resource
def initialize_intelligence_system():
    """Initialize all intelligence system components."""
    
    # Feature extraction
    feature_extractor = BehavioralFeatureExtractor()
    
    # Generate training data (normal behavior only)
    train_data = create_synthetic_normal_data(num_samples=500, feature_dim=len(BEHAVIORAL_FEATURES))
    
    # Compute baseline statistics
    baseline_means = np.mean(train_data, axis=0)
    baseline_stds = np.std(train_data, axis=0) + 1e-6
    
    # LSTM-VAE for temporal normality learning
    lstm_vae = TemporalNormalityLSTMVAE(
        input_dim=len(BEHAVIORAL_FEATURES),
        hidden_dim=32,
        latent_dim=8,
        seq_len=10,
    )
    
    # Create sequences for training
    sequences = []
    for i in range(len(train_data) - 10):
        seq = train_data[i:i+10]
        sequences.append(seq)
    
    if sequences:
        sequences = np.array(sequences)
        lstm_vae.train(sequences, epochs=50)
    
    # Drift Intelligence Engine
    drift_engine = DriftIntelligenceEngine(
        baseline_frames=50,
        ewma_alpha=0.1,
        feature_names=BEHAVIORAL_FEATURES,
    )
    
    # Risk Zone Classifier
    zone_classifier = RiskZoneClassifier()
    
    # Drift Attributor for explainability
    attributor = DriftAttributor(
        feature_names=BEHAVIORAL_FEATURES,
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
    )
    
    # Human-in-the-loop feedback system
    feedback_system = HumanInTheLoop(
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
        storage_path="./feedback_data",
    )
    
    return (lstm_vae, drift_engine, zone_classifier, attributor, 
            feedback_system, baseline_means, baseline_stds, feature_extractor)


# =============================================================================
# INTELLIGENCE PROCESSING
# =============================================================================
def process_frame(
    features: np.ndarray,
    frame_index: int,
    lstm_vae: TemporalNormalityLSTMVAE,
    drift_engine: DriftIntelligenceEngine,
    zone_classifier: RiskZoneClassifier,
    attributor: DriftAttributor,
    feature_buffer: list,
    baseline_means: np.ndarray,
) -> dict:
    """
    Process a single frame through the intelligence pipeline.
    
    Returns dict with: tdi, zone, trend, confidence, top_features, explanation
    """
    
    # Add to feature buffer for temporal analysis
    feature_buffer.append(features)
    
    # Keep buffer at sequence length
    if len(feature_buffer) > 10:
        feature_buffer.pop(0)
    
    # Compute metrics from LSTM-VAE
    if len(feature_buffer) >= 10:
        sequence = np.array(feature_buffer[-10:])
        seq_input = sequence.reshape(1, 10, -1)
        
        # Get detailed output from LSTM-VAE
        output = lstm_vae.forward(seq_input, training=False)
        reconstruction_loss = output.reconstruction_loss
        kl_divergence = output.kl_divergence
        latent_mean = output.latent_mean[0]
        latent_logvar = output.latent_log_var[0]
    else:
        # Simple fallback for early frames
        reconstruction_loss = float(np.mean((features - baseline_means) ** 2))
        kl_divergence = 0.0
        latent_mean = np.zeros(8)
        latent_logvar = np.zeros(8)
    
    # Process through drift intelligence engine
    intelligence = drift_engine.process(
        reconstruction_loss=reconstruction_loss,
        kl_divergence=kl_divergence,
        latent_mean=latent_mean,
        latent_logvar=latent_logvar,
        features=features,
        frame_index=frame_index,
    )
    
    # Classify risk zone
    zone_state = zone_classifier.classify(
        threat_deviation_index=intelligence.threat_deviation_index,
        z_score=intelligence.z_score,
        trend_slope=intelligence.trend_slope,
        trend_persistence=intelligence.trend_persistence,
    )
    
    # Get feature attribution
    attributions = attributor.compute_feature_attributions(
        current_features=features,
        top_k=5,
    )
    
    # Generate explanation
    explanation_obj = attributor.generate_explanation(
        current_features=features,
        threat_deviation_index=intelligence.threat_deviation_index,
        risk_zone=zone_state.zone.name,
        trend_direction=intelligence.drift_trend.name,
        confidence=intelligence.confidence,
    )
    
    # Convert attributions to dict format for dashboard
    top_features = [
        {'name': a.feature_name, 'score': a.z_score}
        for a in attributions[:5]
    ]
    
    return {
        'tdi': intelligence.threat_deviation_index,
        'zone': zone_state.zone,
        'zone_name': zone_state.zone.name,
        'trend': intelligence.drift_trend,
        'trend_name': intelligence.drift_trend.name,
        'confidence': intelligence.confidence,
        'kl_divergence': kl_divergence,
        'z_score': intelligence.z_score,
        'top_features': top_features,
        'explanation': explanation_obj.summary if explanation_obj else "",
        'raw_features': features,
    }


# =============================================================================
# SIMULATION
# =============================================================================
def run_intelligence_simulation(
    lstm_vae,
    drift_engine,
    zone_classifier,
    attributor,
    baseline_means,
    baseline_stds,
    drift_start: int = 100,
    drift_rate: float = 0.02,
):
    """Run simulation with synthetic data."""
    
    # Generate normal and drift data
    n_features = len(BEHAVIORAL_FEATURES)
    normal_data = create_synthetic_normal_data(num_samples=drift_start, feature_dim=n_features)
    drift_data = create_synthetic_drift_data(num_samples=200, feature_dim=n_features, drift_rate=drift_rate)
    
    all_data = np.vstack([normal_data, drift_data])
    
    # Process each frame
    results = []
    feature_buffer = []
    
    drift_engine.reset()
    zone_classifier.reset()
    
    drift_detected = False
    drift_onset_frame = None
    
    for i, features in enumerate(all_data):
        result = process_frame(
            features=features,
            frame_index=i,
            lstm_vae=lstm_vae,
            drift_engine=drift_engine,
            zone_classifier=zone_classifier,
            attributor=attributor,
            feature_buffer=feature_buffer,
            baseline_means=baseline_means,
        )
        
        result['frame'] = i
        results.append(result)
        
        # Track drift onset (first time we leave NORMAL zone)
        if not drift_detected and result['zone'] != RiskZone.NORMAL:
            drift_detected = True
            drift_onset_frame = i
    
    return results, drift_onset_frame


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_tdi_timeline(history: dict, drift_start: int) -> go.Figure:
    """Create Threat Deviation Index timeline chart."""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.85, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    
    if not history['tdi']:
        return fig
    
    frames = list(range(len(history['tdi'])))
    tdi_values = history['tdi']
    
    # Zone bands
    zone_bands = [
        (0, 25, 'rgba(34, 197, 94, 0.1)', 'Normal'),
        (25, 50, 'rgba(234, 179, 8, 0.1)', 'Watch'),
        (50, 75, 'rgba(249, 115, 22, 0.1)', 'Warning'),
        (75, 100, 'rgba(239, 68, 68, 0.1)', 'Critical'),
    ]
    
    for y0, y1, color, name in zone_bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, row=1, col=1)
    
    # TDI line
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tdi_values,
            mode='lines',
            name='Threat Deviation Index',
            line=dict(color='#22c55e', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.1)',
            hovertemplate='<b>Frame %{x}</b><br>TDI: %{y:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Threshold lines
    thresholds = [(25, '#22c55e'), (50, '#eab308'), (75, '#f97316')]
    for thresh, color in thresholds:
        fig.add_hline(y=thresh, line_dash="dot", line_color=color, opacity=0.4, row=1, col=1)
    
    # Drift start marker
    fig.add_vline(
        x=drift_start,
        line_dash="dash",
        line_color="#ef4444",
        opacity=0.6,
        annotation_text="Drift Onset",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color="#94a3b8",
        row=1, col=1
    )
    
    # Zone timeline bar
    if history['zones']:
        zone_colors_map = {
            'NORMAL': '#22c55e',
            'WATCH': '#eab308',
            'WARNING': '#f97316',
            'CRITICAL': '#ef4444',
        }
        zone_colors_list = [zone_colors_map.get(z, '#64748b') for z in history['zones']]
        
        fig.add_trace(
            go.Bar(
                x=frames,
                y=[1] * len(frames),
                marker_color=zone_colors_list,
                marker_line_width=0,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=400,
        margin=dict(l=60, r=40, t=20, b=40),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 13, 18, 0.5)',
        font=dict(color='#94a3b8', family='Inter'),
        hovermode='x unified',
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.03)')
    
    fig.update_yaxes(title_text="TDI", title_font_size=11, range=[0, 100], row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(title_text="Frame", title_font_size=11, row=2, col=1)
    
    return fig


def get_zone_info(zone_name: str) -> dict:
    """Get zone display information."""
    zones = {
        'NORMAL': {
            'color': 'var(--zone-normal)',
            'hex': '#22c55e',
            'desc': 'System operating within learned behavioral baseline. No action required.',
            'action': 'Continue standard monitoring',
        },
        'WATCH': {
            'color': 'var(--zone-watch)',
            'hex': '#eab308',
            'desc': 'Early-stage deviation detected. Behavior diverging from baseline.',
            'action': 'Increase observation frequency',
        },
        'WARNING': {
            'color': 'var(--zone-warning)',
            'hex': '#f97316',
            'desc': 'Significant behavioral drift in progress. Intervention may be needed.',
            'action': 'Prepare response protocols',
        },
        'CRITICAL': {
            'color': 'var(--zone-critical)',
            'hex': '#ef4444',
            'desc': 'Critical deviation threshold exceeded. Immediate attention required.',
            'action': 'Execute response protocol',
        },
    }
    return zones.get(zone_name, zones['NORMAL'])


def get_trend_display(trend_name: str) -> dict:
    """Get trend display information."""
    trends = {
        'RISING': {'arrow': '↑', 'class': 'rising', 'text': 'Deviation increasing'},
        'STABLE': {'arrow': '→', 'class': 'stable', 'text': 'Deviation holding steady'},
        'FALLING': {'arrow': '↓', 'class': 'falling', 'text': 'Returning to baseline'},
    }
    return trends.get(trend_name, trends['STABLE'])


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    init_session_state()
    
    # Initialize intelligence system
    (lstm_vae, drift_engine, zone_classifier, attributor, 
     feedback_system, baseline_means, baseline_stds, 
     feature_extractor) = initialize_intelligence_system()
    
    # Store in session state
    st.session_state.lstm_vae = lstm_vae
    st.session_state.drift_engine = drift_engine
    st.session_state.zone_classifier = zone_classifier
    st.session_state.attributor = attributor
    st.session_state.feedback_system = feedback_system
    st.session_state.baseline_means = baseline_means
    st.session_state.baseline_stds = baseline_stds
    
    # Determine current status
    if st.session_state.history.get('tdi'):
        latest_zone = st.session_state.history['zones'][-1]
        status_class = latest_zone.lower()
        status_text = latest_zone
    else:
        status_class = 'normal'
        status_text = 'STANDBY'
    
    # =========================================================================
    # COMMAND BAR
    # =========================================================================
    st.markdown(f"""
    <div class="command-bar">
        <div class="system-id">
            <div class="system-logo">🛡️</div>
            <div>
                <div class="system-name">NOISE FLOOR</div>
                <div class="system-subtitle">Defense Intelligence System</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 24px;">
            <div style="text-align: right;">
                <div style="font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 1px;">Session</div>
                <div style="font-size: 0.85rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">{st.session_state.session_id}</div>
            </div>
            <div class="status-badge {status_class}">
                <span style="font-size: 8px;">●</span>
                {status_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # SIDEBAR - CONTROL PANEL
    # =========================================================================
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 16px 0;">
            <div style="font-size: 0.65rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 1.5px;">Control Panel</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Mode Selection
        st.markdown("""<div class="sidebar-section">
            <div class="sidebar-section-title">Data Mode</div>
        </div>""", unsafe_allow_html=True)
        
        data_mode = st.radio(
            "Select Data Source",
            options=["synthetic", "real_video"],
            format_func=lambda x: "🔬 Synthetic Data" if x == "synthetic" else "📹 Real Video",
            help="Synthetic: controlled testing | Real: UCSD dataset proxy",
            key="data_mode_selector"
        )
        
        if data_mode == "real_video":
            st.markdown("""
            <div style="background: rgba(234, 179, 8, 0.1); border: 1px solid rgba(234, 179, 8, 0.3); 
                        border-radius: 8px; padding: 10px; margin: 10px 0; font-size: 0.75rem;">
                <strong style="color: #eab308;">📍 UCSD Dataset Mode</strong><br>
                <span style="color: #94a3b8;">Public surveillance dataset used as proxy for border CCTV footage.</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 16px 0;'>", unsafe_allow_html=True)
        
        st.markdown("""<div class="sidebar-section">
            <div class="sidebar-section-title">Simulation Parameters</div>
        </div>""", unsafe_allow_html=True)
        
        drift_start = st.slider(
            "Drift Onset (Frame)", 
            50, 200, 100,
            help="Frame where behavioral drift begins"
        )
        
        drift_rate = st.slider(
            "Drift Intensity", 
            0.01, 0.05, 0.02, 0.005,
            help="Rate of behavioral change"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Zone Thresholds
        st.markdown("""<div class="sidebar-section">
            <div class="sidebar-section-title">Zone Thresholds (TDI)</div>
        </div>""", unsafe_allow_html=True)
        
        watch_thresh = st.slider("Watch Zone", 15, 40, 20)
        warning_thresh = st.slider("Warning Zone", 35, 65, 40)
        critical_thresh = st.slider("Critical Zone", 60, 90, 60)
        
        # Update zone classifier TDI thresholds (not self.thresholds which is AdaptiveThresholds dataclass)
        zone_classifier.tdi_thresholds = {
            'normal': watch_thresh,
            'watch': warning_thresh,
            'warning': critical_thresh,
        }
        
        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 16px 0;'>", unsafe_allow_html=True)
        
        # Baseline Status
        st.markdown("""<div class="sidebar-section">
            <div class="sidebar-section-title">Baseline Status</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); 
                    border-radius: 8px; padding: 10px; font-size: 0.75rem;">
            <strong style="color: #22c55e;">🔒 BASELINE FROZEN</strong><br>
            <span style="color: #94a3b8;">Human-gated adaptation only</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # System Info
        st.markdown("""<div class="sidebar-section">
            <div class="sidebar-section-title">System Info</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="font-size: 0.7rem; color: #64748b; line-height: 1.6;">
            <strong>TRL-4:</strong> Lab-validated prototype<br>
            <strong>Mode:</strong> Decision-support intelligence<br>
            <strong>Philosophy:</strong> AI assists, doesn't replace<br>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # PRIMARY INTELLIGENCE DISPLAY
    # =========================================================================
    
    if st.session_state.history.get('tdi'):
        # We have data - show intelligence display
        latest_idx = -1
        latest_tdi = st.session_state.history['tdi'][latest_idx]
        latest_zone = st.session_state.history['zones'][latest_idx]
        latest_trend = st.session_state.history['trends'][latest_idx]
        latest_confidence = st.session_state.history['confidences'][latest_idx]
        
        zone_info = get_zone_info(latest_zone)
        trend_info = get_trend_display(latest_trend)
        
        # Determine TDI color class
        if latest_tdi < 25:
            tdi_class = 'normal'
        elif latest_tdi < 50:
            tdi_class = 'watch'
        elif latest_tdi < 75:
            tdi_class = 'warning'
        else:
            tdi_class = 'critical'
        
        # Top row: TDI, Zone, Trend, Confidence
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="tdi-card">
                <div class="tdi-label">Threat Deviation Index</div>
                <div class="tdi-value {tdi_class}">{latest_tdi:.0f}</div>
                <div class="tdi-scale">Scale: 0-100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="zone-card">
                <div class="zone-indicator">
                    <div class="zone-dot {latest_zone.lower()}"></div>
                    <div class="zone-name">{latest_zone}</div>
                </div>
                <div class="zone-desc">{zone_info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="trend-card">
                <div class="trend-label">Drift Trend</div>
                <div class="trend-arrow {trend_info['class']}">{trend_info['arrow']}</div>
                <div class="trend-text">{trend_info['text']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="confidence-card">
                <div class="trend-label">Confidence</div>
                <div class="confidence-value">{latest_confidence*100:.0f}%</div>
                <div class="trend-text">Assessment reliability</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row: Top Features, Drift Onset, Operator Feedback
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">📊</span>
            <span class="section-title">Attribution & Response</span>
        </div>
        """, unsafe_allow_html=True)
        
        col_feat, col_onset, col_feedback = st.columns([2, 1, 1])
        
        with col_feat:
            # Top contributing features
            st.markdown("""
            <div class="feature-card">
                <div class="feature-header">Top Contributing Features</div>
            """, unsafe_allow_html=True)
            
            if 'top_features' in st.session_state.history and st.session_state.history['top_features']:
                top_features = st.session_state.history['top_features'][latest_idx]
                for feat in top_features[:5]:
                    feat_name = feat.get('name', 'Unknown')
                    feat_score = feat.get('score', 0)
                    # Scale bar width based on z-score (typically 0-5 range)
                    bar_width = min(100, abs(feat_score) * 20)
                    bar_color = zone_info['hex']
                    
                    st.markdown(f"""
                    <div class="feature-item">
                        <span class="feature-name">{feat_name}</span>
                        <div class="feature-bar-container">
                            <div class="feature-bar" style="width: {bar_width}%; background: {bar_color};"></div>
                        </div>
                        <span class="feature-score">{feat_score:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="color: #64748b; font-size: 0.85rem;">No feature data available</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_onset:
            onset_frame = st.session_state.drift_onset_frame
            if onset_frame:
                elapsed = len(st.session_state.history['tdi']) - onset_frame
                st.markdown(f"""
                <div class="onset-card">
                    <div class="onset-label">Drift Onset</div>
                    <div class="onset-value">Frame {onset_frame}</div>
                    <div class="onset-elapsed">{elapsed} frames ago</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="onset-card">
                    <div class="onset-label">Drift Onset</div>
                    <div class="onset-value" style="color: #22c55e;">—</div>
                    <div class="onset-elapsed">No drift detected</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_feedback:
            st.markdown("""
            <div class="feedback-panel">
                <div class="feedback-title">Operator Response</div>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✓ Confirm", key="confirm_btn", use_container_width=True):
                    st.success("Drift confirmed")
            with col_b:
                if st.button("✗ Benign", key="benign_btn", use_container_width=True):
                    st.info("Marked as benign")
        
        # Explanation Card
        if 'explanations' in st.session_state.history and st.session_state.history['explanations']:
            explanation = st.session_state.history['explanations'][latest_idx]
            if explanation:
                st.markdown(f"""
                <div class="explanation-card">
                    <div class="explanation-header">
                        <span class="explanation-icon">🤖</span>
                        <span class="explanation-title">AI Explanation</span>
                    </div>
                    <div class="explanation-text">{explanation}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # TDI Timeline
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">📈</span>
            <span class="section-title">Threat Deviation Timeline</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_tdi_timeline(st.session_state.history, drift_start)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection Metrics
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">📋</span>
            <span class="section-title">Detection Performance</span>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        # Calculate metrics
        tdi_values = st.session_state.history['tdi']
        zones = st.session_state.history['zones']
        
        # False positives (non-normal before drift_start)
        false_positives = sum(1 for i in range(min(drift_start, len(zones))) if zones[i] != 'NORMAL')
        fp_rate = (false_positives / drift_start * 100) if drift_start > 0 else 0
        
        # Detection delay
        onset = st.session_state.drift_onset_frame
        detection_delay = (onset - drift_start) if onset and onset >= drift_start else None
        
        with col_m1:
            st.metric("Detection Delay", f"{detection_delay if detection_delay is not None else '—'} frames")
        
        with col_m2:
            st.metric("False Positive Rate", f"{fp_rate:.1f}%")
        
        with col_m3:
            st.metric("Peak TDI", f"{max(tdi_values):.1f}")
        
        with col_m4:
            st.metric("Frames Analyzed", len(tdi_values))
        
        # Reset button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Session", use_container_width=False):
            st.session_state.history = {
                'tdi': [],
                'zones': [],
                'trends': [],
                'confidences': [],
                'timestamps': [],
                'features': [],
                'explanations': [],
                'top_features': [],
            }
            st.session_state.drift_onset_frame = None
            st.rerun()
    
    else:
        # No data yet - show start screen
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 20px;">🎯</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #e2e8f0; margin-bottom: 12px;">
                Initialize Intelligence System
            </div>
            <div style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; margin: 0 auto 32px auto; line-height: 1.6;">
                Begin behavioral drift analysis. The system will process simulated surveillance data, 
                learning normal patterns and detecting gradual deviations that may indicate emerging threats.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_spacer1, col_btn, col_spacer2 = st.columns([1, 1, 1])
        with col_btn:
            if st.button("▶ Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Running intelligence analysis..."):
                    results, drift_onset_frame = run_intelligence_simulation(
                        lstm_vae=lstm_vae,
                        drift_engine=drift_engine,
                        zone_classifier=zone_classifier,
                        attributor=attributor,
                        baseline_means=baseline_means,
                        baseline_stds=baseline_stds,
                        drift_start=drift_start,
                        drift_rate=drift_rate,
                    )
                    
                    # Store results in session state
                    st.session_state.history = {
                        'tdi': [r['tdi'] for r in results],
                        'zones': [r['zone_name'] for r in results],
                        'trends': [r['trend_name'] for r in results],
                        'confidences': [r['confidence'] for r in results],
                        'timestamps': [r['frame'] for r in results],
                        'features': [r['raw_features'] for r in results],
                        'explanations': [r['explanation'] for r in results],
                        'top_features': [r['top_features'] for r in results],
                    }
                    st.session_state.drift_onset_frame = drift_onset_frame
                    
                    time.sleep(0.3)
                st.rerun()
        
        # System architecture info
        st.markdown("""
        <div class="section-header" style="margin-top: 48px;">
            <span class="section-icon">🔧</span>
            <span class="section-title">System Architecture</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px;">
            <div class="zone-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">👁️</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #e2e8f0;">Feature Extraction</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">24 behavioral features</div>
            </div>
            <div class="zone-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🧠</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #e2e8f0;">LSTM-VAE</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Temporal normality learning</div>
            </div>
            <div class="zone-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #e2e8f0;">Drift Intelligence</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">KL divergence, EWMA, Z-scores</div>
            </div>
            <div class="zone-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🛡️</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #e2e8f0;">Risk Zones</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">4-tier graduated response</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
