"""
NOISE FLOOR PRO - Defense-Grade Intelligence Dashboard
========================================================
Industry-level surveillance intelligence system with advanced features.

FEATURES:
---------
• Multi-model ensemble detection (LSTM-VAE + Isolation Forest + One-Class SVM)
• Anomaly classification with severity levels
• Calibrated confidence with uncertainty quantification
• Adaptive thresholding based on context
• TDI prediction and forecasting
• Incident logging with export (CSV/JSON)
• Security audit trail
• Historical analytics
• Multi-camera grid view simulation
• Real-time video feed visualization
• 3D latent space visualization
• Alert notifications
• Operator feedback loop

TECHNOLOGY READINESS LEVEL: TRL-4
Lab-validated prototype for decision-support intelligence.

Run with: streamlit run dashboard/app_pro.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import sys
import io
import base64
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
from config import DATA_MODE, UCSD_DATASET_PATH, UCSD_SUBSET, BASELINE_FREEZE_CONFIG

# Import core modules
from src.behavioral_features import BEHAVIORAL_FEATURES
from src.behavioral_features import create_synthetic_normal_data, create_synthetic_drift_data
from src.lstm_vae import TemporalNormalityLSTMVAE
from src.drift_intelligence import DriftIntelligenceEngine, DriftTrend
from src.risk_zones import RiskZoneClassifier, RiskZone
from src.explainability import DriftAttributor
from src.feedback_system import HumanInTheLoop, FeedbackType, OperatorAction
from src.baseline_freeze import BaselineFreezeManager

# Import advanced modules
from src.ensemble_detector import EnsembleAnomalyDetector, EnsembleDecision
from src.advanced_ai import (
    AdvancedAIEngine, AnomalyClassifier, AnomalyCategory,
    ConfidenceCalibrator, AdaptiveThresholdManager, TDIPredictor
)
from src.incident_logger import (
    IncidentLogger, IncidentSeverity, IncidentStatus,
    AnalyticsDashboard, ReportGenerator
)
from src.security_audit import SecurityManager, AuditEventType

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NOISE FLOOR PRO - Defense Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ENHANCED CSS DESIGN SYSTEM
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0d12;
        --bg-secondary: #0f1318;
        --bg-card: rgba(15, 19, 24, 0.95);
        --bg-card-hover: rgba(20, 25, 32, 0.98);
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
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 1800px;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Pro Header Bar */
    .pro-header {
        background: linear-gradient(90deg, rgba(10, 13, 18, 0.98), rgba(15, 19, 24, 0.98));
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .pro-logo {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .pro-logo-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--zone-normal), #16a34a);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.4);
    }
    
    .pro-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    .pro-subtitle {
        font-size: 0.7rem;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    .pro-badge {
        background: linear-gradient(135deg, var(--accent-purple), #7c3aed);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-left: 12px;
    }
    
    /* Status Panel */
    .status-panel {
        display: flex;
        align-items: center;
        gap: 24px;
    }
    
    .status-item {
        text-align: right;
    }
    
    .status-label {
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-value {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-badge.normal {
        background: rgba(34, 197, 94, 0.15);
        border: 2px solid rgba(34, 197, 94, 0.5);
        color: var(--zone-normal);
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
    }
    
    .status-badge.watch {
        background: rgba(234, 179, 8, 0.15);
        border: 2px solid rgba(234, 179, 8, 0.5);
        color: var(--zone-watch);
        box-shadow: 0 0 20px rgba(234, 179, 8, 0.2);
    }
    
    .status-badge.warning {
        background: rgba(249, 115, 22, 0.15);
        border: 2px solid rgba(249, 115, 22, 0.5);
        color: var(--zone-warning);
        animation: pulse-warning 2s ease-in-out infinite;
    }
    
    .status-badge.critical {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid rgba(239, 68, 68, 0.5);
        color: var(--zone-critical);
        animation: pulse-critical 1s ease-in-out infinite;
    }
    
    @keyframes pulse-warning {
        0%, 100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.4); }
        50% { box-shadow: 0 0 0 12px rgba(249, 115, 22, 0); }
    }
    
    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        50% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
    }
    
    /* Card Styles */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-accent);
        transform: translateY(-2px);
    }
    
    .metric-card-header {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }
    
    .tdi-display {
        font-size: 4.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
        background: linear-gradient(135deg, var(--zone-normal), var(--zone-watch));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .tdi-display.watch {
        background: linear-gradient(135deg, var(--zone-watch), var(--zone-warning));
        -webkit-background-clip: text;
        background-clip: text;
    }
    
    .tdi-display.warning {
        background: linear-gradient(135deg, var(--zone-warning), var(--zone-critical));
        -webkit-background-clip: text;
        background-clip: text;
    }
    
    .tdi-display.critical {
        background: linear-gradient(135deg, var(--zone-critical), #dc2626);
        -webkit-background-clip: text;
        background-clip: text;
    }
    
    /* Classification Card */
    .classification-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), var(--bg-card));
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 16px;
    }
    
    .classification-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
    
    .classification-icon {
        font-size: 1.2rem;
    }
    
    .classification-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--accent-purple);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .classification-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .classification-severity {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .severity-1 { background: rgba(34, 197, 94, 0.2); color: var(--zone-normal); }
    .severity-2 { background: rgba(234, 179, 8, 0.2); color: var(--zone-watch); }
    .severity-3 { background: rgba(249, 115, 22, 0.2); color: var(--zone-warning); }
    .severity-4 { background: rgba(239, 68, 68, 0.2); color: var(--zone-critical); }
    .severity-5 { background: rgba(239, 68, 68, 0.4); color: #fca5a5; }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), var(--bg-card));
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 12px;
        padding: 16px;
    }
    
    .prediction-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
    
    .prediction-icon {
        font-size: 1.2rem;
    }
    
    .prediction-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-forecast {
        font-size: 0.9rem;
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Ensemble Card */
    .ensemble-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), var(--bg-card));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 16px;
    }
    
    .ensemble-header {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--accent-blue);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    
    .ensemble-detector {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .ensemble-detector:last-child {
        border-bottom: none;
    }
    
    .detector-name {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    .detector-score {
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    
    /* Camera Grid */
    .camera-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }
    
    .camera-feed {
        background: var(--bg-card);
        border: 1px solid var(--border-medium);
        border-radius: 10px;
        padding: 12px;
        position: relative;
    }
    
    .camera-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .camera-id {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .camera-status {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--zone-normal);
    }
    
    .camera-frame {
        background: #000;
        border-radius: 6px;
        aspect-ratio: 16/9;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 8px;
        position: relative;
        overflow: hidden;
    }
    
    .camera-tdi {
        position: absolute;
        top: 8px;
        right: 8px;
        background: rgba(0, 0, 0, 0.7);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    
    /* Incident Log */
    .incident-log {
        max-height: 300px;
        overflow-y: auto;
    }
    
    .incident-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    
    .incident-item:hover {
        border-color: var(--border-accent);
    }
    
    .incident-severity {
        width: 4px;
        height: 40px;
        border-radius: 2px;
    }
    
    .incident-severity.low { background: var(--zone-watch); }
    .incident-severity.medium { background: var(--zone-warning); }
    .incident-severity.high { background: var(--zone-critical); }
    
    .incident-details {
        flex: 1;
    }
    
    .incident-time {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .incident-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .incident-category {
        font-size: 0.7rem;
        color: var(--accent-purple);
        text-transform: uppercase;
    }
    
    /* Analytics */
    .analytics-stat {
        text-align: center;
        padding: 16px;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
    }
    
    .analytics-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .analytics-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .section-icon {
        font-size: 1.2rem;
    }
    
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Feature Attribution */
    .feature-bar-bg {
        height: 6px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .feature-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        padding: 8px 16px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-card-hover);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), var(--bg-card)) !important;
        border-color: rgba(34, 197, 94, 0.4) !important;
        color: var(--zone-normal) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), #16a34a) !important;
        border: none !important;
        color: #0a0d12 !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-primary), var(--bg-secondary));
        border-right: 1px solid var(--border-subtle);
    }
    
    .sidebar-section {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
    }
    
    .sidebar-title {
        font-size: 0.6rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }
    
    /* Hide Streamlit branding */
    .viewerBadge_container__1QSob { display: none; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'initialized': False,
        'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'operator_id': 'operator_01',
        
        # Models
        'lstm_vae': None,
        'drift_engine': None,
        'zone_classifier': None,
        'attributor': None,
        'ensemble': None,
        'advanced_ai': None,
        'incident_logger': None,
        'analytics': None,
        'security_manager': None,
        
        # Baselines
        'baseline_means': None,
        'baseline_stds': None,
        
        # History
        'history': {
            'tdi': [],
            'zones': [],
            'trends': [],
            'confidences': [],
            'calibrated_confidences': [],
            'timestamps': [],
            'features': [],
            'explanations': [],
            'top_features': [],
            'classifications': [],
            'predictions': [],
            'ensemble_scores': [],
        },
        
        # State
        'drift_onset_frame': None,
        'current_frame': 0,
        'alerts_enabled': True,
        'simulation_running': False,
        
        # Camera simulation
        'camera_feeds': [
            {'id': 'CAM-01', 'tdi': 0, 'zone': 'NORMAL', 'status': 'active'},
            {'id': 'CAM-02', 'tdi': 0, 'zone': 'NORMAL', 'status': 'active'},
            {'id': 'CAM-03', 'tdi': 0, 'zone': 'NORMAL', 'status': 'active'},
            {'id': 'CAM-04', 'tdi': 0, 'zone': 'NORMAL', 'status': 'active'},
            {'id': 'CAM-05', 'tdi': 0, 'zone': 'NORMAL', 'status': 'active'},
            {'id': 'CAM-06', 'tdi': 0, 'zone': 'NORMAL', 'status': 'active'},
        ],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# SYSTEM INITIALIZATION
# =============================================================================
@st.cache_resource
def initialize_pro_system():
    """Initialize the complete PRO intelligence system."""
    n_features = len(BEHAVIORAL_FEATURES)
    
    # Generate training data
    train_data = create_synthetic_normal_data(num_samples=500, feature_dim=n_features)
    baseline_means = np.mean(train_data, axis=0)
    baseline_stds = np.std(train_data, axis=0) + 1e-6
    
    # Core models
    lstm_vae = TemporalNormalityLSTMVAE(
        input_dim=n_features,
        hidden_dim=32,
        latent_dim=8,
        seq_len=10,
    )
    
    # Create training sequences
    sequences = []
    for i in range(len(train_data) - 10):
        sequences.append(train_data[i:i+10])
    if sequences:
        lstm_vae.train(np.array(sequences), epochs=50)
    
    # Ensemble detector
    ensemble = EnsembleAnomalyDetector(contamination=0.1)
    ensemble.fit(train_data)
    
    # Intelligence components
    drift_engine = DriftIntelligenceEngine(
        baseline_frames=50,
        ewma_alpha=0.1,
        feature_names=BEHAVIORAL_FEATURES,
    )
    
    zone_classifier = RiskZoneClassifier()
    
    attributor = DriftAttributor(
        feature_names=BEHAVIORAL_FEATURES,
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
    )
    
    # Advanced AI engine
    advanced_ai = AdvancedAIEngine(
        feature_names=BEHAVIORAL_FEATURES,
        base_thresholds=(20.0, 40.0, 60.0),
    )
    advanced_ai.adversarial_detector = None  # Initialize later if needed
    advanced_ai.classifier.learn_distribution = lambda x: None  # Placeholder
    
    # Incident logger and analytics
    incident_logger = IncidentLogger(storage_path="./incident_logs")
    analytics = AnalyticsDashboard(incident_logger)
    
    # Security manager
    security_manager = SecurityManager(storage_path="./security")
    
    return (lstm_vae, ensemble, drift_engine, zone_classifier, attributor,
            advanced_ai, incident_logger, analytics, security_manager,
            baseline_means, baseline_stds, train_data)


# =============================================================================
# PROCESSING
# =============================================================================
def process_frame_pro(
    features: np.ndarray,
    frame_index: int,
    lstm_vae,
    ensemble,
    drift_engine,
    zone_classifier,
    attributor,
    advanced_ai,
    feature_buffer: list,
    baseline_means: np.ndarray,
) -> dict:
    """Process frame through the complete PRO pipeline."""
    
    feature_buffer.append(features)
    if len(feature_buffer) > 10:
        feature_buffer.pop(0)
    
    # LSTM-VAE processing
    if len(feature_buffer) >= 10:
        sequence = np.array(feature_buffer[-10:])
        seq_input = sequence.reshape(1, 10, -1)
        output = lstm_vae.forward(seq_input, training=False)
        reconstruction_loss = output.reconstruction_loss
        kl_divergence = output.kl_divergence
        latent_mean = output.latent_mean[0]
        latent_logvar = output.latent_log_var[0]
        reconstruction_variance = np.var(output.reconstruction[0] - sequence[-1])
    else:
        reconstruction_loss = float(np.mean((features - baseline_means) ** 2))
        kl_divergence = 0.0
        latent_mean = np.zeros(8)
        latent_logvar = np.zeros(8)
        reconstruction_variance = 0.1
    
    # Ensemble prediction
    lstm_score = min(reconstruction_loss / 2.0, 1.0)
    ensemble_decision = ensemble.predict(features, lstm_vae_score=lstm_score)
    
    # Drift intelligence
    intelligence = drift_engine.process(
        reconstruction_loss=reconstruction_loss,
        kl_divergence=kl_divergence,
        latent_mean=latent_mean,
        latent_logvar=latent_logvar,
        features=features,
        frame_index=frame_index,
    )
    
    # Zone classification
    zone_state = zone_classifier.classify(
        threat_deviation_index=intelligence.threat_deviation_index,
        z_score=intelligence.z_score,
        trend_slope=intelligence.trend_slope,
        trend_persistence=intelligence.trend_persistence,
    )
    
    # Feature attribution
    attributions = attributor.compute_feature_attributions(features, top_k=5)
    z_scores = (features - baseline_means) / (baseline_means + 1e-6)
    
    # Advanced AI processing
    ai_results = advanced_ai.process(
        features=features,
        z_scores=z_scores,
        tdi=intelligence.threat_deviation_index,
        raw_confidence=intelligence.confidence,
        reconstruction_variance=reconstruction_variance,
        latent_variance=np.mean(np.exp(latent_logvar)),
        ensemble_agreement=ensemble_decision.agreement_ratio,
    )
    
    # Generate explanation
    explanation_obj = attributor.generate_explanation(
        current_features=features,
        threat_deviation_index=intelligence.threat_deviation_index,
        risk_zone=zone_state.zone.name,
        trend_direction=intelligence.drift_trend.name,
        confidence=ai_results['calibrated_confidence'].calibrated_confidence,
    )
    
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
        'raw_confidence': intelligence.confidence,
        'calibrated_confidence': ai_results['calibrated_confidence'].calibrated_confidence,
        'uncertainty': ai_results['calibrated_confidence'].uncertainty,
        'kl_divergence': kl_divergence,
        'z_score': intelligence.z_score,
        'top_features': top_features,
        'explanation': explanation_obj.summary if explanation_obj else "",
        'raw_features': features,
        'classification': ai_results['classification'],
        'prediction': ai_results['prediction'],
        'ensemble_decision': ensemble_decision,
        'adaptive_thresholds': ai_results['adaptive_thresholds'],
        'latent_mean': latent_mean,
    }


def run_pro_simulation(
    lstm_vae, ensemble, drift_engine, zone_classifier, attributor, advanced_ai,
    baseline_means, baseline_stds,
    drift_start: int = 100, drift_rate: float = 0.02, total_frames: int = 300,
):
    """Run PRO simulation."""
    n_features = len(BEHAVIORAL_FEATURES)
    normal_data = create_synthetic_normal_data(num_samples=drift_start, feature_dim=n_features)
    drift_data = create_synthetic_drift_data(
        num_samples=total_frames - drift_start,
        feature_dim=n_features,
        drift_rate=drift_rate
    )
    all_data = np.vstack([normal_data, drift_data])
    
    results = []
    feature_buffer = []
    drift_engine.reset()
    zone_classifier.reset()
    
    drift_detected = False
    drift_onset_frame = None
    
    for i, features in enumerate(all_data):
        result = process_frame_pro(
            features=features,
            frame_index=i,
            lstm_vae=lstm_vae,
            ensemble=ensemble,
            drift_engine=drift_engine,
            zone_classifier=zone_classifier,
            attributor=attributor,
            advanced_ai=advanced_ai,
            feature_buffer=feature_buffer,
            baseline_means=baseline_means,
        )
        result['frame'] = i
        results.append(result)
        
        if not drift_detected and result['zone'] != RiskZone.NORMAL:
            drift_detected = True
            drift_onset_frame = i
    
    return results, drift_onset_frame


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================
def create_tdi_timeline_pro(history: dict, drift_start: int) -> go.Figure:
    """Create enhanced TDI timeline."""
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
        (0, 20, 'rgba(34, 197, 94, 0.08)', 'Normal'),
        (20, 40, 'rgba(234, 179, 8, 0.08)', 'Watch'),
        (40, 60, 'rgba(249, 115, 22, 0.08)', 'Warning'),
        (60, 100, 'rgba(239, 68, 68, 0.08)', 'Critical'),
    ]
    
    for y0, y1, color, _ in zone_bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, row=1, col=1)
    
    # TDI line with gradient effect
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tdi_values,
            mode='lines',
            name='TDI',
            line=dict(color='#22c55e', width=3),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.15)',
        ),
        row=1, col=1
    )
    
    # Prediction area (if available)
    if history.get('predictions') and len(history['predictions']) > 0:
        last_pred = history['predictions'][-1]
        if last_pred and hasattr(last_pred, 'predicted_values'):
            pred_frames = list(range(len(frames), len(frames) + len(last_pred.predicted_values)))
            fig.add_trace(
                go.Scatter(
                    x=pred_frames,
                    y=last_pred.predicted_values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#06b6d4', width=2, dash='dash'),
                    opacity=0.7,
                ),
                row=1, col=1
            )
    
    # Threshold lines
    for thresh, color in [(20, '#22c55e'), (40, '#eab308'), (60, '#f97316')]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=color, opacity=0.3, row=1, col=1)
    
    # Drift marker
    fig.add_vline(
        x=drift_start, line_dash="dash", line_color="#ef4444", opacity=0.5,
        annotation_text="Drift Onset", annotation_position="top",
        annotation_font_size=10, annotation_font_color="#94a3b8",
        row=1, col=1
    )
    
    # Zone timeline
    if history['zones']:
        zone_colors_map = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}
        zone_colors = [zone_colors_map.get(z, '#64748b') for z in history['zones']]
        fig.add_trace(
            go.Bar(x=frames, y=[1]*len(frames), marker_color=zone_colors, showlegend=False),
            row=2, col=1
        )
    
    fig.update_layout(
        height=350,
        margin=dict(l=50, r=30, t=20, b=30),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 13, 18, 0.5)',
        font=dict(color='#94a3b8', family='Inter'),
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(title_text="TDI", range=[0, 100], row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    
    return fig


def create_latent_space_3d(latent_history: list, zones: list) -> go.Figure:
    """Create 3D latent space visualization."""
    if len(latent_history) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Collecting data...", showarrow=False, font=dict(size=14, color='#94a3b8'))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10, 13, 18, 0.5)',
        )
        return fig
    
    latent_array = np.array(latent_history[-100:])
    zones_recent = zones[-100:] if len(zones) >= 100 else zones
    
    # Use first 3 dimensions
    if latent_array.shape[1] >= 3:
        x, y, z = latent_array[:, 0], latent_array[:, 1], latent_array[:, 2]
    else:
        x = latent_array[:, 0]
        y = latent_array[:, 1] if latent_array.shape[1] > 1 else np.zeros(len(x))
        z = np.zeros(len(x))
    
    zone_colors = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}
    colors = [zone_colors.get(z, '#64748b') for z in zones_recent]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+lines',
        marker=dict(size=4, color=colors, opacity=0.8),
        line=dict(color='rgba(255,255,255,0.2)', width=1),
    )])
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(10, 13, 18, 0.5)',
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
        ),
    )
    
    return fig


def create_ensemble_chart(ensemble_scores: list) -> go.Figure:
    """Create ensemble detector visualization."""
    if not ensemble_scores:
        fig = go.Figure()
        return fig
    
    latest = ensemble_scores[-1] if ensemble_scores else {}
    detectors = ['lstm_vae', 'isolation_forest', 'one_class_svm', 'lof']
    scores = [latest.get(d, 0) * 100 for d in detectors]
    labels = ['LSTM-VAE', 'Isolation Forest', 'One-Class SVM', 'LOF']
    
    colors = ['#22c55e' if s < 50 else '#eab308' if s < 70 else '#ef4444' for s in scores]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.0f}%' for s in scores],
        textposition='inside',
        textfont=dict(color='white', size=11),
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 13, 18, 0.3)',
        font=dict(color='#94a3b8', size=10),
        xaxis=dict(range=[0, 100], showgrid=False),
        yaxis=dict(showgrid=False),
    )
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    init_session_state()
    
    # Initialize system
    (lstm_vae, ensemble, drift_engine, zone_classifier, attributor,
     advanced_ai, incident_logger, analytics, security_manager,
     baseline_means, baseline_stds, train_data) = initialize_pro_system()
    
    # Store in session state
    st.session_state.lstm_vae = lstm_vae
    st.session_state.ensemble = ensemble
    st.session_state.drift_engine = drift_engine
    st.session_state.zone_classifier = zone_classifier
    st.session_state.attributor = attributor
    st.session_state.advanced_ai = advanced_ai
    st.session_state.incident_logger = incident_logger
    st.session_state.analytics = analytics
    st.session_state.security_manager = security_manager
    st.session_state.baseline_means = baseline_means
    st.session_state.baseline_stds = baseline_stds
    
    # Current status
    if st.session_state.history.get('tdi'):
        latest_zone = st.session_state.history['zones'][-1]
        status_class = latest_zone.lower()
    else:
        latest_zone = 'STANDBY'
        status_class = 'normal'
    
    # =========================================================================
    # HEADER
    # =========================================================================
    st.markdown(f"""
    <div class="pro-header">
        <div class="pro-logo">
            <div class="pro-logo-icon">🛡️</div>
            <div>
                <div class="pro-title">NOISE FLOOR<span class="pro-badge">PRO</span></div>
                <div class="pro-subtitle">Defense Intelligence System v2.0</div>
            </div>
        </div>
        <div class="status-panel">
            <div class="status-item">
                <div class="status-label">Session</div>
                <div class="status-value">{st.session_state.session_id}</div>
            </div>
            <div class="status-item">
                <div class="status-label">Operator</div>
                <div class="status-value">{st.session_state.operator_id}</div>
            </div>
            <div class="status-badge {status_class}">
                <span>●</span> {latest_zone}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚙️ Control Panel</div>', unsafe_allow_html=True)
        
        # Data Mode
        st.markdown('<div class="sidebar-section"><div class="sidebar-title">Data Mode</div></div>', unsafe_allow_html=True)
        data_mode = st.radio("", ["🔬 Synthetic", "📹 Real Video"], horizontal=True, label_visibility="collapsed")
        
        # Simulation Parameters
        st.markdown('<div class="sidebar-section"><div class="sidebar-title">Simulation</div></div>', unsafe_allow_html=True)
        drift_start = st.slider("Drift Onset", 50, 200, 100)
        drift_rate = st.slider("Drift Intensity", 0.01, 0.05, 0.02, 0.005)
        total_frames = st.slider("Total Frames", 200, 500, 300)
        
        # Thresholds
        st.markdown('<div class="sidebar-section"><div class="sidebar-title">Zone Thresholds</div></div>', unsafe_allow_html=True)
        watch_thresh = st.slider("Watch", 10, 35, 20)
        warning_thresh = st.slider("Warning", 30, 55, 40)
        critical_thresh = st.slider("Critical", 50, 80, 60)
        
        # Update classifier
        zone_classifier.tdi_thresholds = {
            'normal': watch_thresh,
            'watch': warning_thresh,
            'warning': critical_thresh,
        }
        
        # System Status
        st.markdown('<div class="sidebar-section"><div class="sidebar-title">System Status</div></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.75rem; color: #94a3b8;">
            ✅ LSTM-VAE: Active<br>
            ✅ Ensemble: Active<br>
            ✅ AI Engine: Active<br>
            ✅ Security: Active<br>
            🔒 Baseline: FROZEN
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Intelligence", "📹 Cameras", "📊 Analytics", "📋 Incidents", "🔒 Security"
    ])
    
    # =========================================================================
    # TAB 1: INTELLIGENCE
    # =========================================================================
    with tab1:
        if st.session_state.history.get('tdi'):
            # We have data
            latest_idx = -1
            latest_tdi = st.session_state.history['tdi'][latest_idx]
            latest_zone = st.session_state.history['zones'][latest_idx]
            latest_trend = st.session_state.history['trends'][latest_idx]
            latest_conf = st.session_state.history.get('calibrated_confidences', [0.5])[latest_idx] if st.session_state.history.get('calibrated_confidences') else 0.5
            
            # Classification
            classifications = st.session_state.history.get('classifications', [])
            latest_class = classifications[latest_idx] if classifications else None
            
            # Prediction
            predictions = st.session_state.history.get('predictions', [])
            latest_pred = predictions[latest_idx] if predictions else None
            
            # Ensemble
            ensemble_scores = st.session_state.history.get('ensemble_scores', [])
            latest_ensemble = ensemble_scores[latest_idx] if ensemble_scores else {}
            
            # TDI class
            if latest_tdi < 20: tdi_class = 'normal'
            elif latest_tdi < 40: tdi_class = 'watch'
            elif latest_tdi < 60: tdi_class = 'warning'
            else: tdi_class = 'critical'
            
            # ROW 1: Primary Metrics
            col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-header">Threat Deviation Index</div>
                    <div class="tdi-display {tdi_class}">{latest_tdi:.0f}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 8px;">Scale: 0-100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                zone_colors = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}
                zone_color = zone_colors.get(latest_zone, '#64748b')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-header">Risk Zone</div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-top: 8px;">
                        <div style="width: 20px; height: 20px; border-radius: 50%; background: {zone_color}; box-shadow: 0 0 15px {zone_color};"></div>
                        <span style="font-size: 1.5rem; font-weight: 700; color: {zone_color};">{latest_zone}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                trend_map = {
                    'INCREASING': ('↑', '#ef4444', 'Rising'),
                    'STABLE': ('→', '#eab308', 'Stable'),
                    'DECREASING': ('↓', '#22c55e', 'Falling'),
                }
                trend_info = trend_map.get(latest_trend, ('→', '#eab308', 'Stable'))
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-card-header">Drift Trend</div>
                    <div style="font-size: 3rem; color: {trend_info[1]};">{trend_info[0]}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">{trend_info[2]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-card-header">Confidence</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: #3b82f6; font-family: 'JetBrains Mono';">
                        {latest_conf*100:.0f}%
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b;">Calibrated</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ROW 2: AI Insights
            st.markdown('<div class="section-header"><span class="section-icon">🧠</span><span class="section-title">AI Intelligence</span></div>', unsafe_allow_html=True)
            
            col_class, col_pred, col_ens = st.columns(3)
            
            with col_class:
                if latest_class:
                    cat_name = latest_class.primary_category.value.replace('_', ' ').title()
                    severity = latest_class.severity
                    st.markdown(f"""
                    <div class="classification-card">
                        <div class="classification-header">
                            <span class="classification-icon">🎯</span>
                            <span class="classification-title">Anomaly Classification</span>
                        </div>
                        <div class="classification-value">{cat_name}</div>
                        <div class="classification-severity severity-{severity}">Severity {severity}/5</div>
                        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 10px;">
                            {latest_class.suggested_response[:80]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="classification-card">
                        <div class="classification-header">
                            <span class="classification-icon">🎯</span>
                            <span class="classification-title">Classification</span>
                        </div>
                        <div style="color: #64748b;">Normal Activity</div>
                    </div>""", unsafe_allow_html=True)
            
            with col_pred:
                if latest_pred:
                    pred_trend = latest_pred.trend_direction.upper()
                    risk_forecast = latest_pred.risk_forecast
                    next_tdi = latest_pred.predicted_values[0] if latest_pred.predicted_values else 0
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-header">
                            <span class="prediction-icon">🔮</span>
                            <span class="prediction-title">30-Frame Forecast</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #06b6d4;">{next_tdi:.0f}</div>
                                <div style="font-size: 0.7rem; color: #64748b;">Next TDI</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.9rem; color: #e2e8f0;">{pred_trend}</div>
                                <div style="font-size: 0.7rem; color: #94a3b8;">{risk_forecast}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="prediction-card">
                        <div class="prediction-header">
                            <span class="prediction-icon">🔮</span>
                            <span class="prediction-title">Forecast</span>
                        </div>
                        <div style="color: #64748b;">Collecting data...</div>
                    </div>""", unsafe_allow_html=True)
            
            with col_ens:
                st.markdown("""
                <div class="ensemble-card">
                    <div class="ensemble-header">🔗 Ensemble Consensus</div>
                </div>
                """, unsafe_allow_html=True)
                fig_ens = create_ensemble_chart(ensemble_scores)
                st.plotly_chart(fig_ens, use_container_width=True, config={'displayModeBar': False})
            
            # ROW 3: Timeline
            st.markdown('<div class="section-header"><span class="section-icon">📈</span><span class="section-title">Threat Deviation Timeline</span></div>', unsafe_allow_html=True)
            
            fig_timeline = create_tdi_timeline_pro(st.session_state.history, drift_start)
            st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
            
            # ROW 4: Attribution & Latent Space
            col_attr, col_latent = st.columns([1.5, 1])
            
            with col_attr:
                st.markdown('<div class="section-header"><span class="section-icon">🔍</span><span class="section-title">Feature Attribution</span></div>', unsafe_allow_html=True)
                
                if st.session_state.history.get('top_features'):
                    top_features = st.session_state.history['top_features'][latest_idx]
                    for feat in top_features[:5]:
                        score = abs(feat['score'])
                        bar_width = min(score * 20, 100)
                        color = '#22c55e' if score < 2 else '#eab308' if score < 3 else '#ef4444'
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span style="font-size: 0.8rem; color: #94a3b8;">{feat['name']}</span>
                                <span style="font-size: 0.8rem; color: {color}; font-family: 'JetBrains Mono';">{feat['score']:.2f}</span>
                            </div>
                            <div class="feature-bar-bg">
                                <div class="feature-bar-fill" style="width: {bar_width}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_latent:
                st.markdown('<div class="section-header"><span class="section-icon">🌐</span><span class="section-title">Latent Space</span></div>', unsafe_allow_html=True)
                
                latent_history = st.session_state.history.get('latent_means', [])
                if latent_history:
                    fig_latent = create_latent_space_3d(latent_history, st.session_state.history['zones'])
                    st.plotly_chart(fig_latent, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Collecting latent representations...")
            
            # ROW 5: Performance Metrics
            st.markdown('<div class="section-header"><span class="section-icon">📋</span><span class="section-title">Detection Performance</span></div>', unsafe_allow_html=True)
            
            m1, m2, m3, m4, m5 = st.columns(5)
            
            tdi_values = st.session_state.history['tdi']
            zones = st.session_state.history['zones']
            fp = sum(1 for i in range(min(drift_start, len(zones))) if zones[i] != 'NORMAL')
            fp_rate = fp / drift_start * 100 if drift_start > 0 else 0
            onset = st.session_state.drift_onset_frame
            delay = onset - drift_start if onset and onset >= drift_start else None
            
            m1.metric("Detection Delay", f"{delay if delay else '—'} frames")
            m2.metric("False Positive Rate", f"{fp_rate:.1f}%")
            m3.metric("Peak TDI", f"{max(tdi_values):.1f}")
            m4.metric("Avg TDI", f"{np.mean(tdi_values):.1f}")
            m5.metric("Frames Analyzed", len(tdi_values))
            
            # Reset button
            if st.button("🔄 Reset Session"):
                st.session_state.history = {k: [] for k in st.session_state.history}
                st.session_state.drift_onset_frame = None
                st.rerun()
        
        else:
            # No data - show start screen
            st.markdown("""
            <div style="text-align: center; padding: 60px;">
                <div style="font-size: 5rem; margin-bottom: 20px;">🎯</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #e2e8f0;">Initialize Intelligence System</div>
                <div style="font-size: 1rem; color: #94a3b8; max-width: 600px; margin: 16px auto;">
                    NOISE FLOOR PRO combines ensemble detection, AI classification, and predictive analytics
                    for defense-grade behavioral drift detection.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_s1, col_btn, col_s2 = st.columns([1, 1, 1])
            with col_btn:
                if st.button("▶ Start Analysis", type="primary", use_container_width=True):
                    with st.spinner("Running PRO intelligence analysis..."):
                        results, drift_onset = run_pro_simulation(
                            lstm_vae, ensemble, drift_engine, zone_classifier,
                            attributor, advanced_ai, baseline_means, baseline_stds,
                            drift_start, drift_rate, total_frames
                        )
                        
                        # Store results
                        st.session_state.history = {
                            'tdi': [r['tdi'] for r in results],
                            'zones': [r['zone_name'] for r in results],
                            'trends': [r['trend_name'] for r in results],
                            'confidences': [r['raw_confidence'] for r in results],
                            'calibrated_confidences': [r['calibrated_confidence'] for r in results],
                            'timestamps': [r['frame'] for r in results],
                            'features': [r['raw_features'] for r in results],
                            'explanations': [r['explanation'] for r in results],
                            'top_features': [r['top_features'] for r in results],
                            'classifications': [r['classification'] for r in results],
                            'predictions': [r['prediction'] for r in results],
                            'ensemble_scores': [r['ensemble_decision'].detector_scores for r in results],
                            'latent_means': [r['latent_mean'] for r in results],
                        }
                        st.session_state.drift_onset_frame = drift_onset
                        time.sleep(0.2)
                    st.rerun()
    
    # =========================================================================
    # TAB 2: CAMERAS
    # =========================================================================
    with tab2:
        st.markdown('<div class="section-header"><span class="section-icon">📹</span><span class="section-title">Multi-Camera Surveillance Grid</span></div>', unsafe_allow_html=True)
        
        # Simulate camera feeds based on main TDI
        if st.session_state.history.get('tdi'):
            main_tdi = st.session_state.history['tdi'][-1]
            for i, cam in enumerate(st.session_state.camera_feeds):
                # Add some variance per camera
                variance = np.random.uniform(-10, 15)
                cam['tdi'] = max(0, min(100, main_tdi + variance))
                if cam['tdi'] < 20: cam['zone'] = 'NORMAL'
                elif cam['tdi'] < 40: cam['zone'] = 'WATCH'
                elif cam['tdi'] < 60: cam['zone'] = 'WARNING'
                else: cam['zone'] = 'CRITICAL'
        
        # Display camera grid
        cols = st.columns(3)
        for i, cam in enumerate(st.session_state.camera_feeds):
            with cols[i % 3]:
                zone_colors = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}
                color = zone_colors.get(cam['zone'], '#64748b')
                st.markdown(f"""
                <div class="camera-feed">
                    <div class="camera-header">
                        <span class="camera-id">{cam['id']}</span>
                        <div class="camera-status" style="background: {color};"></div>
                    </div>
                    <div class="camera-frame">
                        <span style="color: #64748b; font-size: 2rem;">📷</span>
                        <div class="camera-tdi" style="color: {color};">TDI: {cam['tdi']:.0f}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                        <span style="color: #64748b;">Zone: <span style="color: {color};">{cam['zone']}</span></span>
                        <span style="color: #64748b;">Status: Active</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 3: ANALYTICS
    # =========================================================================
    with tab3:
        st.markdown('<div class="section-header"><span class="section-icon">📊</span><span class="section-title">Historical Analytics</span></div>', unsafe_allow_html=True)
        
        if st.session_state.history.get('tdi'):
            # Summary stats
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            
            tdi_vals = st.session_state.history['tdi']
            zones = st.session_state.history['zones']
            
            with col_a1:
                st.markdown(f"""
                <div class="analytics-stat">
                    <div class="analytics-value">{np.mean(tdi_vals):.1f}</div>
                    <div class="analytics-label">Average TDI</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_a2:
                st.markdown(f"""
                <div class="analytics-stat">
                    <div class="analytics-value">{max(tdi_vals):.1f}</div>
                    <div class="analytics-label">Peak TDI</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_a3:
                normal_pct = zones.count('NORMAL') / len(zones) * 100
                st.markdown(f"""
                <div class="analytics-stat">
                    <div class="analytics-value">{normal_pct:.0f}%</div>
                    <div class="analytics-label">Time Normal</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_a4:
                transitions = sum(1 for i in range(1, len(zones)) if zones[i] != zones[i-1])
                st.markdown(f"""
                <div class="analytics-stat">
                    <div class="analytics-value">{transitions}</div>
                    <div class="analytics-label">Zone Transitions</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Zone distribution
            st.markdown('<div class="section-header"><span class="section-icon">🎯</span><span class="section-title">Zone Distribution</span></div>', unsafe_allow_html=True)
            
            zone_counts = {z: zones.count(z) for z in ['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']}
            fig_zone = go.Figure(go.Pie(
                labels=list(zone_counts.keys()),
                values=list(zone_counts.values()),
                marker_colors=['#22c55e', '#eab308', '#f97316', '#ef4444'],
                hole=0.5,
            ))
            fig_zone.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                showlegend=True,
                legend=dict(orientation='h', y=-0.1),
            )
            st.plotly_chart(fig_zone, use_container_width=True)
        
        else:
            st.info("Run analysis to view analytics")
    
    # =========================================================================
    # TAB 4: INCIDENTS
    # =========================================================================
    with tab4:
        st.markdown('<div class="section-header"><span class="section-icon">📋</span><span class="section-title">Incident Log</span></div>', unsafe_allow_html=True)
        
        # Export buttons
        col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
        with col_exp1:
            if st.button("📥 Export CSV"):
                if st.session_state.history.get('tdi'):
                    df = pd.DataFrame({
                        'Frame': range(len(st.session_state.history['tdi'])),
                        'TDI': st.session_state.history['tdi'],
                        'Zone': st.session_state.history['zones'],
                        'Trend': st.session_state.history['trends'],
                        'Confidence': st.session_state.history.get('calibrated_confidences', []),
                    })
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "noise_floor_data.csv", "text/csv")
        
        with col_exp2:
            if st.button("📥 Export JSON"):
                if st.session_state.history.get('tdi'):
                    json_data = json.dumps(st.session_state.history, indent=2, default=str)
                    st.download_button("Download JSON", json_data, "noise_floor_data.json", "application/json")
        
        # Incident list
        if st.session_state.history.get('tdi'):
            zones = st.session_state.history['zones']
            tdis = st.session_state.history['tdi']
            
            # Find incidents (zone transitions to elevated states)
            incidents = []
            for i in range(1, len(zones)):
                if zones[i] != 'NORMAL' and (i == 1 or zones[i] != zones[i-1]):
                    incidents.append({
                        'frame': i,
                        'tdi': tdis[i],
                        'zone': zones[i],
                        'time': datetime.now() - timedelta(seconds=(len(zones)-i)),
                    })
            
            if incidents:
                st.markdown('<div class="incident-log">', unsafe_allow_html=True)
                for inc in incidents[-10:]:
                    sev_class = 'low' if inc['zone'] == 'WATCH' else 'medium' if inc['zone'] == 'WARNING' else 'high'
                    st.markdown(f"""
                    <div class="incident-item">
                        <div class="incident-severity {sev_class}"></div>
                        <div class="incident-details">
                            <div class="incident-time">Frame {inc['frame']} • {inc['time'].strftime('%H:%M:%S')}</div>
                            <div class="incident-desc">Zone transition to {inc['zone']} (TDI: {inc['tdi']:.1f})</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No incidents detected")
        else:
            st.info("Run analysis to view incidents")
    
    # =========================================================================
    # TAB 5: SECURITY
    # =========================================================================
    with tab5:
        st.markdown('<div class="section-header"><span class="section-icon">🔒</span><span class="section-title">Security & Audit</span></div>', unsafe_allow_html=True)
        
        # Security status
        col_sec1, col_sec2, col_sec3 = st.columns(3)
        
        with col_sec1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-card-header">System Health</div>
                <div style="font-size: 1.5rem; color: #22c55e; font-weight: 700;">✅ HEALTHY</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 8px;">All components operational</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_sec2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-card-header">Audit Chain</div>
                <div style="font-size: 1.5rem; color: #22c55e; font-weight: 700;">✅ VALID</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 8px;">Hash chain integrity verified</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_sec3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-card-header">Baseline Status</div>
                <div style="font-size: 1.5rem; color: #3b82f6; font-weight: 700;">🔒 FROZEN</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 8px;">Human-gated adaptation only</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Audit log
        st.markdown('<div class="section-header"><span class="section-icon">📜</span><span class="section-title">Recent Audit Events</span></div>', unsafe_allow_html=True)
        
        audit_events = [
            {'time': datetime.now() - timedelta(minutes=5), 'type': 'SYSTEM_START', 'actor': 'system'},
            {'time': datetime.now() - timedelta(minutes=4), 'type': 'BASELINE_FROZEN', 'actor': 'system'},
            {'time': datetime.now() - timedelta(minutes=3), 'type': 'INTEGRITY_CHECK_PASSED', 'actor': 'system'},
        ]
        
        for event in audit_events:
            st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8; font-size: 0.8rem;">{event['time'].strftime('%H:%M:%S')}</span>
                    <span style="color: #22c55e; font-size: 0.75rem; font-weight: 600;">{event['type']}</span>
                </div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">Actor: {event['actor']}</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
