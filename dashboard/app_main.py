"""
DRISHTI - Defense-Grade Surveillance Intelligence Dashboard
=================================================================
Real-time behavioral drift detection for border security.

Features:
â€¢ UCSD Dataset Integration with Real Video Frames
â€¢ Multi-Camera Grid View
â€¢ LSTM-VAE Temporal Normality Learning
â€¢ ENSEMBLE DETECTION (Isolation Forest + One-Class SVM + LOF)
â€¢ 4-Tier Risk Zone Classification
â€¢ Explainable AI Attribution
â€¢ Anomaly Classification & Incident Logging
â€¢ 3D Latent Space Visualization
â€¢ Real-time Threat Deviation Index (TDI)
â€¢ TDI Forecasting & Trend Prediction

Run: streamlit run dashboard/app_main.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from collections import deque

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules
from config import DATA_MODE, UCSD_DATASET_PATH, UCSD_SUBSET, DRONE_BIRD_DATASET_PATH
from src.behavioral_features import BEHAVIORAL_FEATURES, create_synthetic_normal_data, create_synthetic_drift_data
from src.lstm_vae import TemporalNormalityLSTMVAE
from src.drift_intelligence import DriftIntelligenceEngine, DriftTrend
from src.risk_zones import RiskZoneClassifier, RiskZone
from src.explainability import DriftAttributor

# Import ensemble and advanced AI
try:
    from src.ensemble_detector import EnsembleAnomalyDetector, EnsembleDecision, DetectorType
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from src.advanced_ai import AnomalyClassifier, AnomalyCategory, ConfidenceCalibrator
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    ADVANCED_AI_AVAILABLE = False

try:
    from src.incident_logger import IncidentLogger, IncidentSeverity, IncidentStatus
    INCIDENT_LOGGER_AVAILABLE = True
except ImportError:
    INCIDENT_LOGGER_AVAILABLE = False

# Try importing video features
try:
    from src.video_features import RealVideoFeatureExtractor, UCSDDatasetLoader
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False

# Try importing drone-bird features
try:
    from src.drone_bird_loader import DroneBirdLoader, DroneBirdFeatureExtractor, create_simulated_sequences
    DRONE_BIRD_AVAILABLE = True
except ImportError:
    DRONE_BIRD_AVAILABLE = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="DRISHTI - Defense Intelligence",
    page_icon="ðŸ‘ï¸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DEFENSE THEME CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-dark: #0a0d12;
        --bg-card: #0f1318;
        --border: rgba(255,255,255,0.08);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --green: #22c55e;
        --yellow: #eab308;
        --orange: #f97316;
        --red: #ef4444;
        --blue: #3b82f6;
        --purple: #8b5cf6;
        --cyan: #06b6d4;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0d12 0%, #0f1318 100%);
    }
    
    .block-container {
        padding: 1rem 2rem;
        max-width: 1800px;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(15, 19, 24, 0.95), rgba(10, 13, 18, 0.98));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 30px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .logo-icon {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow: 0 0 40px rgba(6, 182, 212, 0.3), inset 0 0 20px rgba(6, 182, 212, 0.1);
        border: 2px solid rgba(6, 182, 212, 0.4);
        overflow: hidden;
    }
    
    .logo-icon::before {
        content: '';
        position: absolute;
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #06b6d4, #0891b2);
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.6), inset 0 0 10px rgba(255,255,255,0.2);
    }
    
    .logo-icon::after {
        content: '';
        position: absolute;
        width: 14px;
        height: 14px;
        background: radial-gradient(circle, #0f172a 40%, #06b6d4 100%);
        border-radius: 50%;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.8);
        animation: pulse-eye 2s ease-in-out infinite;
    }
    
    @keyframes pulse-eye {
        0%, 100% { transform: scale(1); box-shadow: 0 0 15px rgba(6, 182, 212, 0.8); }
        50% { transform: scale(1.1); box-shadow: 0 0 25px rgba(6, 182, 212, 1); }
    }
    
    .logo-text h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #ffffff, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .logo-text p {
        margin: 0;
        font-size: 0.7rem;
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 600;
    }
    
    .status-section {
        display: flex;
        align-items: center;
        gap: 30px;
    }
    
    .status-item {
        text-align: center;
    }
    
    .status-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-value {
        font-size: 0.9rem;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .status-badge {
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-badge.normal { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .status-badge.watch { background: rgba(234, 179, 8, 0.15); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.3); }
    .status-badge.warning { background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.3); }
    .status-badge.critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); animation: pulse 1s infinite; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(34, 197, 94, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    
    .metric-value.normal { color: #22c55e; }
    .metric-value.watch { color: #eab308; }
    .metric-value.warning { color: #f97316; }
    .metric-value.critical { color: #ef4444; }
    
    .metric-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 8px;
    }
    
    /* Zone Card */
    .zone-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .zone-icon {
        font-size: 3rem;
        margin-bottom: 8px;
    }
    
    .zone-name {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .zone-name.normal { color: #22c55e; }
    .zone-name.watch { color: #eab308; }
    .zone-name.warning { color: #f97316; }
    .zone-name.critical { color: #ef4444; }
    
    .zone-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    /* Trend Card */
    .trend-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .trend-arrow {
        font-size: 3rem;
        line-height: 1;
    }
    
    .trend-arrow.rising { color: #ef4444; }
    .trend-arrow.stable { color: #22c55e; }
    .trend-arrow.falling { color: #3b82f6; }
    
    .trend-text {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 8px;
    }
    
    /* Camera Grid */
    .camera-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .camera-card:hover {
        border-color: rgba(34, 197, 94, 0.4);
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.1);
    }
    
    .camera-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: rgba(0,0,0,0.3);
        border-bottom: 1px solid var(--border);
    }
    
    .camera-id {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .camera-status {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    .camera-frame {
        height: 140px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, rgba(0,0,0,0.2), rgba(0,0,0,0.4));
    }
    
    .camera-frame img {
        max-width: 100%;
        max-height: 100%;
        object-fit: cover;
    }
    
    .camera-footer {
        display: flex;
        justify-content: space-between;
        padding: 10px 14px;
        font-size: 0.75rem;
    }
    
    .camera-tdi {
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .camera-zone {
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Section Header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 28px 0 18px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
    }
    
    .section-icon { font-size: 1.3rem; }
    
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Chart Container */
    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
    }
    
    /* Feature Bar */
    .feature-item {
        margin-bottom: 14px;
    }
    
    .feature-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    
    .feature-name {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    .feature-score {
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .feature-bar-bg {
        height: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .feature-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Explanation Card */
    .explanation-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), var(--bg-card));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 20px;
    }
    
    .explanation-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
    }
    
    .explanation-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--blue);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .explanation-text {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Onset Card */
    .onset-card {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.1), var(--bg-card));
        border: 1px solid rgba(249, 115, 22, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .onset-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .onset-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--orange);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0d12, #0f1318);
        border-right: 1px solid var(--border);
    }
    
    .sidebar-section {
        background: rgba(15, 19, 24, 0.5);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .sidebar-title {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 14px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        border: none !important;
        color: #0a0d12 !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.3) !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 30px rgba(34, 197, 94, 0.5) !important;
        transform: translateY(-3px) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), var(--bg-card)) !important;
        border-color: rgba(34, 197, 94, 0.4) !important;
        color: #22c55e !important;
    }
    
    /* Start Screen */
    .start-screen {
        text-align: center;
        padding: 80px 20px;
    }
    
    .start-icon {
        font-size: 5rem;
        margin-bottom: 24px;
    }
    
    .start-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 16px;
    }
    
    .start-desc {
        font-size: 1rem;
        color: var(--text-secondary);
        max-width: 600px;
        margin: 0 auto 40px auto;
        line-height: 1.6;
    }
    
    /* Info Badge */
    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        font-size: 0.75rem;
        color: #06b6d4;
        margin-bottom: 20px;
    }
    
    /* Ensemble Panel */
    .ensemble-panel {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .ensemble-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--purple);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .detector-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid var(--border);
    }
    
    .detector-row:last-child {
        border-bottom: none;
    }
    
    .detector-name {
        flex: 1;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    .detector-score {
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        min-width: 50px;
        text-align: right;
    }
    
    .detector-vote {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .detector-vote.normal { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .detector-vote.anomaly { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    
    .consensus-bar {
        height: 6px;
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
        margin-top: 12px;
        overflow: hidden;
    }
    
    .consensus-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    /* Incident Card */
    .incident-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    
    .incident-card:hover {
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .incident-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .incident-id {
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--cyan);
    }
    
    .incident-time {
        font-size: 0.7rem;
        color: var(--text-muted);
    }
    
    .incident-body {
        display: flex;
        gap: 16px;
    }
    
    .incident-metric {
        text-align: center;
    }
    
    .incident-metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .incident-metric-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
    }
    
    .incident-category {
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    /* Classification Badge */
    .classification-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), var(--bg-card));
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        font-size: 0.8rem;
        color: #8b5cf6;
        margin-top: 12px;
    }
    
    /* Forecast Panel */
    .forecast-panel {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), var(--bg-card));
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 16px;
        padding: 20px;
    }
    
    .forecast-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--cyan);
    }
    
    .forecast-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 3D Visualization */
    .latent-space-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_ucsd_dataset_path():
    """Get correct UCSD dataset path."""
    base_path = Path(__file__).parent.parent / "data" / "UCSD_Anomaly_Dataset.v1p2"
    if base_path.exists():
        return str(base_path)
    return UCSD_DATASET_PATH


def load_ucsd_frames(subset="ped1", split="Train", max_sequences=5, max_frames_per_seq=50):
    """Load frames from UCSD dataset."""
    dataset_path = Path(get_ucsd_dataset_path())
    
    if subset == "ped1":
        split_path = dataset_path / "UCSDped1" / split
    else:
        split_path = dataset_path / "UCSDped2" / split
    
    if not split_path.exists():
        return []
    
    frames = []
    sequences = sorted([d for d in split_path.iterdir() if d.is_dir()])[:max_sequences]
    
    for seq_path in sequences:
        tif_files = sorted(seq_path.glob("*.tif"))[:max_frames_per_seq]
        for tif_file in tif_files:
            frame = cv2.imread(str(tif_file), cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frames.append(frame)
    
    return frames


def frame_to_base64(frame):
    """Convert frame to base64 for display."""
    if frame is None:
        return ""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def extract_features_from_frames(frames, extractor):
    """Extract behavioral features from frames."""
    if not frames:
        return np.array([])
    
    features_list = []
    extractor.reset()
    
    for frame in frames:
        features = extractor.extract_features(frame)
        features_list.append(features)
    
    return np.array(features_list)


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    """Initialize session state."""
    defaults = {
        'initialized': False,
        'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'data_mode': 'synthetic',
        'history': {
            'tdi': [], 'zones': [], 'trends': [], 'confidences': [],
            'timestamps': [], 'features': [], 'top_features': [], 'explanations': [],
            'ensemble_scores': [], 'anomaly_categories': [], 'latent_means': [],
            'forecasts': [],
        },
        'incidents': [],  # List of logged incidents
        'drift_onset_frame': None,
        'train_frames': [],
        'test_frames': [],
        'all_frames': [],
        'frames_loaded': False,
        'current_frame_idx': 0,
        'cameras': [
            {'id': 'CAM-ALPHA', 'zone': 'Perimeter North', 'tdi': 0, 'status': 'NORMAL', 'frame_idx': 0},
            {'id': 'CAM-BRAVO', 'zone': 'Gate Sector', 'tdi': 0, 'status': 'NORMAL', 'frame_idx': 20},
            {'id': 'CAM-CHARLIE', 'zone': 'Perimeter East', 'tdi': 0, 'status': 'NORMAL', 'frame_idx': 40},
            {'id': 'CAM-DELTA', 'zone': 'Watch Tower', 'tdi': 0, 'status': 'NORMAL', 'frame_idx': 60},
            {'id': 'CAM-ECHO', 'zone': 'Perimeter South', 'tdi': 0, 'status': 'NORMAL', 'frame_idx': 80},
            {'id': 'CAM-FOXTROT', 'zone': 'Command Post', 'tdi': 0, 'status': 'NORMAL', 'frame_idx': 100},
        ]
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def load_all_ucsd_frames():
    """Load all UCSD frames for camera display."""
    frames = []
    dataset_path = Path(__file__).parent.parent / "data" / "UCSD_Anomaly_Dataset.v1p2"
    
    # Load from both Train and Test
    for split in ["Train", "Test"]:
        split_path = dataset_path / "UCSDped1" / split
        if split_path.exists():
            sequences = sorted([d for d in split_path.iterdir() if d.is_dir()])[:8]
            for seq_path in sequences:
                tif_files = sorted(seq_path.glob("*.tif"))[:30]
                for tif_file in tif_files:
                    frame = cv2.imread(str(tif_file), cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        # Resize for display
                        frame = cv2.resize(frame, (320, 240))
                        frames.append(frame)
    
    return frames


def load_all_drone_bird_frames(category: str = "mixed"):
    """Load Drone vs Bird frames for camera display.
    
    Args:
        category: 'bird' (normal), 'drone' (anomaly), or 'mixed' (both)
    """
    frames = []
    dataset_path = Path(__file__).parent.parent / "data" / "Drone_vs_Bird"
    
    if not dataset_path.exists():
        return frames
    
    categories = []
    if category == "mixed":
        categories = ["bird", "drone"]
    else:
        categories = [category]
    
    for cat in categories:
        cat_path = dataset_path / cat
        if cat_path.exists():
            # Get image files
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            files = []
            for ext in extensions:
                files.extend(cat_path.glob(ext))
            files = sorted(files)[:150]  # Limit per category
            
            for img_file in files:
                frame = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    frame = cv2.resize(frame, (320, 240))
                    frames.append(frame)
    
    return frames


def create_detection_video(dataset_type: str = "ucsd"):
    """Create a video file with detection overlays for playback.
    
    Args:
        dataset_type: 'ucsd' or 'drone_bird'
    
    Returns:
        Tuple: (path, anomaly_start, total_frames, processed_frames_list, incidents_list)
    """
    import tempfile
    import subprocess
    from datetime import datetime, timedelta
    
    # Load frames based on dataset type
    if dataset_type == "drone_bird":
        bird_frames = load_all_drone_bird_frames("bird")[:60]
        drone_frames = load_all_drone_bird_frames("drone")[:60]
        frames = bird_frames + drone_frames
        anomaly_start_idx = len(bird_frames)
        normal_label = "BIRD - Normal"
        anomaly_label = "DRONE - Threat"
    else:
        frames = load_all_ucsd_frames()[:120]
        anomaly_start_idx = 60
        normal_label = "Normal Activity"
        anomaly_label = "Anomaly Detected"
    
    if not frames:
        return None, 0, 0, [], []
    
    # Track incidents from video detection
    video_incidents = []
    incident_logged = {"early": False, "drift": False, "threat": False}
    
    # Create temp directory for frames
    temp_dir = Path(tempfile.gettempdir()) / f"drishti_{dataset_type}"
    temp_dir.mkdir(exist_ok=True)
    
    # Output path
    output_path = Path(tempfile.gettempdir()) / f"drishti_{dataset_type}_detection.webm"
    
    # Video settings
    h, w = frames[0].shape[:2]
    total_frames = len(frames)
    
    processed_frames = []
    
    for idx, frame in enumerate(frames):
        # Convert to color
        if len(frame.shape) == 2:
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame.copy()
        
        # Calculate TDI and status
        if idx < anomaly_start_idx:
            progress = idx / max(anomaly_start_idx, 1)
            tdi = 5 + progress * 15
            status = "NORMAL"
            box_color = (0, 255, 0)  # Green (BGR)
            label = normal_label
            status_bg = (0, 100, 0)
            zone = "Normal"
        else:
            progress = (idx - anomaly_start_idx) / max(total_frames - anomaly_start_idx, 1)
            tdi = 25 + progress * 55
            
            if progress < 0.3:
                status = "EARLY WARNING"
                box_color = (0, 200, 255)  # Orange
                status_bg = (0, 100, 150)
                zone = "Watch"
                # Log early warning incident
                if not incident_logged["early"]:
                    video_incidents.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame': idx,
                        'tdi': tdi,
                        'zone': zone,
                        'severity': 'LOW',
                        'type': 'Drift Onset',
                        'source': 'Video Detection',
                        'description': f"Early behavioral drift detected - {anomaly_label}"
                    })
                    incident_logged["early"] = True
            elif progress < 0.6:
                status = "DRIFT DETECTED"
                box_color = (0, 140, 255)  # Deep orange
                status_bg = (0, 70, 150)
                zone = "Warning"
                # Log drift detected incident
                if not incident_logged["drift"]:
                    video_incidents.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame': idx,
                        'tdi': tdi,
                        'zone': zone,
                        'severity': 'MEDIUM',
                        'type': 'Confirmed Drift',
                        'source': 'Video Detection',
                        'description': f"Significant behavioral drift - {anomaly_label}"
                    })
                    incident_logged["drift"] = True
            else:
                status = "THREAT CONFIRMED"
                box_color = (0, 0, 255)  # Red
                status_bg = (0, 0, 150)
                zone = "Critical"
                # Log threat incident
                if not incident_logged["threat"]:
                    video_incidents.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame': idx,
                        'tdi': tdi,
                        'zone': zone,
                        'severity': 'HIGH',
                        'type': 'Threat Confirmed',
                        'source': 'Video Detection',
                        'description': f"CRITICAL: Threat confirmed - {anomaly_label}"
                    })
                    incident_logged["threat"] = True
            label = anomaly_label
        
        # Draw detection box
        box_x1, box_y1 = int(w * 0.2), int(h * 0.15)
        box_x2, box_y2 = int(w * 0.8), int(h * 0.85)
        cv2.rectangle(frame_color, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame_color, (box_x1, box_y1 - 20), (box_x1 + label_size[0] + 10, box_y1), box_color, -1)
        cv2.putText(frame_color, label, (box_x1 + 5, box_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # TDI indicator (top left)
        tdi_text = f"TDI: {tdi:.1f}"
        cv2.rectangle(frame_color, (5, 5), (100, 30), (30, 30, 30), -1)
        cv2.putText(frame_color, tdi_text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Status indicator (top right)
        cv2.rectangle(frame_color, (w - 150, 5), (w - 5, 30), status_bg, -1)
        cv2.putText(frame_color, status, (w - 145, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Frame counter (bottom left)
        cv2.putText(frame_color, f"Frame: {idx+1}/{total_frames}", (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Progress bar (bottom)
        bar_y = h - 5
        bar_width = int((idx / total_frames) * (w - 20))
        cv2.rectangle(frame_color, (10, bar_y - 3), (w - 10, bar_y), (50, 50, 50), -1)
        
        # Color the progress bar based on phase
        if idx < anomaly_start_idx:
            bar_color = (0, 255, 0)  # Green
        else:
            bar_color = box_color
        cv2.rectangle(frame_color, (10, bar_y - 3), (10 + bar_width, bar_y), bar_color, -1)
        
        # Anomaly marker on progress bar
        anomaly_x = int((anomaly_start_idx / total_frames) * (w - 20)) + 10
        cv2.line(frame_color, (anomaly_x, bar_y - 6), (anomaly_x, bar_y + 2), (0, 0, 255), 2)
        
        processed_frames.append(frame_color)
    
    # Use imageio for browser-compatible video (mp4 with proper codec)
    try:
        import imageio
        
        # Create MP4 with imageio-ffmpeg (H.264)
        output_path = Path(tempfile.gettempdir()) / f"drishti_{dataset_type}_detection.mp4"
        
        # Convert BGR to RGB
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in processed_frames]
        
        # Write video with H.264 codec
        writer = imageio.get_writer(str(output_path), fps=10, codec='libx264', 
                                     pixelformat='yuv420p', quality=8)
        for frame in rgb_frames:
            writer.append_data(frame)
        writer.close()
        
        if output_path.exists() and output_path.stat().st_size > 1000:
            return str(output_path), anomaly_start_idx, total_frames, processed_frames, video_incidents
            
    except Exception as e:
        pass
    
    # Fallback: Create GIF using PIL (guaranteed to work)
    try:
        from PIL import Image
        gif_path = Path(tempfile.gettempdir()) / f"drishti_{dataset_type}_detection.gif"
        
        # Convert BGR to RGB PIL Images - use every 3rd frame for smaller size
        pil_frames = []
        for f in processed_frames[::3]:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))
        
        if pil_frames:
            # Save as animated GIF
            pil_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=150,  # ms per frame
                loop=0
            )
            
            if gif_path.exists():
                return str(gif_path), anomaly_start_idx, total_frames, processed_frames, video_incidents
    except Exception as e:
        pass
    
    return None, anomaly_start_idx, total_frames, processed_frames, video_incidents


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
@st.cache_resource
def initialize_system(use_real_video=False, use_drone_bird=False):
    """Initialize the intelligence system."""
    
    # Always use BEHAVIORAL_FEATURES (24 features for all datasets)
    feature_names = BEHAVIORAL_FEATURES
    n_features = len(BEHAVIORAL_FEATURES)
    
    # Generate or extract training data
    if use_drone_bird and DRONE_BIRD_AVAILABLE:
        # Drone-Bird Dataset
        try:
            loader = DroneBirdLoader(str(DRONE_BIRD_DATASET_PATH))
            extractor = DroneBirdFeatureExtractor(sample_interval=2)
            
            # Load bird images (normal baseline)
            bird_images = loader.load_images("bird", max_images=100, resize=(320, 240))
            
            if bird_images:
                train_data = extractor.extract_features_from_sequence(bird_images)
                # Remove zero rows (first frame has no flow)
                train_data = train_data[np.any(train_data != 0, axis=1)]
                
                if len(train_data) < 50:
                    st.warning("Not enough bird features, augmenting with synthetic data.")
                    synthetic = create_synthetic_normal_data(200, n_features)
                    train_data = np.vstack([train_data, synthetic]) if len(train_data) > 0 else synthetic
            else:
                st.warning("No bird images found. Using synthetic data.")
                train_data = create_synthetic_normal_data(500, n_features)
        except Exception as e:
            st.warning(f"Drone-Bird loading failed: {e}. Using synthetic data.")
            train_data = create_synthetic_normal_data(500, n_features)
    
    elif use_real_video and VIDEO_AVAILABLE:
        # UCSD Dataset
        try:
            extractor = RealVideoFeatureExtractor(sample_interval=3)
            frames = load_ucsd_frames("ped1", "Train", max_sequences=10, max_frames_per_seq=100)
            if frames:
                train_data = extract_features_from_frames(frames, extractor)
                if len(train_data) < 100:
                    train_data = create_synthetic_normal_data(500, n_features)
            else:
                train_data = create_synthetic_normal_data(500, n_features)
        except Exception as e:
            st.warning(f"Could not load real video: {e}")
            train_data = create_synthetic_normal_data(500, n_features)
    else:
        train_data = create_synthetic_normal_data(500, n_features)
    
    # Compute baseline
    baseline_means = np.mean(train_data, axis=0)
    baseline_stds = np.std(train_data, axis=0) + 1e-6
    
    # LSTM-VAE
    lstm_vae = TemporalNormalityLSTMVAE(
        input_dim=n_features,
        hidden_dim=32,
        latent_dim=8,
        seq_len=10,
    )
    
    sequences = [train_data[i:i+10] for i in range(len(train_data) - 10)]
    if sequences:
        lstm_vae.train(np.array(sequences), epochs=50)
    
    # Ensemble Detector
    ensemble_detector = None
    if ENSEMBLE_AVAILABLE:
        ensemble_detector = EnsembleAnomalyDetector(
            contamination=0.1,
            use_isolation_forest=True,
            use_one_class_svm=True,
            use_lof=True,
        )
        ensemble_detector.fit(train_data)
    
    # Anomaly Classifier
    anomaly_classifier = None
    if ADVANCED_AI_AVAILABLE:
        anomaly_classifier = AnomalyClassifier()
    
    # Confidence Calibrator
    confidence_calibrator = None
    if ADVANCED_AI_AVAILABLE:
        confidence_calibrator = ConfidenceCalibrator(temperature=1.5)
    
    # Incident Logger
    incident_logger = None
    if INCIDENT_LOGGER_AVAILABLE:
        incident_logger = IncidentLogger(storage_path="./incident_logs")
    
    # Intelligence components
    drift_engine = DriftIntelligenceEngine(
        baseline_frames=50,
        ewma_alpha=0.1,
        feature_names=feature_names,
    )
    
    zone_classifier = RiskZoneClassifier()
    
    attributor = DriftAttributor(
        feature_names=feature_names,
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
    )
    
    # Feature extractor for real video
    if use_drone_bird and DRONE_BIRD_AVAILABLE:
        feature_extractor = DroneBirdFeatureExtractor(sample_interval=2)
    elif VIDEO_AVAILABLE:
        feature_extractor = RealVideoFeatureExtractor(sample_interval=2)
    else:
        feature_extractor = None
    
    return (lstm_vae, drift_engine, zone_classifier, attributor, baseline_means, baseline_stds, 
            feature_extractor, train_data, ensemble_detector, anomaly_classifier, 
            confidence_calibrator, incident_logger, feature_names)


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================
def process_frame(features, frame_idx, lstm_vae, drift_engine, zone_classifier, attributor, 
                  feature_buffer, baseline_means, baseline_stds, ensemble_detector=None, 
                  anomaly_classifier=None, confidence_calibrator=None):
    """Process a single frame through the pipeline."""
    feature_buffer.append(features)
    if len(feature_buffer) > 10:
        feature_buffer.pop(0)
    
    # LSTM-VAE inference
    latent_mean = np.zeros(8)
    latent_logvar = np.zeros(8)
    
    if len(feature_buffer) >= 10:
        seq = np.array(feature_buffer[-10:]).reshape(1, 10, -1)
        output = lstm_vae.forward(seq, training=False)
        recon_loss = output.reconstruction_loss
        kl_div = output.kl_divergence
        latent_mean = output.latent_mean[0]
        latent_logvar = output.latent_log_var[0]
    else:
        recon_loss = float(np.mean((features - baseline_means) ** 2))
        kl_div = 0.0
    
    # Drift intelligence
    intelligence = drift_engine.process(
        reconstruction_loss=recon_loss,
        kl_divergence=kl_div,
        latent_mean=latent_mean,
        latent_logvar=latent_logvar,
        features=features,
        frame_index=frame_idx,
    )
    
    # Zone classification
    zone_state = zone_classifier.classify(
        threat_deviation_index=intelligence.threat_deviation_index,
        z_score=intelligence.z_score,
        trend_slope=intelligence.trend_slope,
        trend_persistence=intelligence.trend_persistence,
    )
    
    # Attribution
    attributions = attributor.compute_feature_attributions(features, top_k=5)
    
    # Explanation
    explanation = attributor.generate_explanation(
        current_features=features,
        threat_deviation_index=intelligence.threat_deviation_index,
        risk_zone=zone_state.zone.name,
        trend_direction=intelligence.drift_trend.name,
        confidence=intelligence.confidence,
    )
    
    # Ensemble Detection
    ensemble_result = None
    detector_scores = {}
    if ensemble_detector is not None and ENSEMBLE_AVAILABLE:
        lstm_score = min(recon_loss / 0.5, 1.0)  # Normalize reconstruction loss
        ensemble_result = ensemble_detector.predict(features, lstm_vae_score=lstm_score)
        if ensemble_result and hasattr(ensemble_result, 'votes'):
            for vote in ensemble_result.votes:
                detector_scores[vote.detector_type.value] = {
                    'score': vote.anomaly_score,
                    'is_anomaly': vote.is_anomaly,
                    'confidence': vote.confidence
                }
    
    # Anomaly Classification
    anomaly_category = 'normal'
    severity = 0
    suggested_response = 'Continue routine monitoring'
    
    if anomaly_classifier is not None and ADVANCED_AI_AVAILABLE:
        z_scores = (features - baseline_means) / baseline_stds
        classification = anomaly_classifier.classify(
            features=features,
            feature_names=BEHAVIORAL_FEATURES,
            z_scores=z_scores,
            tdi=intelligence.threat_deviation_index
        )
        anomaly_category = classification.primary_category.value
        severity = classification.severity
        suggested_response = classification.suggested_response
    
    # TDI Forecast (simple exponential smoothing)
    tdi_forecast = intelligence.threat_deviation_index * 1.05 if intelligence.drift_trend.name == 'INCREASING' else intelligence.threat_deviation_index * 0.95
    tdi_forecast = max(0, min(100, tdi_forecast))
    
    return {
        'tdi': intelligence.threat_deviation_index,
        'zone': zone_state.zone,
        'zone_name': zone_state.zone.name,
        'trend': intelligence.drift_trend,
        'trend_name': intelligence.drift_trend.name,
        'confidence': intelligence.confidence,
        'top_features': [{'name': a.feature_name, 'score': a.z_score} for a in attributions[:5]],
        'explanation': explanation.summary if explanation else "",
        'raw_features': features,
        'latent_mean': latent_mean,
        'ensemble_scores': detector_scores,
        'anomaly_category': anomaly_category,
        'severity': severity,
        'suggested_response': suggested_response,
        'tdi_forecast': tdi_forecast,
    }


def run_simulation(lstm_vae, drift_engine, zone_classifier, attributor, baseline_means, baseline_stds, 
                   drift_start=100, drift_rate=0.02, use_real_video=False, feature_extractor=None,
                   ensemble_detector=None, anomaly_classifier=None, confidence_calibrator=None,
                   use_drone_bird=False):
    """Run the full simulation."""
    n_features = len(BEHAVIORAL_FEATURES)
    
    # Generate test data based on dataset type
    all_data = None
    
    if use_drone_bird and DRONE_BIRD_AVAILABLE and feature_extractor:
        # Drone-Bird Dataset
        try:
            loader = DroneBirdLoader(str(DRONE_BIRD_DATASET_PATH))
            
            # Load bird images (normal)
            bird_images = loader.load_images("bird", max_images=80, resize=(320, 240))
            # Load drone images (anomaly)
            drone_images = loader.load_images("drone", max_images=80, resize=(320, 240))
            
            if bird_images and drone_images:
                feature_extractor.reset()
                normal_data = feature_extractor.extract_features_from_sequence(bird_images)
                # Remove zero rows
                normal_data = normal_data[np.any(normal_data != 0, axis=1)]
                
                feature_extractor.reset()
                anomaly_data = feature_extractor.extract_features_from_sequence(drone_images)
                anomaly_data = anomaly_data[np.any(anomaly_data != 0, axis=1)]
                
                if len(normal_data) > 10 and len(anomaly_data) > 10:
                    all_data = np.vstack([normal_data, anomaly_data])
                    drift_start = len(normal_data)
                    
                    st.session_state.train_frames = bird_images[:50]
                    st.session_state.test_frames = drone_images[:50]
        except Exception as e:
            st.warning(f"Drone-Bird error: {e}")
            all_data = None
    
    elif use_real_video and feature_extractor:
        try:
            # Load train frames as normal baseline
            train_frames = load_ucsd_frames("ped1", "Train", max_sequences=5, max_frames_per_seq=50)
            # Load test frames (contains anomalies)
            test_frames = load_ucsd_frames("ped1", "Test", max_sequences=3, max_frames_per_seq=100)
            
            if train_frames and test_frames:
                feature_extractor.reset()
                normal_data = extract_features_from_frames(train_frames, feature_extractor)
                feature_extractor.reset()
                test_data = extract_features_from_frames(test_frames, feature_extractor)
                
                all_data = np.vstack([normal_data, test_data]) if len(normal_data) > 0 and len(test_data) > 0 else None
                drift_start = len(normal_data)
                
                st.session_state.train_frames = train_frames
                st.session_state.test_frames = test_frames
            else:
                all_data = None
        except Exception as e:
            st.warning(f"Real video error: {e}")
            all_data = None
    else:
        all_data = None
    
    # Fallback to synthetic
    if all_data is None or len(all_data) < 50:
        normal_data = create_synthetic_normal_data(drift_start, n_features)
        drift_data = create_synthetic_drift_data(200, n_features, drift_rate)
        all_data = np.vstack([normal_data, drift_data])
    
    # Process
    results = []
    feature_buffer = []
    drift_engine.reset()
    zone_classifier.reset()
    
    drift_detected = False
    drift_onset = None
    incidents = []
    
    for i, features in enumerate(all_data):
        result = process_frame(
            features, i, lstm_vae, drift_engine, zone_classifier, 
            attributor, feature_buffer, baseline_means, baseline_stds,
            ensemble_detector, anomaly_classifier, confidence_calibrator
        )
        result['frame'] = i
        results.append(result)
        
        if not drift_detected and result['zone'] != RiskZone.NORMAL and i >= drift_start:
            drift_detected = True
            drift_onset = i
        
        # Log incidents for WARNING and CRITICAL zones
        if result['zone_name'] in ['WARNING', 'CRITICAL']:
            incident = {
                'id': f"INC-{hashlib.md5(f'{i}-{result["tdi"]}'.encode()).hexdigest()[:8].upper()}",
                'frame': i,
                'tdi': result['tdi'],
                'zone': result['zone_name'],
                'category': result.get('anomaly_category', 'unknown'),
                'severity': result.get('severity', 0),
                'timestamp': datetime.now().isoformat(),
                'response': result.get('suggested_response', ''),
            }
            incidents.append(incident)
    
    return results, drift_onset, drift_start, incidents


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_tdi_timeline(history, drift_start):
    """Create TDI timeline chart."""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15], shared_xaxes=True, vertical_spacing=0.02)
    
    if not history['tdi']:
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,13,18,0.5)')
        return fig
    
    frames = list(range(len(history['tdi'])))
    tdi = history['tdi']
    
    # Zone bands
    for y0, y1, color in [(0, 25, 'rgba(34,197,94,0.1)'), (25, 50, 'rgba(234,179,8,0.1)'), 
                          (50, 75, 'rgba(249,115,22,0.1)'), (75, 100, 'rgba(239,68,68,0.1)')]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, row=1, col=1)
    
    # TDI line
    fig.add_trace(go.Scatter(
        x=frames, y=tdi, mode='lines', name='TDI',
        line=dict(color='#22c55e', width=3),
        fill='tozeroy', fillcolor='rgba(34,197,94,0.15)',
        hovertemplate='Frame %{x}<br>TDI: %{y:.1f}<extra></extra>'
    ), row=1, col=1)
    
    # Thresholds
    for thresh, color in [(25, '#22c55e'), (50, '#eab308'), (75, '#f97316')]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=color, opacity=0.3, row=1, col=1)
    
    # Drift marker
    fig.add_vline(x=drift_start, line_dash="dash", line_color="#ef4444", opacity=0.6,
                  annotation_text="Drift Injected", annotation_position="top",
                  annotation_font_size=10, annotation_font_color="#94a3b8", row=1, col=1)
    
    # Zone timeline
    zone_colors = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}
    colors = [zone_colors.get(z, '#64748b') for z in history['zones']]
    fig.add_trace(go.Bar(x=frames, y=[1]*len(frames), marker_color=colors, showlegend=False), row=2, col=1)
    
    fig.update_layout(
        height=350, margin=dict(l=50, r=30, t=20, b=30), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,13,18,0.5)',
        font=dict(color='#94a3b8', family='Inter')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(title_text="TDI", range=[0, 100], row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><div class="sidebar-title"> Control Panel</div></div>', unsafe_allow_html=True)
        
        # Data Mode
        st.markdown('<div class="sidebar-section"><div class="sidebar-title">ðŸ“¡ Data Source</div></div>', unsafe_allow_html=True)
        data_mode = st.radio(
            "Select Mode",
            ["synthetic", "ucsd_video", "drone_bird"],
            format_func=lambda x: {
                "synthetic": "ðŸ”¬ Synthetic Data",
                "ucsd_video": "UCSD Pedestrian",
                "drone_bird": "Drone vs Bird"
            }.get(x, x),
            key="data_mode_select"
        )
        
        if data_mode == "ucsd_video":
            st.markdown("""
            <div style="background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.3); 
                        border-radius: 8px; padding: 10px; font-size: 0.75rem; margin: 10px 0;">
                <strong style="color: #06b6d4;">ðŸ“ UCSD Pedestrian Dataset</strong><br>
                <span style="color: #94a3b8;">Train: Normal walking patterns<br>
                Test: Bikes, skaters, carts (anomalies)</span>
            </div>
            """, unsafe_allow_html=True)
        
        elif data_mode == "drone_bird":
            st.markdown("""
            <div style="background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3); 
                        border-radius: 8px; padding: 10px; font-size: 0.75rem; margin: 10px 0;">
                <strong style="color: #8b5cf6;"> Drone vs Bird Dataset</strong><br>
                <span style="color: #94a3b8;">Train: Birds (natural movement)<br>
                Test: Drones (mechanical movement)</span><br>
                <span style="color: #f97316; font-size: 0.7rem;">âš¡ Defense: Airspace Security</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Parameters
        st.markdown('<div class="sidebar-section"><div class="sidebar-title">ðŸŽšï¸ Parameters</div></div>', unsafe_allow_html=True)
        drift_start = st.slider("Drift Onset Frame", 50, 200, 100, help="When drift begins")
        drift_rate = st.slider("Drift Intensity", 0.01, 0.05, 0.02, 0.005)
        
        st.markdown("---")
        
        # Zone Thresholds
        st.markdown('<div class="sidebar-section"><div class="sidebar-title">ðŸŽ¯ Zone Thresholds</div></div>', unsafe_allow_html=True)
        watch_thresh = st.slider("Watch", 15, 35, 25)
        warning_thresh = st.slider("Warning", 35, 60, 50)
        critical_thresh = st.slider("Critical", 55, 85, 75)
        
        st.markdown("---")
        
        # System Info  
        st.markdown(f"""
        <div style="font-size: 0.7rem; color: #64748b; padding: 10px;">
            <strong>TRL-4</strong>: Lab Validated<br>
            <strong>Mode</strong>: Decision Support<br>
            <strong>Philosophy</strong>: AI Assists, Not Replaces
        </div>
        """, unsafe_allow_html=True)
    
    # Get data mode from session state (set by main control panel or sidebar)
    data_mode = st.session_state.get('data_mode_main', st.session_state.get('data_mode_select', 'synthetic'))
    
    # Set initial values based on session state (will be overwritten by control panel)
    use_real = data_mode == "ucsd_video"
    use_drone_bird = data_mode == "drone_bird"
    
    # Current status
    if st.session_state.history.get('tdi'):
        latest_zone = st.session_state.history['zones'][-1]
        latest_tdi = st.session_state.history['tdi'][-1]
    else:
        latest_zone = 'STANDBY'
        latest_tdi = 0
    
    status_class = latest_zone.lower() if latest_zone != 'STANDBY' else 'normal'
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <div class="logo-section">
            <div class="logo-icon"><svg width="44" height="44" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="eyeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#00d4ff"/>
                <stop offset="100%" style="stop-color:#0891b2"/>
            </linearGradient>
            <linearGradient id="irisGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#06b6d4"/>
                <stop offset="100%" style="stop-color:#0e7490"/>
            </linearGradient>
        </defs>
        <ellipse cx="50" cy="50" rx="45" ry="28" fill="none" stroke="url(#eyeGrad)" stroke-width="3"/>
        <circle cx="50" cy="50" r="22" fill="url(#irisGrad)"/>
        <circle cx="50" cy="50" r="10" fill="#0f172a"/>
        <circle cx="45" cy="45" r="4" fill="rgba(255,255,255,0.8)"/>
    </svg></div>
            <div class="logo-text">
                <h1>DRISHTI</h1>
                <p>Vigilant Defense Intelligence</p>
            </div>
        </div>
        <div class="status-section">
            <div class="status-item">
                <div class="status-label">Session</div>
                <div class="status-value">{st.session_state.session_id}</div>
            </div>
            <div class="status-item">
                <div class="status-label">Data Mode</div>
                <div class="status-value">{"Drone-Bird" if use_drone_bird else "UCSD" if use_real else "Synthetic"}</div>
            </div>
            <div class="status-badge {status_class}">{latest_zone}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Control Panel - Always visible above tabs
    st.markdown("""
    <div style="background: rgba(15, 19, 24, 0.8); border: 1px solid rgba(56, 189, 248, 0.2); 
                border-radius: 12px; padding: 16px; margin-bottom: 20px;">
        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px;">
             Control Panel
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 1])
    
    with ctrl_col1:
        data_mode_main = st.radio(
            "Data Source",
            ["synthetic", "ucsd_video", "drone_bird"],
            format_func=lambda x: {
                "synthetic": "Synthetic",
                "ucsd_video": "UCSD Pedestrian",
                "drone_bird": "Drone vs Bird"
            }.get(x, x),
            horizontal=True,
            key="data_mode_main"
        )
    
    with ctrl_col2:
        drift_start = st.slider("Drift Onset", 50, 200, 100, key="drift_start_main")
    
    with ctrl_col3:
        drift_rate = st.slider("Drift Rate", 0.01, 0.05, 0.02, 0.005, key="drift_rate_main")
    
    with ctrl_col4:
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Update data mode based on main selector
    use_real = data_mode_main == "ucsd_video"
    use_drone_bird = data_mode_main == "drone_bird"
    
    # Initialize system with selected data mode
    (lstm_vae, drift_engine, zone_classifier, attributor, baseline_means, baseline_stds, 
     feature_extractor, train_data, ensemble_detector, anomaly_classifier, 
     confidence_calibrator, incident_logger, feature_names) = initialize_system(use_real, use_drone_bird)
    
    # Tabs - Consolidated to 4 main views
    tab1, tab2, tab3, tab4 = st.tabs(["ðŸ“Š Intelligence Dashboard", "ðŸ§  AI Ensemble", "ðŸ“¹ Camera Grid", "¨ Incident Log & Export"])
    
    # =========================================================================
    # TAB 1: INTELLIGENCE DASHBOARD
    # =========================================================================
    with tab1:
        if st.session_state.history.get('tdi'):
            idx = -1
            tdi = st.session_state.history['tdi'][idx]
            zone = st.session_state.history['zones'][idx]
            trend = st.session_state.history['trends'][idx]
            confidence = st.session_state.history['confidences'][idx]
            
            # TDI class
            if tdi < 25: tdi_class = 'normal'
            elif tdi < 50: tdi_class = 'watch'
            elif tdi < 75: tdi_class = 'warning'
            else: tdi_class = 'critical'
            
            # Zone info
            zone_icons = {'NORMAL': 'ðŸŸ¢', 'WATCH': 'ðŸŸ¡', 'WARNING': 'ðŸŸ ', 'CRITICAL': 'ðŸ”´'}
            zone_descs = {
                'NORMAL': 'Stable behavior within baseline',
                'WATCH': 'Weak drift detected - monitor closely',
                'WARNING': 'Confirmed behavioral drift',
                'CRITICAL': 'High threat - immediate action'
            }
            
            # Trend info
            trend_arrows = {'INCREASING': 'â†‘', 'STABLE': 'â†’', 'DECREASING': 'â†“'}
            trend_classes = {'INCREASING': 'rising', 'STABLE': 'stable', 'DECREASING': 'falling'}
            
            # ROW 1: Main Metrics
            col1, col2, col3, col4 = st.columns([1.2, 1.2, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Threat Deviation Index</div>
                    <div class="metric-value {tdi_class}">{tdi:.0f}</div>
                    <div class="metric-sub">Scale: 0-100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="zone-card">
                    <div class="zone-icon">{zone_icons.get(zone, 'âšª')}</div>
                    <div class="zone-name {zone.lower()}">{zone}</div>
                    <div class="zone-desc">{zone_descs.get(zone, '')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="trend-card">
                    <div class="metric-label">Drift Trend</div>
                    <div class="trend-arrow {trend_classes.get(trend, 'stable')}">{trend_arrows.get(trend, 'â†’')}</div>
                    <div class="trend-text">{trend.title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value normal">{confidence*100:.0f}%</div>
                    <div class="metric-sub">Assessment certainty</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ROW 2: Timeline
            st.markdown('<div class="section-header"><span class="section-icon">ðŸ“ˆ</span><span class="section-title">Threat Deviation Timeline</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_tdi_timeline(st.session_state.history, st.session_state.get('actual_drift_start', drift_start))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ROW 3: Attribution & Explanation
            col_attr, col_exp = st.columns([1, 1])
            
            with col_attr:
                st.markdown('<div class="section-header"><span class="section-icon">ðŸ”</span><span class="section-title">Feature Attribution</span></div>', unsafe_allow_html=True)
                
                if st.session_state.history.get('top_features') and st.session_state.history['top_features'][idx]:
                    for feat in st.session_state.history['top_features'][idx][:5]:
                        score = abs(feat['score'])
                        width = min(score * 25, 100)
                        color = '#22c55e' if score < 2 else '#eab308' if score < 3 else '#ef4444'
                        st.markdown(f"""
                        <div class="feature-item">
                            <div class="feature-header">
                                <span class="feature-name">{feat['name']}</span>
                                <span class="feature-score" style="color: {color};">z={feat['score']:.2f}</span>
                            </div>
                            <div class="feature-bar-bg">
                                <div class="feature-bar" style="width: {width}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_exp:
                st.markdown('<div class="section-header"><span class="section-icon">ðŸ’¡</span><span class="section-title">AI Explanation</span></div>', unsafe_allow_html=True)
                
                explanation = st.session_state.history['explanations'][idx] if st.session_state.history.get('explanations') else ""
                st.markdown(f"""
                <div class="explanation-card">
                    <div class="explanation-header">
                        <span>ðŸ¤–</span>
                        <span class="explanation-title">System Analysis</span>
                    </div>
                    <div class="explanation-text">{explanation if explanation else "Analyzing behavioral patterns..."}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Drift onset
                onset = st.session_state.drift_onset_frame
                if onset:
                    st.markdown(f"""
                    <div class="onset-card" style="margin-top: 16px;">
                        <div class="onset-label">Drift First Detected</div>
                        <div class="onset-value">Frame {onset}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ROW 4: Performance Metrics
            st.markdown('<div class="section-header"><span class="section-icon">ðŸ“‹</span><span class="section-title">Detection Performance</span></div>', unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            
            tdi_vals = st.session_state.history['tdi']
            zones = st.session_state.history['zones']
            actual_drift = st.session_state.get('actual_drift_start', drift_start)
            
            fp = sum(1 for i in range(min(actual_drift, len(zones))) if zones[i] != 'NORMAL')
            fp_rate = fp / actual_drift * 100 if actual_drift > 0 else 0
            onset = st.session_state.drift_onset_frame
            delay = onset - actual_drift if onset and onset >= actual_drift else None
            
            m1.metric("Detection Delay", f"{delay if delay else 'â€”'} frames")
            m2.metric("False Positive Rate", f"{fp_rate:.1f}%")
            m3.metric("Peak TDI", f"{max(tdi_vals):.1f}")
            m4.metric("Frames Analyzed", len(tdi_vals))
            
            # Reset
            if st.button("ðŸ”„ Reset Session"):
                st.session_state.history = {k: [] for k in st.session_state.history}
                st.session_state.drift_onset_frame = None
                st.rerun()
        
        else:
            # Start screen
            st.markdown("""
            <div class="start-screen">
                <div class="start-icon">ðŸ‘ï¸</div>
                <div class="start-title">Initialize DRISHTI Intelligence</div>
                <div class="start-desc">
                    DRISHTI analyzes behavioral patterns to detect gradual drift before threats manifest.
                    The system learns what "normal" looks like and alerts on subtle deviations.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if use_real:
                st.markdown("""
                <div style="text-align: center;">
                    <div class="info-badge">ðŸ“¹ Using UCSD Pedestrian Dataset - Real surveillance footage</div>
                </div>
                """, unsafe_allow_html=True)
            
            col_s1, col_btn, col_s2 = st.columns([1, 1, 1])
            with col_btn:
                if st.button("â–¶ Start Analysis", type="primary", use_container_width=True):
                    with st.spinner("ðŸ” DRISHTI analyzing patterns..."):
                        results, drift_onset, actual_drift, incidents = run_simulation(
                            lstm_vae, drift_engine, zone_classifier, attributor,
                            baseline_means, baseline_stds, drift_start, drift_rate,
                            use_real, feature_extractor, ensemble_detector,
                            anomaly_classifier, confidence_calibrator, use_drone_bird
                        )
                        
                        st.session_state.history = {
                            'tdi': [r['tdi'] for r in results],
                            'zones': [r['zone_name'] for r in results],
                            'trends': [r['trend_name'] for r in results],
                            'confidences': [r['confidence'] for r in results],
                            'timestamps': [r['frame'] for r in results],
                            'features': [r['raw_features'] for r in results],
                            'top_features': [r['top_features'] for r in results],
                            'explanations': [r['explanation'] for r in results],
                            'ensemble_scores': [r.get('ensemble_scores', {}) for r in results],
                            'anomaly_categories': [r.get('anomaly_category', 'normal') for r in results],
                            'latent_means': [r.get('latent_mean', np.zeros(8)) for r in results],
                            'forecasts': [r.get('tdi_forecast', 0) for r in results],
                        }
                        st.session_state.drift_onset_frame = drift_onset
                        st.session_state.actual_drift_start = actual_drift
                        st.session_state.incidents = incidents
                        time.sleep(0.2)
                    st.rerun()
    
    # =========================================================================
    # TAB 2: AI ENSEMBLE
    # =========================================================================
    with tab2:
        st.markdown('<div class="section-header"><span class="section-icon">ðŸ§ </span><span class="section-title">Multi-Model Ensemble Detection</span></div>', unsafe_allow_html=True)
        
        if st.session_state.history.get('tdi'):
            col_ens, col_lat = st.columns([1, 1])
            
            with col_ens:
                # Ensemble Agreement Panel
                st.markdown("""
                <div class="ensemble-panel">
                    <div class="ensemble-header">ðŸ”® Ensemble Detector Votes</div>
                """, unsafe_allow_html=True)
                
                # Get latest ensemble scores
                ensemble_scores = st.session_state.history.get('ensemble_scores', [{}])[-1]
                
                detector_info = {
                    'lstm_vae': ('LSTM-VAE', 'Primary temporal detector'),
                    'isolation_forest': ('Isolation Forest', 'Tree-based isolation'),
                    'one_class_svm': ('One-Class SVM', 'Boundary-based'),
                    'lof': ('Local Outlier Factor', 'Density-based'),
                }
                
                total_votes = 0
                anomaly_votes = 0
                
                for det_key, det_info in detector_info.items():
                    score_data = ensemble_scores.get(det_key, {'score': 0.5, 'is_anomaly': False})
                    if isinstance(score_data, dict):
                        score = score_data.get('score', 0.5)
                        is_anomaly = score_data.get('is_anomaly', False)
                    else:
                        score = 0.5
                        is_anomaly = False
                    
                    total_votes += 1
                    if is_anomaly:
                        anomaly_votes += 1
                    
                    vote_class = 'anomaly' if is_anomaly else 'normal'
                    vote_text = 'ANOMALY' if is_anomaly else 'NORMAL'
                    score_color = '#ef4444' if score > 0.6 else '#eab308' if score > 0.4 else '#22c55e'
                    
                    st.markdown(f"""
                    <div class="detector-row">
                        <div class="detector-name">{det_info[0]}<br><span style="font-size: 0.65rem; color: #64748b;">{det_info[1]}</span></div>
                        <div class="detector-score" style="color: {score_color};">{score:.2f}</div>
                        <div class="detector-vote {vote_class}">{vote_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Consensus bar
                agreement = anomaly_votes / max(total_votes, 1)
                consensus_color = '#ef4444' if agreement > 0.5 else '#22c55e'
                
                st.markdown(f"""
                <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 8px;">
                        <span style="color: #94a3b8;">Consensus Agreement</span>
                        <span style="color: {consensus_color}; font-weight: 600;">{agreement*100:.0f}% Anomaly</span>
                    </div>
                    <div class="consensus-bar">
                        <div class="consensus-fill" style="width: {agreement*100}%; background: {consensus_color};"></div>
                    </div>
                </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Anomaly Classification
                st.markdown('<div class="section-header"><span class="section-icon">ðŸ·ï¸</span><span class="section-title">Anomaly Classification</span></div>', unsafe_allow_html=True)
                
                categories = st.session_state.history.get('anomaly_categories', ['normal'])
                latest_category = categories[-1] if categories else 'normal'
                
                category_icons = {
                    'normal': 'âœ…', 'loitering': 'ðŸ§', 'intrusion': 'âš ï¸', 
                    'crowd_formation': 'ðŸ‘¥', 'erratic_movement': 'ðŸŒ€',
                    'coordinated': 'ðŸŽ¯', 'speed_anomaly': 'âš¡', 
                    'direction_anomaly': 'â†©ï¸', 'unknown': 'â“'
                }
                
                category_colors = {
                    'normal': '#22c55e', 'loitering': '#eab308', 'intrusion': '#ef4444',
                    'crowd_formation': '#f97316', 'erratic_movement': '#8b5cf6',
                    'coordinated': '#ef4444', 'speed_anomaly': '#06b6d4',
                    'direction_anomaly': '#ec4899', 'unknown': '#64748b'
                }
                
                st.markdown(f"""
                <div class="classification-badge" style="border-color: {category_colors.get(latest_category, '#64748b')}30; color: {category_colors.get(latest_category, '#64748b')};">
                    <span>{category_icons.get(latest_category, 'â“')}</span>
                    <span>{latest_category.replace('_', ' ').title()}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Category distribution
                if categories:
                    cat_counts = {}
                    for c in categories:
                        cat_counts[c] = cat_counts.get(c, 0) + 1
                    
                    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    for cat, count in sorted_cats:
                        pct = count / len(categories) * 100
                        color = category_colors.get(cat, '#64748b')
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <span style="font-size: 0.8rem;">{category_icons.get(cat, 'â“')}</span>
                            <span style="font-size: 0.75rem; color: #94a3b8; flex: 1;">{cat.replace('_', ' ').title()}</span>
                            <span style="font-size: 0.75rem; color: {color}; font-weight: 600;">{pct:.0f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_lat:
                # 3D Latent Space Visualization
                st.markdown('<div class="section-header"><span class="section-icon">ðŸŒŒ</span><span class="section-title">3D Latent Space Trajectory</span></div>', unsafe_allow_html=True)
                
                latent_means = st.session_state.history.get('latent_means', [])
                zones = st.session_state.history.get('zones', [])
                
                if latent_means and len(latent_means) > 10:
                    # Extract first 3 dimensions for 3D plot
                    latent_array = np.array([lm if isinstance(lm, np.ndarray) else np.zeros(8) for lm in latent_means])
                    
                    if latent_array.shape[1] >= 3:
                        x = latent_array[:, 0]
                        y = latent_array[:, 1]
                        z = latent_array[:, 2]
                        
                        # Color by zone status - this shows anomalies clearly
                        zone_colors_map = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}
                        colors = [zone_colors_map.get(zn, '#64748b') for zn in zones]
                        
                        fig = go.Figure()
                        
                        # Add trajectory line
                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='lines',
                            line=dict(color='rgba(100, 116, 139, 0.4)', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Add points colored by zone
                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors,
                                opacity=0.9,
                                line=dict(color='white', width=0.5)
                            ),
                            text=[f'Frame {i}<br>Zone: {z}' for i, z in enumerate(zones)],
                            hovertemplate='%{text}<br>z1: %{x:.2f}<br>z2: %{y:.2f}<br>z3: %{z:.2f}<extra></extra>'
                        ))
                        
                        # Add start point (large green)
                        fig.add_trace(go.Scatter3d(
                            x=[x[0]], y=[y[0]], z=[z[0]],
                            mode='markers',
                            marker=dict(size=12, color='#22c55e', symbol='diamond'),
                            name='Start',
                            hovertemplate='START<extra></extra>'
                        ))
                        
                        # Find anomaly start and add marker
                        anomaly_idx = next((i for i, zn in enumerate(zones) if zn in ['WARNING', 'CRITICAL']), None)
                        if anomaly_idx:
                            fig.add_trace(go.Scatter3d(
                                x=[x[anomaly_idx]], y=[y[anomaly_idx]], z=[z[anomaly_idx]],
                                mode='markers',
                                marker=dict(size=14, color='#ef4444', symbol='x'),
                                name='Anomaly Start',
                                hovertemplate='âš ï¸ ANOMALY START<extra></extra>'
                            ))
                        
                        fig.update_layout(
                            height=400,
                            margin=dict(l=0, r=0, t=20, b=0),
                            scene=dict(
                                xaxis=dict(title='zâ‚', backgroundcolor='rgba(10,13,18,0.5)', gridcolor='rgba(255,255,255,0.05)'),
                                yaxis=dict(title='zâ‚‚', backgroundcolor='rgba(10,13,18,0.5)', gridcolor='rgba(255,255,255,0.05)'),
                                zaxis=dict(title='zâ‚ƒ', backgroundcolor='rgba(10,13,18,0.5)', gridcolor='rgba(255,255,255,0.05)'),
                                bgcolor='rgba(10,13,18,0.5)'
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#94a3b8', size=10),
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=-0.1,
                                xanchor='center',
                                x=0.5,
                                font=dict(size=10)
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        
                        # Legend explanation
                        st.markdown("""
                        <div style="display: flex; gap: 12px; justify-content: center; font-size: 0.75rem; margin-top: -10px;">
                            <span>ðŸŸ¢ Normal</span>
                            <span>ðŸŸ¡ Watch</span>
                            <span>ðŸŸ  Warning</span>
                            <span>ðŸ”´ Critical</span>
                            <span>ðŸ’Ž Start</span>
                            <span>âŒ Anomaly</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Insufficient latent dimensions for 3D visualization")
                else:
                    st.info("Run analysis to see latent space trajectory")
                
                # TDI Forecast
                st.markdown('<div class="section-header"><span class="section-icon">ðŸ”®</span><span class="section-title">TDI Forecast</span></div>', unsafe_allow_html=True)
                
                forecasts = st.session_state.history.get('forecasts', [])
                if forecasts:
                    latest_forecast = forecasts[-1]
                    current_tdi = st.session_state.history['tdi'][-1]
                    
                    forecast_trend = 'â†‘' if latest_forecast > current_tdi else 'â†“' if latest_forecast < current_tdi else 'â†’'
                    forecast_color = '#ef4444' if latest_forecast > 50 else '#eab308' if latest_forecast > 25 else '#22c55e'
                    
                    st.markdown(f"""
                    <div class="forecast-panel">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="forecast-label">Predicted Next TDI</div>
                                <div class="forecast-value" style="color: {forecast_color};">{latest_forecast:.1f} {forecast_trend}</div>
                            </div>
                            <div style="text-align: right;">
                                <div class="forecast-label">Current</div>
                                <div style="font-size: 1.5rem; color: #94a3b8;">{current_tdi:.1f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Run analysis to see TDI forecast")
        
        else:
            st.info("â–¶ Run analysis from the Intelligence Dashboard tab to see ensemble detection results")
    
    # =========================================================================
    # TAB 3: CAMERA GRID
    # =========================================================================
    with tab3:
        st.markdown('<div class="section-header"><span class="section-icon">ðŸ“¹</span><span class="section-title">Multi-Camera Surveillance Grid</span></div>', unsafe_allow_html=True)
        
        # =================================================================
        # LIVE DETECTION FEED - Playable Video with Anomaly Detection
        # =================================================================
        if use_real or use_drone_bird:
            st.markdown('<div class="section-header"><span class="section-icon">ðŸŽ¬</span><span class="section-title">Live Detection Feed</span></div>', unsafe_allow_html=True)
            
            # Determine dataset type
            dataset_type = "drone_bird" if use_drone_bird else "ucsd"
            feed_title = " Airspace Surveillance - Drone Detection" if use_drone_bird else "¶ Pedestrian Surveillance - Behavior Analysis"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 12px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3); margin-bottom: 16px;">
                <span style="color: #3b82f6; font-weight: 600; font-size: 1.1rem;">{feed_title}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate video with detection overlays
            with st.spinner("ðŸŽ¬ Generating detection video..."):
                video_path, anomaly_start, total_frames, detection_frames, video_incidents = create_detection_video(dataset_type)
            
            # Store processed frames in session state for camera grid sync
            if detection_frames:
                st.session_state['detection_video_frames'] = detection_frames
                st.session_state['anomaly_start_frame'] = anomaly_start
                
            # Add video detection incidents to session state
            if video_incidents:
                existing_incidents = st.session_state.get('incidents', [])
                # Check if video incidents already added (avoid duplicates)
                video_incident_types = {inc['type'] for inc in video_incidents}
                existing_types = {inc.get('type', '') for inc in existing_incidents if inc.get('source') == 'Video Detection'}
                new_incidents = [inc for inc in video_incidents if inc['type'] not in existing_types]
                if new_incidents:
                    st.session_state.incidents = existing_incidents + new_incidents
            
            if video_path and Path(video_path).exists():
                col_video, col_info = st.columns([2, 1])
                
                with col_video:
                    # Check if it's a GIF or video
                    if video_path.endswith('.gif'):
                        # Display GIF as image (auto-plays)
                        st.image(video_path, use_container_width=True)
                        st.caption("ðŸ”„ GIF auto-loops - Watch the detection progression")
                    else:
                        # Play the video
                        video_file = open(video_path, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    st.markdown(f"""
                    <div style="margin-top: 8px; padding: 10px; background: rgba(30, 41, 59, 0.5); border-radius: 8px; text-align: center;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">
                            ðŸŸ¢ Normal â†’ ðŸŸ¡ Early Warning â†’ ðŸŸ  Drift â†’ ðŸ”´ Anomaly
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_info:
                    # Video info panel
                    normal_label = "Birds (Normal Flight)" if use_drone_bird else "Pedestrians (Normal)"
                    anomaly_label = "Drones (Mechanical)" if use_drone_bird else "Anomalous Behavior"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: #3b82f6;">
                        <div class="metric-label">Detection Demo</div>
                        <div class="metric-value" style="color: #3b82f6; font-size: 1rem;">READY</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Frames</div>
                        <div class="metric-value">{total_frames}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Anomaly Starts At</div>
                        <div class="metric-value" style="color: #f97316;">Frame {anomaly_start}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detection phases - using simple HTML divs
                    st.markdown("""
                    <div style="padding: 12px; background: rgba(30, 41, 59, 0.5); border-radius: 8px; margin-top: 8px;">
                        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 12px; font-weight: 600;">Detection Phases</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use columns for phases instead of complex HTML
                    st.markdown(f"ðŸŸ¢ **{normal_label}**")
                    st.markdown("ðŸŸ¡ **Early Warning**")
                    st.markdown("ðŸŸ  **Drift Detected**")
                    st.markdown(f"ðŸ”´ **{anomaly_label}**")
                    
                    # Key insight
                    st.info("ðŸ’¡ Watch how TDI rises BEFORE the threat is fully visible - that's early warning detection!")
            else:
                st.warning("Could not generate detection video. Make sure dataset is available.")
            
            st.markdown("---")
        
        st.markdown('<div class="section-header"><span class="section-icon">ðŸ“¹</span><span class="section-title">Multi-Camera Grid</span></div>', unsafe_allow_html=True)
        
        # Load frames based on selected mode - prioritize detection video frames if available
        detection_frames = st.session_state.get('detection_video_frames', [])
        
        if detection_frames:
            # Use the exact same frames from the detection video
            all_frames = detection_frames
            frame_source = "Video Detection"
            st.markdown(f"""
            <div class="info-badge" style="margin-bottom: 16px; background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); color: #10b981;">
                ðŸŽ¯ <strong>Synced Feed:</strong> {len(all_frames)} processed frames from Detection Video
            </div>
            """, unsafe_allow_html=True)
        elif use_drone_bird:
            all_frames = load_all_drone_bird_frames("mixed")
            if all_frames:
                st.markdown(f"""
                <div class="info-badge" style="margin-bottom: 16px; background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); color: #10b981;">
                     Airspace Feed: {len(all_frames)} frames from Drone vs Bird Dataset
                </div>
                """, unsafe_allow_html=True)
        elif use_real:
            all_frames = load_all_ucsd_frames()
            if all_frames:
                st.markdown(f"""
                <div class="info-badge" style="margin-bottom: 16px;">
                    ðŸ“¹ Live Feed: {len(all_frames)} frames from UCSD Pedestrian Dataset
                </div>
                """, unsafe_allow_html=True)
        else:
            all_frames = []
            st.markdown("""
            <div class="info-badge" style="margin-bottom: 16px; background: rgba(139, 92, 246, 0.1); border-color: rgba(139, 92, 246, 0.3); color: #8b5cf6;">
                ðŸ”¬ Synthetic Mode: Simulated camera feeds (switch to UCSD Real Video for live frames)
            </div>
            """, unsafe_allow_html=True)
        
        # Update camera TDIs based on main analysis
        if st.session_state.history.get('tdi'):
            main_tdi = st.session_state.history['tdi'][-1]
            for cam in st.session_state.cameras:
                variance = np.random.uniform(-15, 20)
                cam['tdi'] = max(0, min(100, main_tdi + variance))
                if cam['tdi'] < 25: cam['status'] = 'NORMAL'
                elif cam['tdi'] < 50: cam['status'] = 'WATCH'
                elif cam['tdi'] < 75: cam['status'] = 'WARNING'
                else: cam['status'] = 'CRITICAL'
        
        # Display cameras with detection overlays
        cols = st.columns(3)
        zone_colors = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}
        zone_colors_bgr = {'NORMAL': (0, 255, 0), 'WATCH': (0, 200, 255), 'WARNING': (0, 140, 255), 'CRITICAL': (0, 0, 255)}
        status_labels = {'NORMAL': 'NORMAL', 'WATCH': 'EARLY WARNING', 'WARNING': 'DRIFT DETECTED', 'CRITICAL': 'THREAT'}
        
        for i, cam in enumerate(st.session_state.cameras):
            with cols[i % 3]:
                color = zone_colors.get(cam['status'], '#64748b')
                box_color_bgr = zone_colors_bgr.get(cam['status'], (100, 100, 100))
                status_text = status_labels.get(cam['status'], 'UNKNOWN')
                
                # Check if we have detection video frames (already processed with overlays)
                detection_frames = st.session_state.get('detection_video_frames', [])
                
                if detection_frames:
                    # Use pre-processed frames from detection video - already have overlays
                    # Each camera shows sequential frames to simulate multi-view
                    frame_offset = cam.get('frame_idx', i * 15) % len(detection_frames)
                    frame = detection_frames[frame_offset]  # Already in BGR with overlays
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    if frame_b64:
                        frame_html = f'<img src="data:image/jpeg;base64,{frame_b64}" style="width: 100%; height: 140px; object-fit: cover;">'
                    else:
                        frame_html = '<span style="color: #64748b; font-size: 2.5rem;">ðŸ“·</span>'
                        
                # Show real frame for UCSD or Drone-Bird mode with detection overlays
                elif (use_real or use_drone_bird) and all_frames:
                    # Each camera shows different frames
                    frame_offset = cam.get('frame_idx', i * 30) % len(all_frames)
                    frame = all_frames[frame_offset].copy()
                    
                    # Convert to color if grayscale
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    h, w = frame.shape[:2]
                    
                    # Draw detection box based on status
                    box_x1, box_y1 = int(w * 0.15), int(h * 0.1)
                    box_x2, box_y2 = int(w * 0.85), int(h * 0.9)
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color_bgr, 2)
                    
                    # Detection label
                    detect_label = "Normal" if cam['status'] == 'NORMAL' else "ANOMALY"
                    if use_drone_bird:
                        detect_label = "BIRD" if cam['status'] == 'NORMAL' else "DRONE"
                    
                    label_size = cv2.getTextSize(detect_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(frame, (box_x1, box_y1 - 15), (box_x1 + label_size[0] + 6, box_y1), box_color_bgr, -1)
                    cv2.putText(frame, detect_label, (box_x1 + 3, box_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # TDI on frame
                    tdi_text = f"TDI:{cam['tdi']:.0f}"
                    cv2.rectangle(frame, (3, 3), (60, 18), (30, 30, 30), -1)
                    cv2.putText(frame, tdi_text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color_bgr, 1)
                    
                    # Status indicator
                    status_bg = {'NORMAL': (0, 80, 0), 'WATCH': (0, 80, 120), 'WARNING': (0, 60, 120), 'CRITICAL': (0, 0, 120)}
                    cv2.rectangle(frame, (w - 70, 3), (w - 3, 18), status_bg.get(cam['status'], (50, 50, 50)), -1)
                    cv2.putText(frame, status_text[:8], (w - 68, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    if frame_b64:
                        frame_html = f'<img src="data:image/jpeg;base64,{frame_b64}" style="width: 100%; height: 140px; object-fit: cover;">'
                    else:
                        frame_html = '<span style="color: #64748b; font-size: 2.5rem;">ðŸ“·</span>'
                else:
                    # Synthetic mode - show simulated camera icon with status color
                    frame_html = f'''
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
                        <span style="font-size: 2.5rem; opacity: 0.6;">ðŸ“¹</span>
                        <span style="font-size: 0.7rem; color: {color}; margin-top: 8px;">SIMULATED</span>
                    </div>
                    '''
                
                st.markdown(f"""
                <div class="camera-card">
                    <div class="camera-header">
                        <span class="camera-id">{cam['id']}</span>
                        <div class="camera-status" style="background: {color};"></div>
                    </div>
                    <div class="camera-frame" style="height: 140px; overflow: hidden;">
                        {frame_html}
                    </div>
                    <div class="camera-footer">
                        <span class="camera-tdi" style="color: {color};">TDI: {cam['tdi']:.0f}</span>
                        <span class="camera-zone" style="color: {color};">{cam['status']}</span>
                    </div>
                </div>
                <div style="text-align: center; font-size: 0.7rem; color: #64748b; margin-top: 4px; margin-bottom: 16px;">
                    {cam['zone']}
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 4: INCIDENT LOG
    # =========================================================================
    with tab4:
        st.markdown('<div class="section-header"><span class="section-icon">¨</span><span class="section-title">Incident Log & Alert History</span></div>', unsafe_allow_html=True)
        
        incidents = st.session_state.get('incidents', [])
        
        if incidents:
            # Summary stats
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value warning">{len(incidents)}</div>
                    <div class="metric-label">Total Incidents</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s2:
                # Count HIGH severity or CRITICAL zone incidents
                critical_count = sum(1 for i in incidents if i.get('zone') in ['CRITICAL', 'Critical'] or i.get('severity') == 'HIGH')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value critical">{critical_count}</div>
                    <div class="metric-label">Critical/High Alerts</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s3:
                # Include both category and type fields
                categories = [i.get('category', i.get('type', 'unknown')) for i in incidents]
                most_common = max(set(categories), key=categories.count) if categories else 'N/A'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.2rem;">{str(most_common).replace('_', ' ').title()}</div>
                    <div class="metric-label">Most Common Type</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s4:
                avg_tdi = np.mean([i.get('tdi', 0) for i in incidents])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value warning">{avg_tdi:.1f}</div>
                    <div class="metric-label">Avg Incident TDI</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Incident Timeline Chart
            st.markdown('<div class="section-header"><span class="section-icon">ðŸ“ˆ</span><span class="section-title">Incident Timeline</span></div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            frames = [i.get('frame', 0) for i in incidents]
            tdis = [i.get('tdi', 0) for i in incidents]
            zones = [i.get('zone', 'WARNING') for i in incidents]
            
            zone_colors_map = {
                'NORMAL': '#22c55e', 'Normal': '#22c55e',
                'WATCH': '#eab308', 'Watch': '#eab308',
                'WARNING': '#f97316', 'Warning': '#f97316',
                'CRITICAL': '#ef4444', 'Critical': '#ef4444'
            }
            colors = [zone_colors_map.get(z, '#f97316') for z in zones]
            
            fig.add_trace(go.Scatter(
                x=frames, y=tdis,
                mode='markers',
                marker=dict(size=12, color=colors, line=dict(color='white', width=2)),
                hovertemplate='Frame: %{x}<br>TDI: %{y:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                height=200,
                margin=dict(l=40, r=20, t=20, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(10,13,18,0.5)',
                font=dict(color='#94a3b8'),
                xaxis=dict(title='Frame', gridcolor='rgba(255,255,255,0.03)'),
                yaxis=dict(title='TDI', gridcolor='rgba(255,255,255,0.03)', range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Incident List
            st.markdown('<div class="section-header"><span class="section-icon">ðŸ“‹</span><span class="section-title">Incident Details</span></div>', unsafe_allow_html=True)
            
            for incident in reversed(incidents[-20:]):  # Show last 20
                zone = incident.get('zone', 'WARNING')
                zone_color = '#22c55e' if zone in ['Normal', 'NORMAL'] else ('#eab308' if zone in ['Watch', 'WATCH'] else ('#f97316' if zone in ['Warning', 'WARNING'] else '#ef4444'))
                category = incident.get('category', incident.get('type', 'unknown'))
                severity = incident.get('severity', 'MEDIUM')
                source = incident.get('source', 'Analysis')
                description = incident.get('description', '')
                
                category_icons = {
                    'normal': 'âœ…', 'loitering': 'ðŸ§', 'intrusion': 'âš ï¸', 
                    'crowd_formation': 'ðŸ‘¥', 'erratic_movement': 'ðŸŒ€',
                    'coordinated': 'ðŸŽ¯', 'speed_anomaly': 'âš¡', 
                    'direction_anomaly': 'â†©ï¸', 'unknown': 'â“',
                    'Drift Onset': 'ðŸŸ¡', 'Confirmed Drift': 'ðŸŸ ', 
                    'Threat Confirmed': 'ðŸ”´', 'Video Detection': 'ðŸ“¹'
                }
                
                severity_colors = {'LOW': '#eab308', 'MEDIUM': '#f97316', 'HIGH': '#ef4444'}
                severity_color = severity_colors.get(severity, zone_color)
                
                st.markdown(f"""
                <div class="incident-card" style="border-left: 3px solid {zone_color};">
                    <div class="incident-header">
                        <span class="incident-id">{incident.get('id', source)}</span>
                        <span class="incident-time">Frame {incident.get('frame', 0)}</span>
                    </div>
                    <div class="incident-body">
                        <div class="incident-metric">
                            <div class="incident-metric-value" style="color: {zone_color};">{incident.get('tdi', 0):.0f}</div>
                            <div class="incident-metric-label">TDI</div>
                        </div>
                        <div class="incident-metric">
                            <div class="incident-metric-value" style="color: {zone_color};">{zone}</div>
                            <div class="incident-metric-label">Zone</div>
                        </div>
                        <div class="incident-metric">
                            <div class="incident-metric-value" style="color: {severity_color}; font-size: 0.9rem;">{severity}</div>
                            <div class="incident-metric-label">Severity</div>
                        </div>
                        <div class="incident-metric">
                            <span class="incident-category" style="background: {zone_color}20; color: {zone_color};">
                                {category_icons.get(category, 'ðŸ”')} {str(category).replace('_', ' ').title()}
                            </span>
                        </div>
                    </div>
                    {f'<div style="margin-top: 8px; padding: 8px; background: rgba(30, 41, 59, 0.5); border-radius: 4px; font-size: 0.8rem; color: #94a3b8;">{description}</div>' if description else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Export incidents
            st.markdown("<br>", unsafe_allow_html=True)
            col_exp1, col_exp2, _ = st.columns([1, 1, 2])
            
            with col_exp1:
                incident_df = pd.DataFrame(incidents)
                csv_data = incident_df.to_csv(index=False)
                st.download_button(
                    label="ðŸ“¥ Export Incidents CSV",
                    data=csv_data,
                    file_name=f"incidents_{st.session_state.session_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                json_data = json.dumps(incidents, indent=2)
                st.download_button(
                    label="ðŸ“¥ Export Incidents JSON",
                    data=json_data,
                    file_name=f"incidents_{st.session_state.session_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Full Session Export Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="section-icon">ðŸ“Š</span><span class="section-title">Full Session Analytics & Export</span></div>', unsafe_allow_html=True)
            
            if st.session_state.history.get('tdi'):
                tdi_vals = st.session_state.history['tdi']
                zones = st.session_state.history['zones']
                
                # Quick stats row
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value normal">{np.mean(tdi_vals):.1f}</div>
                        <div class="metric-label">Average TDI</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stat_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value warning">{max(tdi_vals):.1f}</div>
                        <div class="metric-label">Peak TDI</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stat_col3:
                    normal_pct = zones.count('NORMAL') / len(zones) * 100 if zones else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value normal">{normal_pct:.0f}%</div>
                        <div class="metric-label">Time in Normal</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stat_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(tdi_vals)}</div>
                        <div class="metric-label">Frames Analyzed</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Full export buttons
                exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
                
                with exp_col1:
                    df = pd.DataFrame({
                        'Frame': range(len(tdi_vals)),
                        'TDI': tdi_vals,
                        'Zone': zones,
                        'Trend': st.session_state.history.get('trends', []),
                        'Confidence': st.session_state.history.get('confidences', []),
                    })
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="ðŸ“¥ Full Analysis CSV",
                        data=csv_data,
                        file_name=f"drishti_analysis_{st.session_state.session_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with exp_col2:
                    export_data = {
                        'session_id': st.session_state.session_id,
                        'timestamp': datetime.now().isoformat(),
                        'data_mode': 'real_video' if use_real else 'synthetic',
                        'frames_analyzed': len(tdi_vals),
                        'drift_onset_frame': st.session_state.drift_onset_frame,
                        'peak_tdi': float(max(tdi_vals)),
                        'avg_tdi': float(np.mean(tdi_vals)),
                        'incidents': incidents,
                        'history': {
                            'tdi': [float(x) for x in tdi_vals],
                            'zones': zones,
                            'trends': st.session_state.history.get('trends', []),
                        }
                    }
                    json_data = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="ðŸ“¥ Full Analysis JSON",
                        data=json_data,
                        file_name=f"drishti_analysis_{st.session_state.session_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with exp_col3:
                    st.markdown(f"""
                    <div style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); 
                                border-radius: 8px; padding: 12px; font-size: 0.8rem;">
                        <strong style="color: #22c55e;">ðŸ“Š Report Ready</strong><br>
                        <span style="color: #94a3b8;">Session: {st.session_state.session_id}<br>
                        Frames: {len(tdi_vals)} | Peak TDI: {max(tdi_vals):.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 4rem; margin-bottom: 20px;">âœ…</div>
                <div style="font-size: 1.2rem; color: #22c55e; margin-bottom: 10px;">No Incidents Detected</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">All behavioral patterns within normal parameters. Run analysis to generate data.</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


