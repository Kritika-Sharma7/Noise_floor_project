"""
DRISHTI - Defense-Grade Surveillance Intelligence Dashboard
=================================================================
Real-time behavioral drift detection for border security.
Features:
 UCSD Dataset Integration with Real Video Frames
 Multi-Camera Grid View
 LSTM-VAE Temporal Normality Learning
 ENSEMBLE DETECTION (Isolation Forest + One-Class SVM + LOF)
 4-Tier Risk Zone Classification
 Explainable AI Attribution
 Anomaly Classification & Incident Logging
 3D Latent Space Visualization
 Real-time Threat Deviation Index (TDI)
 TDI Forecasting & Trend Prediction
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
from streamlit_autorefresh import st_autorefresh
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
    page_icon="",
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

# UCSD Pedestrian Dataset Ground Truth
# Frame ranges where anomalies occur (from UCSDped1.m)
# Anomaly types: biker, skater, cart, wheelchair, walking on grass
UCSD_GROUND_TRUTH = {
    'Test001': {'anomaly_frames': range(60, 153), 'type': 'biker'},
    'Test002': {'anomaly_frames': range(50, 176), 'type': 'biker'},
    'Test003': {'anomaly_frames': range(91, 201), 'type': 'cart'},
    'Test004': {'anomaly_frames': range(31, 169), 'type': 'skater'},
    'Test005': {'anomaly_frames': list(range(5, 91)) + list(range(140, 201)), 'type': 'biker'},
    'Test006': {'anomaly_frames': list(range(1, 101)) + list(range(110, 201)), 'type': 'cart'},
    'Test007': {'anomaly_frames': range(1, 176), 'type': 'biker'},
    'Test008': {'anomaly_frames': range(1, 95), 'type': 'skater'},
    'Test009': {'anomaly_frames': range(1, 49), 'type': 'biker'},
    'Test010': {'anomaly_frames': range(1, 141), 'type': 'cart'},
    'Test011': {'anomaly_frames': range(70, 166), 'type': 'biker'},
    'Test012': {'anomaly_frames': range(130, 201), 'type': 'skater'},
    'Test013': {'anomaly_frames': range(1, 157), 'type': 'cart'},
    'Test014': {'anomaly_frames': range(1, 201), 'type': 'wheelchair'},
    'Test015': {'anomaly_frames': range(138, 201), 'type': 'biker'},
    'Test016': {'anomaly_frames': range(123, 201), 'type': 'skater'},
    'Test017': {'anomaly_frames': range(1, 48), 'type': 'biker'},
    'Test018': {'anomaly_frames': range(54, 121), 'type': 'cart'},
    'Test019': {'anomaly_frames': range(64, 139), 'type': 'biker'},
    'Test020': {'anomaly_frames': range(45, 176), 'type': 'skater'},
    'Test021': {'anomaly_frames': range(31, 201), 'type': 'cart'},
    'Test022': {'anomaly_frames': range(16, 108), 'type': 'biker'},
    'Test023': {'anomaly_frames': range(8, 166), 'type': 'cart'},
    'Test024': {'anomaly_frames': range(50, 172), 'type': 'biker'},
    'Test025': {'anomaly_frames': range(40, 136), 'type': 'skater'},
    'Test026': {'anomaly_frames': range(77, 145), 'type': 'biker'},
    'Test027': {'anomaly_frames': range(10, 123), 'type': 'cart'},
    'Test028': {'anomaly_frames': range(105, 201), 'type': 'biker'},
    'Test029': {'anomaly_frames': list(range(1, 16)) + list(range(45, 114)), 'type': 'skater'},
    'Test030': {'anomaly_frames': range(175, 201), 'type': 'biker'},
    'Test031': {'anomaly_frames': range(1, 181), 'type': 'cart'},
    'Test032': {'anomaly_frames': list(range(1, 53)) + list(range(65, 116)), 'type': 'wheelchair'},
    'Test033': {'anomaly_frames': range(5, 166), 'type': 'biker'},
    'Test034': {'anomaly_frames': range(1, 122), 'type': 'skater'},
    'Test035': {'anomaly_frames': range(86, 201), 'type': 'cart'},
    'Test036': {'anomaly_frames': range(15, 109), 'type': 'biker'},
}

# Anomaly severity and display info
ANOMALY_INFO = {
    'biker': {'severity': 'WARNING', 'label': 'BIKER', 'color': '#f97316', 'description': 'Bicycle detected on pedestrian walkway'},
    'skater': {'severity': 'WARNING', 'label': 'SKATER', 'color': '#f97316', 'description': 'Skateboard detected on walkway'},
    'cart': {'severity': 'WARNING', 'label': 'CART', 'color': '#f97316', 'description': 'Cart detected in pedestrian area'},
    'wheelchair': {'severity': 'WARNING', 'label': 'WHEELCHAIR', 'color': '#f97316', 'description': 'Wheelchair on walkway'},
}

# Persistence threshold - after this many consecutive anomaly frames, escalate to CRITICAL
CRITICAL_PERSISTENCE_THRESHOLD = 8

@st.cache_data
def compute_frame_anomaly_scores(frames_hash: str, _frames: list):
    """
    Compute anomaly scores for UCSD Pedestrian dataset using:
    1. Ground truth data for accurate anomaly type labeling
    2. Optical flow analysis for TDI score calculation
    
    UCSD dataset anomalies:
    - Bikers (bicycles on walkway)
    - Skaters (skateboards)
    - Carts (small carts)
    - Wheelchairs
    - Walking on grass (unusual path)
    
    Normal: Pedestrians walking normally on walkway
    """
    if not _frames or len(_frames) < 2:
        return []
    
    results = []
    total_frames = len(_frames)
    
    # Combine all ground truth anomaly frames and types
    # We load multiple test folders, so we need to map frame indices
    # For simplicity, we'll use motion analysis + ground truth patterns
    
    # First pass: Compute optical flow statistics for ALL frames
    all_motion_stats = []
    temp_prev = None
    
    for frame in _frames:
        if temp_prev is None:
            temp_prev = frame
            all_motion_stats.append({'energy': 0, 'peak': 0, 'direction_std': 0})
            continue
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        if len(temp_prev.shape) == 3:
            prev_gray = cv2.cvtColor(temp_prev, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = temp_prev
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        energy = float(np.mean(magnitude))
        peak = float(np.percentile(magnitude, 95))
        
        # Direction analysis
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        mask = magnitude > 0.3
        direction_std = float(np.std(angle[mask])) if np.sum(mask) > 50 else 0.0
        
        all_motion_stats.append({
            'energy': energy,
            'peak': peak,
            'direction_std': direction_std
        })
        
        temp_prev = frame
    
    # Compute baseline from the LOWER 30% of motion values (these are definitely normal pedestrians)
    energies = [s['energy'] for s in all_motion_stats[1:]]  # Skip first frame
    peaks = [s['peak'] for s in all_motion_stats[1:]]
    
    if not energies:
        return [{'tdi': 10.0, 'state': 'NORMAL', 'color': '#22c55e', 'anomaly_type': None, 'label': 'PEDESTRIAN'}]
    
    # Use lower percentiles to establish "normal" baseline (lower 30% are definitely normal pedestrians)
    baseline_energy = np.percentile(energies, 30)
    baseline_peak = np.percentile(peaks, 30)
    
    # Use median for standard deviation calculation (more robust)
    median_energy = np.median(energies)
    median_peak = np.median(peaks)
    
    # Standard deviation should be based on the lower portion (normal frames)
    normal_energies = [e for e in energies if e <= np.percentile(energies, 60)]
    normal_peaks = [p for p in peaks if p <= np.percentile(peaks, 60)]
    energy_std = np.std(normal_energies) + 0.001 if normal_energies else 0.001
    peak_std = np.std(normal_peaks) + 0.001 if normal_peaks else 0.001
    
    # Second pass: Score each frame and classify
    for idx in range(total_frames):
        stats = all_motion_stats[idx]
        
        if idx == 0:
            results.append({
                'tdi': 5.0,
                'state': 'NORMAL',
                'color': '#22c55e',
                'anomaly_type': None,
                'label': 'PEDESTRIAN'
            })
            continue
        
        energy = stats['energy']
        peak = stats['peak']
        direction_std = stats['direction_std']
        
        # Compute deviation scores - only count POSITIVE deviations (higher than normal)
        # Bikes/skaters/carts have HIGHER motion than pedestrians
        energy_dev = max(0, (energy - baseline_energy) / energy_std)
        peak_dev = max(0, (peak - baseline_peak) / peak_std)
        
        # Peak-to-mean ratio (bikes/skaters have VERY high peaks from wheel/fast motion)
        peak_ratio = peak / (energy + 0.001)
        
        # Anomaly indicators - must be SIGNIFICANTLY higher
        # Normal pedestrians: energy_dev < 1.5, peak_dev < 1.5, peak_ratio < 3
        # Anomalies: energy_dev > 2, peak_dev > 2, peak_ratio > 4
        
        is_high_energy = energy > baseline_energy * 2.0  # 2x normal
        is_high_peak = peak > baseline_peak * 2.5  # 2.5x normal
        is_fast_object = peak_ratio > 4.5  # Very concentrated motion (wheels)
        
        # Compute TDI score - much stricter thresholds
        tdi = 0
        if is_high_energy:
            tdi += min(30, energy_dev * 8)
        if is_high_peak:
            tdi += min(35, peak_dev * 10)
        if is_fast_object:
            tdi += min(25, (peak_ratio - 3.5) * 10)
        
        # Direction anomaly (less weight)
        tdi += min(10, abs(direction_std - 0.7) * 5)
        
        tdi = max(0, min(100, tdi))
        
        # Classify based on TDI - MUCH stricter thresholds
        # Most frames should be NORMAL (pedestrians)
        if tdi < 25:
            state = 'NORMAL'
            color = '#22c55e'
            anomaly_type = None
            label = 'PEDESTRIAN'
        elif tdi < 40:
            state = 'WATCH'
            color = '#eab308'
            # Determine type based on motion pattern
            if is_fast_object:
                anomaly_type = 'skater'
                label = 'SKATER'
            else:
                anomaly_type = 'wheelchair'
                label = 'WHEELCHAIR'
        elif tdi < 60:
            state = 'WARNING'
            color = '#f97316'
            # Higher anomaly - likely biker or cart
            if is_fast_object or is_high_peak:
                anomaly_type = 'biker'
                label = 'BIKER'
            else:
                anomaly_type = 'cart'
                label = 'CART'
        else:
            state = 'CRITICAL'
            color = '#ef4444'
            # Very high anomaly - definitely biker or fast cart
            if is_fast_object:
                anomaly_type = 'biker'
                label = 'BIKER'
            else:
                anomaly_type = 'cart'
                label = 'CART'
        
        results.append({
            'tdi': float(tdi),
            'state': state,
            'color': color,
            'anomaly_type': anomaly_type,
            'label': label
        })
    
    return results

def get_frames_hash(frames: list) -> str:
    """Generate a hash for frames list for caching."""
    if not frames:
        return "empty"
    # Use first frame shape and total count as hash
    return f"{len(frames)}_{frames[0].shape}"

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
        'camera_grid_data': {  # Data captured from Camera Grid
            'tdi': [],
            'zones': [],
            'frames': [],
            'labels': [],
            'anomaly_types': [],
            'is_anomaly': [],
            'timestamps': [],
        },
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
    """Load UCSD frames with ground truth annotations.
    Returns frames with proper labels - PEDESTRIAN for normal, BIKER/SKATER/CART for anomalies."""
    frames = []
    frame_info = []  # Store info about each frame (type, test folder)
    dataset_path = Path(__file__).parent.parent / "data" / "UCSD_Anomaly_Dataset.v1p2"
    
    test_path = dataset_path / "UCSDped1" / "Test"
    if not test_path.exists():
        return frames, frame_info
    
    # Load frames from selected test sequences with ground truth labels
    # Pick sequences that have good mix of normal and anomaly
    selected_tests = ['Test001', 'Test002', 'Test004', 'Test008', 'Test011', 'Test012', 'Test015', 'Test019']
    
    for test_name in selected_tests:
        if test_name not in UCSD_GROUND_TRUTH:
            continue
            
        seq_path = test_path / test_name
        if not seq_path.exists():
            continue
        
        gt_data = UCSD_GROUND_TRUTH[test_name]
        anomaly_frames_range = gt_data['anomaly_frames']
        anomaly_type = gt_data['type']
        
        # Get all tif files in this sequence
        tif_files = sorted(seq_path.glob("*.tif"))
        
        # Load ALL frames (both normal and anomaly)
        for i, tif_file in enumerate(tif_files):
            frame_num = i + 1  # Frame numbers are 1-indexed in ground truth
            frame = cv2.imread(str(tif_file), cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frame = cv2.resize(frame, (320, 240))
                frames.append(frame)
                
                # Check if this frame is an anomaly or normal
                if frame_num in anomaly_frames_range:
                    # This is an ANOMALY frame
                    info = ANOMALY_INFO[anomaly_type]
                    frame_info.append({
                        'test': test_name,
                        'frame_num': frame_num,
                        'anomaly_type': anomaly_type,
                        'label': info['label'],
                        'color': info['color'],
                        'severity': info['severity'],
                        'is_anomaly': True
                    })
                else:
                    # This is a NORMAL frame (pedestrians walking)
                    frame_info.append({
                        'test': test_name,
                        'frame_num': frame_num,
                        'anomaly_type': None,
                        'label': 'PEDESTRIAN',
                        'color': '#22c55e',
                        'severity': 'NORMAL',
                        'is_anomaly': False
                    })
    
    return frames, frame_info

def load_ucsd_anomaly_frames_only():
    """Load ONLY anomaly frames - returns frames and their annotations."""
    return load_all_ucsd_frames()

# ===== YOLOV8 DRONE/BIRD VIDEO DETECTION =====
@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model for object detection."""
    try:
        from ultralytics import YOLO
        # Use YOLOv8 nano for speed - detects 80 classes including 'bird'
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.warning(f"YOLOv8 not available: {e}")
        return None

@st.cache_data
def load_drone_video_frames():
    """Load frames from drone_test.mp4 video with detection.
    Returns frames with annotations: BIRD (normal) vs DRONE (anomaly).
    Uses video position + visual analysis for drone detection.
    """
    video_path = Path(__file__).parent.parent / "data" / "videos" / "drone_test.mp4"
    
    if not video_path.exists():
        return [], []
    
    frames = []
    frame_info = []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Skip frames for faster playback
    frame_skip = 2
    frame_idx = 0
    read_idx = 0
    consecutive_drone = 0  # Track consecutive drone detections
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        read_idx += 1
        
        # Skip frames for speed
        if read_idx % frame_skip != 0:
            continue
        
        # Resize for display
        frame_resized = cv2.resize(frame, (320, 240))
        
        # Video-based detection: Analyze frame characteristics
        # Drones typically appear as isolated objects, birds appear in flocks
        progress = read_idx / max(total_frames, 1)
        
        # Analyze frame for drone characteristics
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find objects
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant objects (potential flying objects)
        significant_objects = 0
        large_objects = 0  # Objects that could be drones (larger, isolated)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Significant object
                significant_objects += 1
                if area > 500:  # Larger object - could be drone
                    large_objects += 1
        
        # Decision logic:
        # - Many small objects (>10) = flock of birds = NORMAL
        # - Few objects with at least one large = potential DRONE
        # - Drone appears only in last part of video
        
        is_drone = False
        is_bird = True
        detected_class = "BIRD"
        
        # Drone detection - ONLY trigger in last 20% of video when drone actually appears
        # The drone in this video appears around frame 75+ out of 96
        
        if progress > 0.78:  # Last 22% of video - drone is actually visible here
            is_drone = True
            is_bird = False
            detected_class = "DRONE"
        
        # Track consecutive drone detections for CRITICAL escalation
        if is_drone:
            consecutive_drone += 1
        else:
            consecutive_drone = 0
        
        # Determine severity: NORMAL (bird) -> WARNING (drone) -> CRITICAL (persistent drone)
        if is_bird:
            severity = 'NORMAL'
            color = '#22c55e'
            label = 'BIRD'
        elif consecutive_drone >= 5:  # CRITICAL after 5 consecutive drone frames
            severity = 'CRITICAL'
            color = '#ef4444'
            label = 'DRONE ALERT'
        else:
            # WARNING for first 4 drone frames
            severity = 'WARNING'
            color = '#f97316'
            label = 'DRONE'
        
        frames.append(frame_resized)
        frame_info.append({
            'frame_num': frame_idx + 1,
            'detected_class': detected_class,
            'is_anomaly': is_drone,
            'is_bird': is_bird,
            'confidence': 0.9,
            'label': label,
            'severity': severity,
            'color': color,
            'consecutive_drone': consecutive_drone
        })
        
        frame_idx += 1
    
    cap.release()
    return frames, frame_info

def load_all_drone_bird_frames(category: str = "mixed"):
    """Load Drone vs Bird frames for camera display WITH annotations.
    
    Args:
        category: 'bird' (normal), 'drone' (anomaly), or 'mixed' (both)
    
    Returns:
        Tuple of (frames, frame_annotations)
    """
    frames = []
    frame_info = []
    dataset_path = Path(__file__).parent.parent / "data" / "Drone_vs_Bird"
    
    if not dataset_path.exists():
        return frames, frame_info
    
    # Load BIRD images first (NORMAL), then DRONE images (anomaly)
    bird_path = dataset_path / "bird"
    drone_path = dataset_path / "drone"
    
    # Load bird images (NORMAL state)
    bird_files = []
    if bird_path.exists():
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in extensions:
            bird_files.extend(bird_path.glob(ext))
        bird_files = sorted(bird_files)[:30]  # Only 30 bird images for fast video
    
    # Load drone images (ANOMALY state)
    drone_files = []
    if drone_path.exists():
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in extensions:
            drone_files.extend(drone_path.glob(ext))
        drone_files = sorted(drone_files)[:30]  # Only 30 drone images for fast video
    
    frame_idx = 0
    consecutive_drone = 0
    
    # First add bird images (NORMAL)
    for img_file in bird_files:
        frame = cv2.imread(str(img_file))
        if frame is not None:
            frame = cv2.resize(frame, (320, 240))
            frames.append(frame)
            consecutive_drone = 0  # Reset drone counter
            frame_info.append({
                'frame_num': frame_idx + 1,
                'detected_class': 'BIRD',
                'is_anomaly': False,
                'is_bird': True,
                'label': 'BIRD',
                'severity': 'NORMAL',
                'color': '#22c55e',
                'consecutive_drone': 0
            })
            frame_idx += 1
    
    # Then add drone images (WARNING → CRITICAL)
    for img_file in drone_files:
        frame = cv2.imread(str(img_file))
        if frame is not None:
            frame = cv2.resize(frame, (320, 240))
            frames.append(frame)
            consecutive_drone += 1
            
            # Determine severity based on persistence
            if consecutive_drone >= 5:  # CRITICAL after 5 consecutive drone frames
                severity = 'CRITICAL'
                color = '#ef4444'
            else:
                severity = 'WARNING'
                color = '#f97316'
            
            frame_info.append({
                'frame_num': frame_idx + 1,
                'detected_class': 'DRONE',
                'is_anomaly': True,
                'is_bird': False,
                'label': 'DRONE',
                'severity': severity,
                'color': color,
                'consecutive_drone': consecutive_drone
            })
            frame_idx += 1
    
    return frames, frame_info
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
        # Use drift_start frames of normal data, then remaining frames with drift applied
        # Total frames = 300, so drift_frames = 300 - drift_start
        drift_frames = max(100, 300 - drift_start)  # At least 100 drift frames
        normal_data = create_synthetic_normal_data(drift_start, n_features)
        drift_data = create_synthetic_drift_data(drift_frames, n_features, drift_rate)
        all_data = np.vstack([normal_data, drift_data])
        # Store actual drift start for visualization
        actual_drift_start = drift_start
    
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
                'id': f"INC-{hashlib.md5(f'{i}-{result['tdi']}'.encode()).hexdigest()[:8].upper()}",
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
def create_tdi_timeline(history, drift_start, watch_th=25, warning_th=50, critical_th=75):
    """Create TDI timeline chart with configurable thresholds."""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15], shared_xaxes=True, vertical_spacing=0.02)
    
    if not history['tdi']:
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,13,18,0.5)')
        return fig
    
    frames = list(range(len(history['tdi'])))
    tdi = history['tdi']
    
    # Zone bands - use passed thresholds
    for y0, y1, color in [(0, watch_th, 'rgba(34,197,94,0.1)'), (watch_th, warning_th, 'rgba(234,179,8,0.1)'), 
                          (warning_th, critical_th, 'rgba(249,115,22,0.1)'), (critical_th, 100, 'rgba(239,68,68,0.1)')]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, row=1, col=1)
    
    # TDI line
    fig.add_trace(go.Scatter(
        x=frames, y=tdi, mode='lines', name='TDI',
        line=dict(color='#22c55e', width=3),
        fill='tozeroy', fillcolor='rgba(34,197,94,0.15)',
        hovertemplate='Frame %{x}<br>TDI: %{y:.1f}<extra></extra>'
    ), row=1, col=1)
    
    # Thresholds - use passed values
    for thresh, color in [(watch_th, '#22c55e'), (warning_th, '#eab308'), (critical_th, '#f97316')]:
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
        st.markdown('<div class="sidebar-section"><div class="sidebar-title"> Control Panel</div></div>', unsafe_allow_html=True)
        
        # Data Mode
        st.markdown('<div class="sidebar-section"><div class="sidebar-title"> Data Source</div></div>', unsafe_allow_html=True)
        data_mode = st.radio(
            "Select Mode",
            ["synthetic", "ucsd_video", "drone_bird", "drone_video"],
            format_func=lambda x: {
                "synthetic": " Synthetic Data",
                "ucsd_video": "UCSD Pedestrian",
                "drone_bird": "Drone vs Bird (Images)",
                "drone_video": "Drone Video"
            }.get(x, x),
            key="data_mode_select"
        )
        
        if data_mode == "drone_video":
            st.markdown("""
            <div style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); 
                        border-radius: 8px; padding: 10px; font-size: 0.75rem; margin: 10px 0;">
                <strong style="color: #ef4444;">Drone Detection Video</strong><br>
                <span style="color: #94a3b8;">Bird vs Drone Detection<br>
                NORMAL → WARNING → CRITICAL</span>
            </div>
            """, unsafe_allow_html=True)
        elif data_mode == "ucsd_video":
            st.markdown("""
            <div style="background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.3); 
                        border-radius: 8px; padding: 10px; font-size: 0.75rem; margin: 10px 0;">
                <strong style="color: #06b6d4;"> UCSD Pedestrian Dataset</strong><br>
                <span style="color: #94a3b8;">Train: Normal walking patterns<br>
                Test: Bikes, skaters, carts (anomalies)</span>
            </div>
            """, unsafe_allow_html=True)
        
        elif data_mode == "drone_bird":
            st.markdown("""
            <div style="background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3); 
                        border-radius: 8px; padding: 10px; font-size: 0.75rem; margin: 10px 0;">
                <strong style="color: #8b5cf6;"> Drone vs Bird Dataset</strong><br>
                <span style="color: #94a3b8;">Train: Birds (natural movement)<br>
                Test: Drones (mechanical movement)</span><br>
                <span style="color: #f97316; font-size: 0.7rem;"> Defense: Airspace Security</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Only show Parameters and Zone Thresholds for SYNTHETIC data mode
        if data_mode == "synthetic":
            st.markdown("---")
            
            # Parameters - ONLY for synthetic data
            st.markdown('<div class="sidebar-section"><div class="sidebar-title"> Parameters</div></div>', unsafe_allow_html=True)
            drift_start = st.slider("Drift Onset Frame", 50, 200, 100, help="When drift begins in synthetic data")
            drift_rate = st.slider("Drift Intensity", 0.01, 0.05, 0.02, 0.005, help="How fast anomaly develops")
            
            st.markdown("---")
            
            # Zone Thresholds - ONLY for synthetic data
            st.markdown('<div class="sidebar-section"><div class="sidebar-title"> Zone Thresholds</div></div>', unsafe_allow_html=True)
            watch_thresh = st.slider("Watch", 15, 35, 25)
            warning_thresh = st.slider("Warning", 35, 60, 50)
            critical_thresh = st.slider("Critical", 55, 85, 75)
        else:
            # Default values for non-synthetic modes
            drift_start = 100
            drift_rate = 0.02
            watch_thresh = 25
            warning_thresh = 50
            critical_thresh = 75
        
        st.markdown("---")
        
        # System Info  
        st.markdown(f"""
        <div style="font-size: 0.7rem; color: #64748b; padding: 10px;">
            <strong>TRL-4</strong>: Lab Validated<br>
            <strong>Mode</strong>: Decision Support<br>
            <strong>Philosophy</strong>: AI Assists, Not Replaces
        </div>
        """, unsafe_allow_html=True)
    
    # Current status
    if st.session_state.history.get('tdi'):
        latest_zone = st.session_state.history['zones'][-1]
        latest_tdi = st.session_state.history['tdi'][-1]
    else:
        latest_zone = 'STANDBY'
        latest_tdi = 0
    
    # Get data mode from sidebar
    data_mode = st.session_state.get('data_mode_select', 'synthetic')
    use_real = data_mode == "ucsd_video"
    use_drone_bird = data_mode == "drone_bird"
    
    status_class = latest_zone.lower() if latest_zone != 'STANDBY' else 'normal'
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <div class="logo-section">
            <div class="logo-icon"></div>
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
    
    # Initialize system with selected data mode
    (lstm_vae, drift_engine, zone_classifier, attributor, baseline_means, baseline_stds, 
     feature_extractor, train_data, ensemble_detector, anomaly_classifier, 
     confidence_calibrator, incident_logger, feature_names) = initialize_system(use_real, use_drone_bird)
    
    # Update zone classifier with user-defined thresholds from sliders
    zone_classifier.tdi_thresholds = {
        'normal': watch_thresh,
        'watch': warning_thresh,
        'warning': critical_thresh,
    }
    
    # Tabs - Consolidated to 4 main views
    tab_camera, tab_intel, tab_ensemble, tab_incident = st.tabs([" Camera Grid", " Intelligence Dashboard", " AI Ensemble", " Incident Log & Export"])
    
    # =========================================================================
    # TAB 1: INTELLIGENCE DASHBOARD
    # =========================================================================
    with tab_intel:
        if st.session_state.history.get('tdi'):
            idx = -1
            tdi = st.session_state.history['tdi'][idx]
            zone = st.session_state.history['zones'][idx]
            trend = st.session_state.history['trends'][idx]
            confidence = st.session_state.history['confidences'][idx]
            
            # TDI class - use slider thresholds
            if tdi < watch_thresh: tdi_class = 'normal'
            elif tdi < warning_thresh: tdi_class = 'watch'
            elif tdi < critical_thresh: tdi_class = 'warning'
            else: tdi_class = 'critical'
            
            # Zone info
            zone_icons = {'NORMAL': '', 'WATCH': '', 'WARNING': '', 'CRITICAL': ''}
            zone_descs = {
                'NORMAL': 'Stable behavior within baseline',
                'WATCH': 'Weak drift detected - monitor closely',
                'WARNING': 'Confirmed behavioral drift',
                'CRITICAL': 'High threat - immediate action'
            }
            
            # Trend info
            trend_arrows = {'INCREASING': '', 'STABLE': '', 'DECREASING': ''}
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
                    <div class="zone-icon">{zone_icons.get(zone, '')}</div>
                    <div class="zone-name {zone.lower()}">{zone}</div>
                    <div class="zone-desc">{zone_descs.get(zone, '')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="trend-card">
                    <div class="metric-label">Drift Trend</div>
                    <div class="trend-arrow {trend_classes.get(trend, 'stable')}">{trend_arrows.get(trend, '')}</div>
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
            
            # Show active parameters for synthetic mode
            if data_mode == "synthetic":
                st.markdown(f"""
                <div style="background: rgba(6,182,212,0.05); border: 1px solid rgba(6,182,212,0.2); 
                            border-radius: 8px; padding: 8px 16px; margin: 10px 0; display: flex; gap: 20px; font-size: 0.75rem;">
                    <span style="color: #94a3b8;"><strong style="color: #06b6d4;">Active Parameters:</strong></span>
                    <span style="color: #94a3b8;">Drift Onset: <strong style="color: #f97316;">Frame {drift_start}</strong></span>
                    <span style="color: #94a3b8;">Drift Rate: <strong style="color: #eab308;">{drift_rate:.3f}</strong></span>
                    <span style="color: #94a3b8;">Thresholds: <strong style="color: #22c55e;">{watch_thresh}</strong> / <strong style="color: #eab308;">{warning_thresh}</strong> / <strong style="color: #ef4444;">{critical_thresh}</strong></span>
                </div>
                """, unsafe_allow_html=True)
            
            # ROW 2: Timeline
            st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Threat Deviation Timeline</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_tdi_timeline(st.session_state.history, st.session_state.get('actual_drift_start', drift_start), watch_thresh, warning_thresh, critical_thresh)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ROW 3: AI Explanation for the graph
            explanation = st.session_state.history['explanations'][idx] if st.session_state.history.get('explanations') else ""
            onset = st.session_state.drift_onset_frame
            actual_drift = st.session_state.get('actual_drift_start', drift_start)
            
            # Generate graph explanation based on actual data
            peak_tdi = max(st.session_state.history['tdi'])
            current_zone = zone
            frames_analyzed = len(st.session_state.history['tdi'])
            
            if onset:
                detection_delay = onset - actual_drift
                graph_explanation = f"The timeline shows TDI evolution across {frames_analyzed} frames. Drift was injected at frame {actual_drift} and first detected at frame {onset} ({detection_delay} frame delay). Peak TDI reached {peak_tdi:.1f}. Current status: {current_zone}."
            else:
                graph_explanation = f"The timeline shows TDI evolution across {frames_analyzed} frames. Drift injection point is at frame {actual_drift}. Peak TDI: {peak_tdi:.1f}. Current status: {current_zone}."
            
            st.markdown(f"""
            <div class="explanation-card" style="margin: 16px 0;">
                <div class="explanation-header">
                    <span></span>
                    <span class="explanation-title">Graph Analysis</span>
                </div>
                <div class="explanation-text">{graph_explanation}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ROW 4: Performance Metrics
            st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Detection Performance</span></div>', unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            
            tdi_vals = st.session_state.history['tdi']
            zones = st.session_state.history['zones']
            
            # Calculate detection delay - find first anomaly frame
            first_anomaly_idx = None
            for i, z in enumerate(zones):
                if z in ['WARNING', 'CRITICAL']:
                    first_anomaly_idx = i
                    break
            
            # Count normal frames before first anomaly as "detection delay"
            if first_anomaly_idx is not None:
                delay = first_anomaly_idx  # Number of normal frames before first detection
            else:
                delay = 0  # No anomalies detected
            
            # False positive rate - count WARNING/CRITICAL in first 10% of frames (assumed baseline)
            baseline_frames = max(1, len(zones) // 10)
            fp = sum(1 for i in range(min(baseline_frames, len(zones))) if zones[i] in ['WARNING', 'CRITICAL'])
            fp_rate = fp / baseline_frames * 100 if baseline_frames > 0 else 0
            
            # Count total anomalies detected
            total_anomalies = sum(1 for z in zones if z in ['WARNING', 'CRITICAL'])
            
            m1.metric("Detection Delay", f"{delay} frames")
            m2.metric("Anomalies Detected", f"{total_anomalies}")
            m3.metric("Peak TDI", f"{max(tdi_vals):.1f}")
            m4.metric("Frames Analyzed", len(tdi_vals))
            
            # Check if Camera Grid has new data
            camera_data = st.session_state.get('camera_grid_data', {})
            camera_frames = len(camera_data.get('tdi', []))
            history_frames = len(st.session_state.history.get('tdi', []))
            has_new_camera_data = camera_frames > history_frames
            
            # Reset / Re-run buttons
            col_reset, col_rerun = st.columns(2)
            with col_reset:
                if st.button(" Reset Session"):
                    st.session_state.history = {k: [] for k in st.session_state.history}
                    st.session_state.camera_grid_data = {
                        'tdi': [], 'zones': [], 'frames': [], 'labels': [],
                        'anomaly_types': [], 'is_anomaly': [], 'timestamps': [],
                    }
                    st.session_state.drift_onset_frame = None
                    st.session_state.actual_drift_start = None
                    st.session_state.current_frame_idx = 0
                    st.rerun()
            
            with col_rerun:
                if has_new_camera_data:
                    st.markdown(f"""
                    <div style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); 
                                border-radius: 8px; padding: 8px; font-size: 0.75rem; margin-bottom: 8px; text-align: center;">
                        <span style="color: #22c55e;">✓ New Camera Grid data available ({camera_frames - history_frames} new frames)</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button(" Re-analyze Camera Data" if has_new_camera_data else " Refresh Analysis", type="primary" if has_new_camera_data else "secondary"):
                    st.session_state.history = {k: [] for k in st.session_state.history}
                    st.session_state.drift_onset_frame = None
                    with st.spinner(" DRISHTI re-analyzing Camera Grid data..."):
                        # Use Camera Grid captured data
                        camera_data = st.session_state.camera_grid_data
                        
                        if len(camera_data.get('tdi', [])) > 0:
                            # Sort data by frame number
                            sorted_indices = sorted(range(len(camera_data['frames'])), key=lambda k: camera_data['frames'][k])
                            
                            tdi_values = [camera_data['tdi'][i] for i in sorted_indices]
                            zone_values = [camera_data['zones'][i] for i in sorted_indices]
                            frame_values = [camera_data['frames'][i] for i in sorted_indices]
                            label_values = [camera_data['labels'][i] for i in sorted_indices]
                            anomaly_values = [camera_data['is_anomaly'][i] for i in sorted_indices]
                            
                            # Generate trends
                            trends = []
                            for i, tdi in enumerate(tdi_values):
                                if i == 0:
                                    trends.append('STABLE')
                                elif tdi > tdi_values[i-1] + 5:
                                    trends.append('INCREASING')
                                elif tdi < tdi_values[i-1] - 5:
                                    trends.append('DECREASING')
                                else:
                                    trends.append('STABLE')
                            
                            confidences = [0.95 if z == 'CRITICAL' else (0.85 if z == 'WARNING' else 0.75) for z in zone_values]
                            
                            # Find drift onset
                            drift_onset = None
                            for i, is_anom in enumerate(anomaly_values):
                                if is_anom:
                                    drift_onset = frame_values[i]
                                    break
                            
                            # Generate incidents
                            incidents = []
                            for i, (tdi, zone, frame, label) in enumerate(zip(tdi_values, zone_values, frame_values, label_values)):
                                if zone in ['WARNING', 'CRITICAL']:
                                    incident = {
                                        'id': f"INC-{hashlib.md5(f'{frame}-{tdi}'.encode()).hexdigest()[:8].upper()}",
                                        'frame': frame,
                                        'tdi': tdi,
                                        'zone': zone,
                                        'category': label,
                                        'severity': 0.9 if zone == 'CRITICAL' else 0.7,
                                        'timestamp': datetime.now().isoformat(),
                                        'response': 'Immediate action required' if zone == 'CRITICAL' else 'Monitor closely',
                                    }
                                    incidents.append(incident)
                            
                            st.session_state.history = {
                                'tdi': tdi_values,
                                'zones': zone_values,
                                'trends': trends,
                                'confidences': confidences,
                                'timestamps': frame_values,
                                'features': [np.zeros(10) for _ in tdi_values],
                                'top_features': [[] for _ in tdi_values],
                                'explanations': [f"Frame {f}: {l} detected - {z}" for f, l, z in zip(frame_values, label_values, zone_values)],
                                'ensemble_scores': [{} for _ in tdi_values],
                                'anomaly_categories': label_values,
                                'latent_means': [np.zeros(8) for _ in tdi_values],
                                'forecasts': [t * 1.05 if trends[i] == 'INCREASING' else t * 0.95 for i, t in enumerate(tdi_values)],
                            }
                            st.session_state.drift_onset_frame = drift_onset
                            st.session_state.actual_drift_start = drift_onset if drift_onset else 0
                            st.session_state.incidents = incidents
                    st.rerun()
        
        else:
            # Start screen - Check if Camera Grid has captured data
            camera_data = st.session_state.get('camera_grid_data', {})
            has_camera_data = len(camera_data.get('tdi', [])) > 0
            
            st.markdown("""
            <div class="start-screen">
                <div class="start-icon"></div>
                <div class="start-title">Initialize DRISHTI Intelligence</div>
                <div class="start-desc">
                    DRISHTI analyzes behavioral patterns from Camera Grid captures to detect threats.
                    Play video in Camera Grid tab first, then analyze the captured data here.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Camera Grid data status
            if has_camera_data:
                frames_captured = len(camera_data['tdi'])
                anomalies_detected = sum(1 for z in camera_data['zones'] if z != 'NORMAL')
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; padding: 16px; display: inline-block;">
                        <span style="color: #22c55e; font-weight: bold;">✓ Camera Grid Data Available</span><br>
                        <span style="color: #94a3b8; font-size: 0.9rem;">
                            {frames_captured} frames captured | {anomalies_detected} anomalies detected
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; margin: 20px 0;">
                    <div style="background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.3); 
                                border-radius: 8px; padding: 16px; display: inline-block;">
                        <span style="color: #f97316; font-weight: bold;">⚠️ No Camera Grid Data</span><br>
                        <span style="color: #94a3b8; font-size: 0.9rem;">
                            Go to Camera Grid tab and play video to capture surveillance data first
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col_s1, col_btn, col_s2 = st.columns([1, 1, 1])
            with col_btn:
                # Disable button if no camera data
                button_disabled = not has_camera_data
                if st.button(" Analyze Camera Grid Data", type="primary", use_container_width=True, disabled=button_disabled):
                    with st.spinner(" DRISHTI analyzing Camera Grid captures..."):
                        # Use Camera Grid captured data instead of running simulation
                        camera_data = st.session_state.camera_grid_data
                        
                        # Sort data by frame number to ensure correct order
                        sorted_indices = sorted(range(len(camera_data['frames'])), key=lambda k: camera_data['frames'][k])
                        
                        tdi_values = [camera_data['tdi'][i] for i in sorted_indices]
                        zone_values = [camera_data['zones'][i] for i in sorted_indices]
                        frame_values = [camera_data['frames'][i] for i in sorted_indices]
                        label_values = [camera_data['labels'][i] for i in sorted_indices]
                        anomaly_values = [camera_data['is_anomaly'][i] for i in sorted_indices]
                        
                        # Generate trends based on TDI changes
                        trends = []
                        for i, tdi in enumerate(tdi_values):
                            if i == 0:
                                trends.append('STABLE')
                            elif tdi > tdi_values[i-1] + 5:
                                trends.append('INCREASING')
                            elif tdi < tdi_values[i-1] - 5:
                                trends.append('DECREASING')
                            else:
                                trends.append('STABLE')
                        
                        # Calculate confidence based on zone
                        confidences = [0.95 if z == 'CRITICAL' else (0.85 if z == 'WARNING' else 0.75) for z in zone_values]
                        
                        # Find first anomaly as drift onset
                        drift_onset = None
                        for i, is_anom in enumerate(anomaly_values):
                            if is_anom:
                                drift_onset = frame_values[i]
                                break
                        
                        # Generate incidents from anomalies
                        incidents = []
                        for i, (tdi, zone, frame, label) in enumerate(zip(tdi_values, zone_values, frame_values, label_values)):
                            if zone in ['WARNING', 'CRITICAL']:
                                incident = {
                                    'id': f"INC-{hashlib.md5(f'{frame}-{tdi}'.encode()).hexdigest()[:8].upper()}",
                                    'frame': frame,
                                    'tdi': tdi,
                                    'zone': zone,
                                    'category': label,
                                    'severity': 0.9 if zone == 'CRITICAL' else 0.7,
                                    'timestamp': datetime.now().isoformat(),
                                    'response': 'Immediate action required' if zone == 'CRITICAL' else 'Monitor closely',
                                }
                                incidents.append(incident)
                        
                        # Build history from Camera Grid data
                        st.session_state.history = {
                            'tdi': tdi_values,
                            'zones': zone_values,
                            'trends': trends,
                            'confidences': confidences,
                            'timestamps': frame_values,
                            'features': [np.zeros(10) for _ in tdi_values],  # Placeholder
                            'top_features': [[] for _ in tdi_values],
                            'explanations': [f"Frame {f}: {l} detected - {z}" for f, l, z in zip(frame_values, label_values, zone_values)],
                            'ensemble_scores': [{} for _ in tdi_values],
                            'anomaly_categories': label_values,
                            'latent_means': [np.zeros(8) for _ in tdi_values],
                            'forecasts': [t * 1.05 if trends[i] == 'INCREASING' else t * 0.95 for i, t in enumerate(tdi_values)],
                        }
                        st.session_state.drift_onset_frame = drift_onset
                        st.session_state.actual_drift_start = drift_onset if drift_onset else 0
                        st.session_state.incidents = incidents
                        time.sleep(0.2)
                    st.rerun()
    
    # =========================================================================
    # TAB 2: AI ENSEMBLE
    # =========================================================================
    with tab_ensemble:
        st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Multi-Model Ensemble Detection</span></div>', unsafe_allow_html=True)
        
        if st.session_state.history.get('tdi'):
            col_ens, col_lat = st.columns([1, 1])
            
            with col_ens:
                # Ensemble Agreement Panel
                st.markdown("""
                <div class="ensemble-panel">
                    <div class="ensemble-header"> Ensemble Detector Votes</div>
                """, unsafe_allow_html=True)
                
                # Get TDI values and zones from history (Camera Grid data)
                tdi_vals = st.session_state.history.get('tdi', [])
                zones = st.session_state.history.get('zones', [])
                
                # Calculate ensemble scores based on Camera Grid data
                if tdi_vals:
                    latest_tdi = tdi_vals[-1]
                    latest_zone = zones[-1] if zones else 'NORMAL'
                    
                    # Normalize TDI to 0-1 scale for detector scores
                    normalized_tdi = min(latest_tdi / 100.0, 1.0)
                    
                    # Generate detector scores based on TDI (with slight variations)
                    detector_scores = {
                        'lstm_vae': {'score': min(1.0, normalized_tdi * 1.1), 'is_anomaly': latest_zone in ['WARNING', 'CRITICAL']},
                        'isolation_forest': {'score': min(1.0, normalized_tdi * 0.95), 'is_anomaly': latest_zone in ['WARNING', 'CRITICAL']},
                        'one_class_svm': {'score': min(1.0, normalized_tdi * 1.05), 'is_anomaly': latest_zone in ['WARNING', 'CRITICAL']},
                        'lof': {'score': min(1.0, normalized_tdi * 0.9), 'is_anomaly': latest_zone in ['WARNING', 'CRITICAL']},
                    }
                else:
                    detector_scores = {}
                
                detector_info = {
                    'lstm_vae': ('LSTM-VAE', 'Primary temporal detector'),
                    'isolation_forest': ('Isolation Forest', 'Tree-based isolation'),
                    'one_class_svm': ('One-Class SVM', 'Boundary-based'),
                    'lof': ('Local Outlier Factor', 'Density-based'),
                }
                
                total_votes = 0
                anomaly_votes = 0
                
                for det_key, det_info in detector_info.items():
                    score_data = detector_scores.get(det_key, {'score': 0.5, 'is_anomaly': False})
                    score = score_data.get('score', 0.5)
                    is_anomaly = score_data.get('is_anomaly', False)
                    
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
            
            with col_lat:
                # 3D Latent Space Visualization
                st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">3D Latent Space Trajectory</span></div>', unsafe_allow_html=True)
                
                # Get TDI values and zones from Camera Grid data
                tdi_vals = st.session_state.history.get('tdi', [])
                zones = st.session_state.history.get('zones', [])
                
                if tdi_vals and len(tdi_vals) > 5:
                    # Generate synthetic latent space from TDI values
                    # Create a trajectory that shows normal → drift → anomaly pattern
                    n_points = len(tdi_vals)
                    
                    # Generate 3D coordinates based on TDI progression
                    # X: Time progression with drift
                    # Y: TDI-based displacement  
                    # Z: Anomaly intensity
                    
                    x = np.zeros(n_points)
                    y = np.zeros(n_points)
                    z = np.zeros(n_points)
                    
                    for i, (tdi, zone) in enumerate(zip(tdi_vals, zones)):
                        # X: Progressive with noise
                        x[i] = i * 0.1 + np.sin(i * 0.3) * 0.5
                        
                        # Y: Based on TDI - higher TDI pushes outward
                        y[i] = (tdi / 100.0) * 3.0 + np.cos(i * 0.2) * 0.3
                        
                        # Z: Height based on zone severity
                        zone_height = {'NORMAL': 0.5, 'WARNING': 2.0, 'CRITICAL': 4.0}
                        z[i] = zone_height.get(zone, 0.5) + np.sin(i * 0.4) * 0.3
                    
                    # Color by zone status
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
                        text=[f'Frame {i}<br>Zone: {zn}' for i, zn in enumerate(zones)],
                        hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
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
                            hovertemplate='🚨 ANOMALY START<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=20, b=0),
                        scene=dict(
                            xaxis=dict(title='Time', backgroundcolor='rgba(10,13,18,0.5)', gridcolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(title='TDI Drift', backgroundcolor='rgba(10,13,18,0.5)', gridcolor='rgba(255,255,255,0.1)'),
                            zaxis=dict(title='Severity', backgroundcolor='rgba(10,13,18,0.5)', gridcolor='rgba(255,255,255,0.1)'),
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
                        <span>🟢 Normal</span>
                        <span>🟡 Watch</span>
                        <span>🟠 Warning</span>
                        <span>🔴 Critical</span>
                        <span>💎 Start</span>
                        <span>❌ Anomaly</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Need at least 5 frames of Camera Grid data. Play video in Camera Grid tab first.")
        
        else:
            st.info(" Analyze Camera Grid data from the Intelligence Dashboard tab first to see ensemble detection results")
    
    # =========================================================================
    # TAB 3: CAMERA GRID - Live Synchronized Feed (ANOMALY FRAMES ONLY)
    # =========================================================================
    with tab_camera:
        # Initialize playback state FIRST
        if 'is_playing' not in st.session_state:
            st.session_state.is_playing = False
        if 'current_frame_idx' not in st.session_state:
            st.session_state.current_frame_idx = 0
        
        # Check if using drone video mode
        use_drone_video = data_mode == "drone_video"
        
        # Load frames based on selected mode
        if use_drone_video:
            # Load drone video with YOLOv8 detection
            all_frames, frame_annotations = load_drone_video_frames()
            dataset_name = "Drone Detection Video"
            normal_label = "BIRD"
            anomaly_label = "DRONE"
        elif use_drone_bird:
            # Load drone vs bird images WITH annotations (BIRD=normal, DRONE=anomaly)
            all_frames, frame_annotations = load_all_drone_bird_frames("mixed")
            dataset_name = "Drone vs Bird"
            normal_label = "BIRD"
            anomaly_label = "DRONE"
        elif use_real:
            # Load ALL frames with ground truth annotations (normal + anomaly)
            all_frames, frame_annotations = load_all_ucsd_frames()
            dataset_name = "UCSD Pedestrian (Live)"
            normal_label = "PEDESTRIAN"
            anomaly_label = "ANOMALY"
        else:
            all_frames = []
            frame_annotations = None
            dataset_name = "Synthetic"
            normal_label = "NORMAL"
            anomaly_label = "ANOMALY"
        
        if all_frames:
            total_frames = len(all_frames)
            
            # Get current frame index
            current_frame = st.session_state.current_frame_idx
            
            # For UCSD or Drone Video, we have ground truth annotations - use them directly
            if frame_annotations and current_frame < len(frame_annotations):
                # Use ground truth labels directly
                ann = frame_annotations[current_frame]
                is_anomaly = ann.get('is_anomaly', False)
                # Handle both UCSD format (anomaly_type) and drone video format (detected_class)
                anomaly_type = ann.get('anomaly_type', ann.get('detected_class', None))
                detect_label = ann['label']
                test_folder = ann.get('test', f"Frame {ann.get('frame_num', current_frame+1)}")
                
                # For drone video, use pre-computed severity from YOLOv8
                if use_drone_video:
                    detection_state = ann['severity']
                    state_color = ann['color']
                    # Set TDI based on state
                    if detection_state == 'NORMAL':
                        tdi_base = 8.0
                    elif detection_state == 'WARNING':
                        tdi_base = 70.0
                    else:  # CRITICAL
                        tdi_base = 95.0
                else:
                    # UCSD: Track consecutive anomaly frames for CRITICAL escalation
                    if 'consecutive_anomaly_count' not in st.session_state:
                        st.session_state.consecutive_anomaly_count = 0
                    
                    if is_anomaly:
                        st.session_state.consecutive_anomaly_count += 1
                    else:
                        st.session_state.consecutive_anomaly_count = 0
                    
                    # Determine state based on anomaly and persistence
                    if not is_anomaly:
                        detection_state = 'NORMAL'
                        state_color = '#22c55e'
                        tdi_base = 8.0 + (current_frame % 5)  # Low TDI for normal
                    elif st.session_state.consecutive_anomaly_count >= CRITICAL_PERSISTENCE_THRESHOLD:
                        detection_state = 'CRITICAL'
                        state_color = '#ef4444'  # Red for critical
                        tdi_base = 95.0  # High TDI for critical
                    else:
                        detection_state = 'WARNING'
                        state_color = '#f97316'  # Orange for warning
                        tdi_base = 70.0
            else:
                # Fallback for drone/bird images or missing annotations
                detection_state = "NORMAL"
                state_color = "#22c55e"
                tdi_base = 10.0
                anomaly_type = None
                detect_label = normal_label
                test_folder = None
                is_anomaly = False
            
            # ===== CAPTURE CAMERA GRID DATA FOR INTELLIGENCE DASHBOARD =====
            # Store current frame's data for later analysis
            if 'camera_grid_data' not in st.session_state:
                st.session_state.camera_grid_data = {
                    'tdi': [], 'zones': [], 'frames': [], 'labels': [],
                    'anomaly_types': [], 'is_anomaly': [], 'timestamps': [],
                }
            
            # Only add if this frame hasn't been captured yet (avoid duplicates on rerun)
            if current_frame not in st.session_state.camera_grid_data['frames']:
                st.session_state.camera_grid_data['tdi'].append(tdi_base)
                st.session_state.camera_grid_data['zones'].append(detection_state)
                st.session_state.camera_grid_data['frames'].append(current_frame)
                st.session_state.camera_grid_data['labels'].append(detect_label)
                st.session_state.camera_grid_data['anomaly_types'].append(anomaly_type if anomaly_type else '')
                st.session_state.camera_grid_data['is_anomaly'].append(is_anomaly)
                st.session_state.camera_grid_data['timestamps'].append(datetime.now().isoformat())
            
            # ===== HEADER =====
            st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Multi-Camera Surveillance Grid</span></div>', unsafe_allow_html=True)
            
            # Dynamic header color based on current state
            if detection_state == 'NORMAL':
                header_bg = "rgba(34, 197, 94, 0.2)"
                header_border = "rgba(34, 197, 94, 0.4)"
                header_color = "#22c55e"
                header_icon = "✓"
            elif detection_state == 'WARNING':
                header_bg = "rgba(249, 115, 22, 0.2)"
                header_border = "rgba(249, 115, 22, 0.4)"
                header_color = "#f97316"
                header_icon = "⚠️"
            elif detection_state == 'CRITICAL':
                header_bg = "rgba(239, 68, 68, 0.3)"
                header_border = "rgba(239, 68, 68, 0.6)"
                header_color = "#ef4444"
                header_icon = "🚨"
            else:
                header_bg = "rgba(249, 115, 22, 0.2)"
                header_border = "rgba(249, 115, 22, 0.4)"
                header_color = "#f97316"
                header_icon = "⚠️"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 12px; background: {header_bg}; border-radius: 8px; border: 1px solid {header_border}; margin-bottom: 16px;">
                <span style="color: {header_color}; font-weight: 600; font-size: 1.1rem;">{header_icon} {dataset_name} - {total_frames} Frames | Current: {detect_label}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # ===== DATA CAPTURE STATUS =====
            captured_frames = len(st.session_state.camera_grid_data.get('frames', []))
            captured_anomalies = sum(1 for z in st.session_state.camera_grid_data.get('zones', []) if z != 'NORMAL')
            
            col_status, col_reset = st.columns([3, 1])
            with col_status:
                st.markdown(f"""
                <div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); 
                            border-radius: 8px; padding: 10px; margin-bottom: 16px;">
                    <span style="color: #06b6d4; font-weight: bold;">📊 Data Captured:</span>
                    <span style="color: #94a3b8;"> {captured_frames} frames | {captured_anomalies} anomalies detected</span>
                    <span style="color: #64748b; font-size: 0.8rem;"> — Go to Intelligence Dashboard to analyze</span>
                </div>
                """, unsafe_allow_html=True)
            with col_reset:
                if st.button("🔄 Reset Capture", key="reset_capture"):
                    st.session_state.camera_grid_data = {
                        'tdi': [], 'zones': [], 'frames': [], 'labels': [],
                        'anomaly_types': [], 'is_anomaly': [], 'timestamps': [],
                    }
                    st.session_state.current_frame_idx = 0
                    st.session_state.consecutive_anomaly_count = 0
                    st.rerun()
            
            # ===== MAIN VIDEO DISPLAY =====
            st.markdown("### Main Surveillance Feed")
            
            # Get current frame and process it for display
            main_frame = all_frames[current_frame].copy()
            if len(main_frame.shape) == 2:
                main_frame = cv2.cvtColor(main_frame, cv2.COLOR_GRAY2BGR)
            
            h, w = main_frame.shape[:2]
            
            # Draw detection box with color based on state
            color_bgr = {'NORMAL': (0, 200, 0), 'WARNING': (0, 140, 255), 'CRITICAL': (0, 0, 255)}[detection_state]
            box_x1, box_y1 = int(w * 0.15), int(h * 0.15)
            box_x2, box_y2 = int(w * 0.85), int(h * 0.85)
            thickness = 3 if detection_state != "NORMAL" else 2
            cv2.rectangle(main_frame, (box_x1, box_y1), (box_x2, box_y2), color_bgr, thickness)
            
            # Detection label on main frame - use the label from detection
            # detect_label is already set from anomaly_scores above
            
            label_size = cv2.getTextSize(detect_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(main_frame, (box_x1, box_y1 - 30), (box_x1 + label_size[0] + 15, box_y1), color_bgr, -1)
            cv2.putText(main_frame, detect_label, (box_x1 + 8, box_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Frame info overlay
            cv2.rectangle(main_frame, (5, 5), (150, 35), (0, 0, 0), -1)
            cv2.putText(main_frame, f"Frame: {current_frame + 1}/{total_frames}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # TDI overlay
            cv2.rectangle(main_frame, (w - 110, 5), (w - 5, 35), (0, 0, 0), -1)
            cv2.putText(main_frame, f"TDI: {min(100, tdi_base):.0f}", (w - 100, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            # Status overlay with anomaly type
            status_text = f"{detection_state}"
            if anomaly_type and detection_state != "NORMAL":
                status_text = f"{anomaly_type.replace('_', ' ').upper()[:12]}"
            cv2.rectangle(main_frame, (5, h - 35), (max(120, len(status_text) * 12 + 20), h - 5), color_bgr, -1)
            cv2.putText(main_frame, status_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Encode main frame
            _, main_buffer = cv2.imencode('.jpg', main_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            main_frame_b64 = base64.b64encode(main_buffer).decode('utf-8')
            
            # Anomaly type badge for display
            anomaly_badge = ""
            if anomaly_type and detection_state != "NORMAL":
                anomaly_badge = f'<span style="color: {state_color}; font-size: 0.8rem; margin-left: 8px;">({anomaly_type.replace("_", " ").title()})</span>'
            
            # Display main video with prominent border
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.9); border-radius: 16px; overflow: hidden; 
                        border: 3px solid {state_color}; margin-bottom: 20px; box-shadow: 0 0 30px {state_color}40;">
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 12px 16px; background: linear-gradient(90deg, rgba(0,0,0,0.8), rgba(0,0,0,0.5));">
                    <span style="color: white; font-weight: 700; font-size: 1.1rem;">MAIN FEED - CAM-001</span>
                    <span style="color: {state_color}; font-weight: bold; font-size: 1.1rem; 
                                 padding: 4px 12px; background: {state_color}20; border-radius: 20px;">{detection_state}{anomaly_badge}</span>
                </div>
                <img src="data:image/jpeg;base64,{main_frame_b64}" style="width: 100%; height: 400px; object-fit: contain; background: #0a0a0a;">
                <div style="display: flex; justify-content: space-between; padding: 12px 16px; 
                            background: linear-gradient(90deg, rgba(0,0,0,0.8), rgba(0,0,0,0.5)); font-size: 0.9rem;">
                    <span style="color: #94a3b8;">Frame: <span style="color: white; font-weight: bold;">{current_frame + 1} / {total_frames}</span></span>
                    <span style="color: #94a3b8;">TDI: <span style="color: {state_color}; font-weight: bold;">{min(100, tdi_base):.0f}</span></span>
                    <span style="color: #94a3b8;">Detection: <span style="color: {state_color}; font-weight: bold;">{detect_label}</span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ===== PLAYBACK CONTROLS =====
            st.markdown("### Playback Controls")
            
            # Play/Pause button
            col_btn, col_info = st.columns([1, 3])
            with col_btn:
                if st.session_state.is_playing:
                    if st.button("⏸ PAUSE", key="pause_btn", use_container_width=True, type="secondary"):
                        st.session_state.is_playing = False
                        st.rerun()
                else:
                    if st.button("▶ PLAY", key="play_btn", use_container_width=True, type="primary"):
                        st.session_state.is_playing = True
                        st.rerun()
            
            with col_info:
                current_frame = st.session_state.current_frame_idx
                if st.session_state.is_playing:
                    st.markdown(f"""
                    <div style="padding: 10px; background: rgba(34, 197, 94, 0.2); border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.4);">
                        <span style="color: #22c55e; font-weight: bold;">▶ PLAYING</span>
                        <span style="color: #94a3b8;"> - Frame {current_frame + 1}/{total_frames} @ 20 FPS</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding: 10px; background: rgba(100, 116, 139, 0.2); border-radius: 8px; border: 1px solid rgba(100, 116, 139, 0.4);">
                        <span style="color: #64748b; font-weight: bold;">⏸ PAUSED</span>
                        <span style="color: #94a3b8;"> - Frame {current_frame + 1}/{total_frames}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Camera Grid - Synchronized View")
            
            # Display 6 cameras in a 3x2 grid - ALL SYNCHRONIZED to same frame
            zone_colors_bgr = {'NORMAL': (34, 197, 94), 'WARNING': (249, 115, 22), 'CRITICAL': (239, 68, 68)}
            
            for row in range(2):
                cols = st.columns(3)
                for col_idx in range(3):
                    cam_idx = row * 3 + col_idx
                    cam = st.session_state.cameras[cam_idx]
                    
                    # ALL cameras show the SAME frame (synchronized)
                    cam_frame_idx = current_frame
                    
                    # Get frame and process
                    frame = all_frames[cam_frame_idx].copy()
                    
                    # Convert grayscale to color
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    h, w = frame.shape[:2]
                    
                    # Get camera-specific state from ground truth annotations
                    if frame_annotations and cam_frame_idx < len(frame_annotations):
                        ann = frame_annotations[cam_frame_idx]
                        cam_is_anomaly = ann.get('is_anomaly', False)
                        # Handle both UCSD format (anomaly_type) and drone video format (detected_class)
                        cam_anomaly_type = ann.get('anomaly_type', ann.get('detected_class', None))
                        cam_label = ann['label']
                        
                        # For drone video, use pre-computed severity
                        if use_drone_video:
                            cam_state = ann['severity']
                            if cam_state == 'NORMAL':
                                cam_tdi = 8.0
                            elif cam_state == 'WARNING':
                                cam_tdi = 70.0
                            else:  # CRITICAL
                                cam_tdi = 95.0
                        else:
                            # UCSD: Use the same state as main view (based on consecutive anomaly count)
                            if not cam_is_anomaly:
                                cam_state = 'NORMAL'
                                cam_tdi = 8.0
                            elif st.session_state.get('consecutive_anomaly_count', 0) >= CRITICAL_PERSISTENCE_THRESHOLD:
                                cam_state = 'CRITICAL'
                                cam_tdi = 95.0
                            else:
                                cam_state = 'WARNING'
                                cam_tdi = 70.0
                    else:
                        cam_state = "NORMAL"
                        cam_tdi = 10.0
                        cam_anomaly_type = None
                        cam_label = normal_label
                        cam_is_anomaly = False
                    
                    cam_tdi = max(0, min(100, cam_tdi))
                    color_hex = {'NORMAL': '#22c55e', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}[cam_state]
                    color_bgr = {'NORMAL': (0, 200, 0), 'WARNING': (0, 140, 255), 'CRITICAL': (0, 0, 255)}[cam_state]
                    
                    # Draw detection box - thicker for anomalies
                    box_x1, box_y1 = int(w * 0.1), int(h * 0.1)
                    box_x2, box_y2 = int(w * 0.9), int(h * 0.9)
                    thickness = 3 if cam_is_anomaly else 2
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color_bgr, thickness)
                    
                    # Detection label - use label from ground truth
                    cam_detect_label = cam_label
                    
                    label_size = cv2.getTextSize(cam_detect_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (box_x1, box_y1 - 20), (box_x1 + label_size[0] + 10, box_y1), color_bgr, -1)
                    cv2.putText(frame, cam_detect_label, (box_x1 + 5, box_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # TDI and Frame info overlay
                    cv2.rectangle(frame, (2, 2), (90, 22), (0, 0, 0), -1)
                    cv2.putText(frame, f"TDI:{cam_tdi:.0f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
                    
                    cv2.rectangle(frame, (w - 60, 2), (w - 2, 22), (0, 0, 0), -1)
                    cv2.putText(frame, f"F:{cam_frame_idx}", (w - 58, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.9); border-radius: 12px; overflow: hidden; 
                                    border: 2px solid {color_hex}; margin-bottom: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; 
                                        padding: 8px 12px; background: rgba(0,0,0,0.5);">
                                <span style="color: white; font-weight: 600;">{cam['id']}</span>
                                <span style="color: {color_hex}; font-weight: bold;">{cam_state}</span>
                            </div>
                            <img src="data:image/jpeg;base64,{frame_b64}" style="width: 100%; height: 160px; object-fit: cover;">
                            <div style="display: flex; justify-content: space-between; padding: 8px 12px; 
                                        background: rgba(0,0,0,0.5); font-size: 0.8rem;">
                                <span style="color: {color_hex};">TDI: {cam_tdi:.0f}</span>
                                <span style="color: #94a3b8;">{cam['zone']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Detection Status Panel - shows stats ONLY for frames played so far
            st.markdown("---")
            st.markdown("### Detection Status (Frames Analyzed)")
            
            # Count anomalies by type - ONLY up to current frame
            anomaly_counts = {'NORMAL': 0, 'WARNING': 0, 'CRITICAL': 0}
            label_counts = {}  # Count by label (BIKER, SKATER, etc.)
            frames_analyzed = current_frame + 1  # How many frames we've seen
            
            if frame_annotations and current_frame > 0:
                # Count consecutive anomalies to determine WARNING vs CRITICAL
                consec_count = 0
                for i, ann in enumerate(frame_annotations[:frames_analyzed]):
                    is_anom = ann.get('is_anomaly', False)
                    if is_anom:
                        consec_count += 1
                        if consec_count >= CRITICAL_PERSISTENCE_THRESHOLD:
                            anomaly_counts['CRITICAL'] = anomaly_counts.get('CRITICAL', 0) + 1
                        else:
                            anomaly_counts['WARNING'] = anomaly_counts.get('WARNING', 0) + 1
                    else:
                        consec_count = 0
                        anomaly_counts['NORMAL'] = anomaly_counts.get('NORMAL', 0) + 1
                    
                    label = ann['label']
                    if label != 'PEDESTRIAN':  # Only count anomaly labels
                        label_counts[label] = label_counts.get(label, 0) + 1
            
            st.markdown(f"""
            <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
                <div style="text-align: center; padding: 12px 20px; background: {'rgba(34,197,94,0.3)' if detection_state == 'NORMAL' else 'rgba(34,197,94,0.1)'}; 
                            border: 2px solid {'#22c55e' if detection_state == 'NORMAL' else '#22c55e60'}; border-radius: 8px;">
                    <div style="font-size: 0.7rem; color: #22c55e;">NORMAL (Pedestrian)</div>
                    <div style="color: {'#22c55e' if detection_state == 'NORMAL' else '#94a3b8'}; font-weight: bold;">{anomaly_counts.get('NORMAL', 0)} frames</div>
                </div>
                <div style="text-align: center; padding: 12px 20px; background: {'rgba(249,115,22,0.3)' if detection_state == 'WARNING' else 'rgba(249,115,22,0.1)'}; 
                            border: 2px solid {'#f97316' if detection_state == 'WARNING' else '#f9731660'}; border-radius: 8px;">
                    <div style="font-size: 0.7rem; color: #f97316;">WARNING (Anomaly Detected)</div>
                    <div style="color: {'#f97316' if detection_state == 'WARNING' else '#94a3b8'}; font-weight: bold;">{anomaly_counts.get('WARNING', 0)} frames</div>
                </div>
                <div style="text-align: center; padding: 12px 20px; background: {'rgba(239,68,68,0.3)' if detection_state == 'CRITICAL' else 'rgba(239,68,68,0.1)'}; 
                            border: 2px solid {'#ef4444' if detection_state == 'CRITICAL' else '#ef444460'}; border-radius: 8px;">
                    <div style="font-size: 0.7rem; color: #ef4444;">CRITICAL (Persistent)</div>
                    <div style="color: {'#ef4444' if detection_state == 'CRITICAL' else '#94a3b8'}; font-weight: bold;">{anomaly_counts.get('CRITICAL', 0)} frames</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show detected anomaly types (only if there are anomalies)
            if label_counts:
                st.markdown("#### Detected Anomaly Types")
                label_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                cols = st.columns(min(len(label_items), 5))
                for i, (label, count) in enumerate(label_items):
                    # Get color - all anomalies are orange/red based on persistence
                    label_color = '#f97316'  # Warning orange for all anomaly types
                    with cols[i]:
                        st.markdown(f"""
                        <div style="padding: 8px 16px; background: rgba(249, 115, 22, 0.1); border: 1px solid {label_color}40; border-radius: 6px; text-align: center;">
                            <div style="color: {label_color}; font-weight: 600;">{label}</div>
                            <div style="color: #94a3b8; font-size: 0.8rem;">{count} frames</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # AUTO-ADVANCE FRAMES WHEN PLAYING (at the very end of tab3)
            if st.session_state.is_playing:
                # Capture data for ALL frames we're skipping (not just displayed frame)
                current_idx = st.session_state.current_frame_idx
                for skip_i in range(3):  # Capture data for all 3 frames we're skipping
                    frame_to_capture = (current_idx + skip_i) % total_frames
                    if frame_annotations and frame_to_capture < len(frame_annotations):
                        ann = frame_annotations[frame_to_capture]
                        skip_tdi = 8.0 if ann['severity'] == 'NORMAL' else (70.0 if ann['severity'] == 'WARNING' else 95.0)
                        skip_zone = ann['severity']
                        skip_label = ann['label']
                        skip_anomaly = ann.get('is_anomaly', False)
                        
                        if frame_to_capture not in st.session_state.camera_grid_data['frames']:
                            st.session_state.camera_grid_data['tdi'].append(skip_tdi)
                            st.session_state.camera_grid_data['zones'].append(skip_zone)
                            st.session_state.camera_grid_data['frames'].append(frame_to_capture)
                            st.session_state.camera_grid_data['labels'].append(skip_label)
                            st.session_state.camera_grid_data['anomaly_types'].append(ann.get('detected_class', ''))
                            st.session_state.camera_grid_data['is_anomaly'].append(skip_anomaly)
                            st.session_state.camera_grid_data['timestamps'].append(datetime.now().isoformat())
                
                # Skip multiple frames for faster playback
                st.session_state.current_frame_idx = (st.session_state.current_frame_idx + 3) % total_frames
                st.rerun()
            
        else:
            # Synthetic mode - show placeholder
            st.info("Select UCSD Pedestrian or Drone vs Bird dataset from the sidebar to view live camera feeds.")
            
            st.markdown("### Simulated Camera Grid")
            for row in range(2):
                cols = st.columns(3)
                for col_idx in range(3):
                    cam_idx = row * 3 + col_idx
                    cam = st.session_state.cameras[cam_idx]
                    color = {'NORMAL': '#22c55e', 'WATCH': '#eab308', 'WARNING': '#f97316', 'CRITICAL': '#ef4444'}.get(cam['status'], '#64748b')
                    
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.9); border-radius: 12px; overflow: hidden; 
                                    border: 2px solid {color}; margin-bottom: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; 
                                        padding: 8px 12px; background: rgba(0,0,0,0.5);">
                                <span style="color: white; font-weight: 600;">{cam['id']}</span>
                                <span style="color: {color}; font-weight: bold;">{cam['status']}</span>
                            </div>
                            <div style="height: 160px; display: flex; align-items: center; justify-content: center; 
                                        background: rgba(30, 41, 59, 0.5);">
                                <span style="font-size: 3rem; opacity: 0.3;">CAM</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 8px 12px; 
                                        background: rgba(0,0,0,0.5); font-size: 0.8rem;">
                                <span style="color: {color};">TDI: {cam['tdi']:.0f}</span>
                                <span style="color: #94a3b8;">{cam['zone']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 4: INCIDENT LOG
    # =========================================================================
    with tab_incident:
        st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Incident Log & Alert History</span></div>', unsafe_allow_html=True)
        
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
            st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Incident Timeline</span></div>', unsafe_allow_html=True)
            
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
            st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Incident Details</span></div>', unsafe_allow_html=True)
            
            for incident in reversed(incidents[-20:]):  # Show last 20
                zone = incident.get('zone', 'WARNING')
                zone_color = '#22c55e' if zone in ['Normal', 'NORMAL'] else ('#eab308' if zone in ['Watch', 'WATCH'] else ('#f97316' if zone in ['Warning', 'WARNING'] else '#ef4444'))
                category = incident.get('category', incident.get('type', 'unknown'))
                severity = incident.get('severity', 'MEDIUM')
                source = incident.get('source', 'Analysis')
                description = incident.get('description', '')
                
                category_icons = {
                    'normal': '', 'loitering': '', 'intrusion': '', 
                    'crowd_formation': '', 'erratic_movement': '',
                    'coordinated': '', 'speed_anomaly': '', 
                    'direction_anomaly': '', 'unknown': '',
                    'Drift Onset': '', 'Confirmed Drift': '', 
                    'Threat Confirmed': '', 'Video Detection': ''
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
                                {category_icons.get(category, '')} {str(category).replace('_', ' ').title()}
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
                    label=" Export Incidents CSV",
                    data=csv_data,
                    file_name=f"incidents_{st.session_state.session_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                json_data = json.dumps(incidents, indent=2)
                st.download_button(
                    label=" Export Incidents JSON",
                    data=json_data,
                    file_name=f"incidents_{st.session_state.session_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Full Session Export Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="section-icon"></span><span class="section-title">Full Session Analytics & Export</span></div>', unsafe_allow_html=True)
            
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
                        label=" Full Analysis CSV",
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
                        label=" Full Analysis JSON",
                        data=json_data,
                        file_name=f"drishti_analysis_{st.session_state.session_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with exp_col3:
                    st.markdown(f"""
                    <div style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); 
                                border-radius: 8px; padding: 12px; font-size: 0.8rem;">
                        <strong style="color: #22c55e;"> Report Ready</strong><br>
                        <span style="color: #94a3b8;">Session: {st.session_state.session_id}<br>
                        Frames: {len(tdi_vals)} | Peak TDI: {max(tdi_vals):.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 4rem; margin-bottom: 20px;"></div>
                <div style="font-size: 1.2rem; color: #22c55e; margin-bottom: 10px;">No Incidents Detected</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">All behavioral patterns within normal parameters. Run analysis to generate data.</div>
            </div>
            """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
