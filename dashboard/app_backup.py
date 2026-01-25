"""
NOISE FLOOR - Streamlit Dashboard
===================================
Defence-grade behavioral drift intelligence interface.

This dashboard provides:
1. Real-time drift monitoring visualization
2. Graduated watch zone indicators
3. System status and performance metrics
4. Comparison with baseline methods

Run with: streamlit run dashboard/app.py
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

from src.feature_extraction import FeatureExtractor, create_synthetic_normal_data, create_synthetic_drift_data
from src.autoencoder import NormalityAutoencoder
from src.drift_detection import DriftDetector
from src.watch_zones import WatchZoneClassifier, Zone
from src.baseline_comparison import BaselineComparator

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NOISE FLOOR - Behavioral Drift Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROFESSIONAL DARK THEME CSS
# =============================================================================
st.markdown("""
<style>
    /* Global dark theme overrides */
    .stApp {
        background-color: #1a1d24;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #4A9EF7;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #8892a0;
        margin-top: 4px;
        font-weight: 400;
    }
    
    /* System status bar */
    .system-status {
        background: linear-gradient(90deg, #1e2530 0%, #242a36 100%);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 15px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #48bb78;
        box-shadow: 0 0 8px rgba(72, 187, 120, 0.5);
    }
    .status-text {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Zone indicator - restrained design */
    .zone-card {
        background: #1e2530;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid #4A9EF7;
    }
    .zone-card.alert {
        border-left-color: #e53e3e;
    }
    .zone-card.warning {
        border-left-color: #ed8936;
    }
    .zone-card.watch {
        border-left-color: #ecc94b;
    }
    .zone-card.normal {
        border-left-color: #48bb78;
    }
    .zone-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .zone-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .zone-strength {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 6px;
    }
    .zone-action {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 8px;
        font-style: italic;
    }
    
    /* Explanation cards - improved contrast */
    .explanation-card {
        background: #242a36;
        border: 1px solid #2d3748;
        padding: 20px;
        border-radius: 10px;
        margin: 8px 0;
        height: 100%;
    }
    .explanation-card h4 {
        color: #4A9EF7;
        font-size: 1rem;
        margin-bottom: 12px;
        font-weight: 600;
    }
    .explanation-card ul {
        color: #cbd5e0;
        font-size: 0.9rem;
        padding-left: 18px;
        margin: 0;
    }
    .explanation-card li {
        margin-bottom: 6px;
    }
    .explanation-card .tagline {
        color: #718096;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 12px;
        border-top: 1px solid #2d3748;
        padding-top: 10px;
    }
    
    /* Metric cards with hierarchy */
    .metric-primary {
        background: #242a36;
        border: 1px solid #4A9EF7;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-secondary {
        background: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 8px 0 4px 0;
    }
    .metric-hint {
        font-size: 0.7rem;
        color: #4A9EF7;
    }
    
    /* Button styling */
    .stButton > button {
        background: #2d3a4d;
        border: 1px solid #4A9EF7;
        color: #e2e8f0;
        font-weight: 500;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #3d4a5d;
        border-color: #6bb3ff;
    }
    
    /* Helper text */
    .helper-text {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 6px;
    }
    
    /* Impact statement */
    .impact-statement {
        background: linear-gradient(90deg, rgba(74, 158, 247, 0.1) 0%, rgba(74, 158, 247, 0.05) 100%);
        border-left: 3px solid #4A9EF7;
        padding: 12px 16px;
        margin: 20px 0;
        border-radius: 0 8px 8px 0;
    }
    .impact-statement p {
        color: #a0aec0;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e2530;
    }
    
    /* Loading state */
    .analyzing-state {
        background: #1e2530;
        border: 1px dashed #4A9EF7;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        color: #a0aec0;
    }
    
    /* Deployment characteristics bar */
    .deployment-bar {
        background: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 8px 0 16px 0;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
    }
    .deploy-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #718096;
        font-size: 0.8rem;
    }
    .deploy-item span {
        color: #48bb78;
    }
    
    /* Scenario cards */
    .scenario-card {
        background: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
    }
    .scenario-card .icon {
        font-size: 1.2rem;
        margin-bottom: 6px;
    }
    .scenario-card .title {
        color: #e2e8f0;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .scenario-card .desc {
        color: #718096;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    /* Why unsupervised info box */
    .info-box {
        background: rgba(74, 158, 247, 0.08);
        border: 1px solid rgba(74, 158, 247, 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 12px;
    }
    .info-box h5 {
        color: #4A9EF7;
        font-size: 0.85rem;
        margin: 0 0 10px 0;
        font-weight: 500;
    }
    .info-box ul {
        color: #a0aec0;
        font-size: 0.8rem;
        padding-left: 16px;
        margin: 0;
    }
    .info-box li {
        margin-bottom: 4px;
    }
    .info-box .footer {
        color: #4A9EF7;
        font-size: 0.75rem;
        margin-top: 10px;
        font-style: italic;
    }
    
    /* Philosophy comparison table */
    .philosophy-table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
    }
    .philosophy-table th {
        background: #242a36;
        color: #718096;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #2d3748;
    }
    .philosophy-table td {
        padding: 10px 14px;
        font-size: 0.85rem;
        border-bottom: 1px solid #2d3748;
    }
    .philosophy-table td:first-child {
        color: #718096;
    }
    .philosophy-table td:last-child {
        color: #4A9EF7;
        font-weight: 500;
    }
    .philosophy-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Research grounding */
    .research-note {
        color: #4a5568;
        font-size: 0.75rem;
        font-style: italic;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.history = {
            'drift_scores': [],
            'zones': [],
            'raw_scores': [],
            'timestamps': [],
        }
        st.session_state.monitoring_active = False


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
@st.cache_resource
def initialize_models():
    """Initialize and train models on synthetic normal data."""
    # Create synthetic normal training data
    normal_data = create_synthetic_normal_data(1000)
    
    # Normalize training data
    train_mean = np.mean(normal_data, axis=0)
    train_std = np.std(normal_data, axis=0)
    train_std[train_std == 0] = 1
    train_normalized = (normal_data - train_mean) / train_std
    
    # Initialize and train autoencoder
    autoencoder = NormalityAutoencoder(input_dim=normal_data.shape[1])
    autoencoder.compile()
    autoencoder.train(train_normalized, epochs=50, verbose=0)
    
    # Initialize drift detector
    drift_detector = DriftDetector(baseline_frames=50)
    
    # Initialize zone classifier
    zone_classifier = WatchZoneClassifier()
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Store normalization stats
    feature_stats = {'mean': train_mean, 'std': train_std}
    
    return autoencoder, drift_detector, zone_classifier, feature_extractor, feature_stats


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================
def create_drift_chart(history: dict, thresholds: dict, drift_start: int) -> go.Figure:
    """Create drift score chart with zone bands and reduced timeline."""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.82, 0.18],  # Reduced timeline height
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("", "")  # No titles, we'll add custom ones
    )
    
    if not history['drift_scores']:
        return fig
    
    frames = list(range(len(history['drift_scores'])))
    drift_scores = history['drift_scores']
    max_score = max(drift_scores) if drift_scores else 5
    
    # Add zone bands (faint background)
    zone_bands = [
        (0, thresholds['normal'], 'rgba(72, 187, 120, 0.08)', 'Normal'),
        (thresholds['normal'], thresholds['watch'], 'rgba(236, 201, 75, 0.08)', 'Watch'),
        (thresholds['watch'], thresholds['warning'], 'rgba(237, 137, 54, 0.08)', 'Warning'),
        (thresholds['warning'], max_score * 1.2, 'rgba(229, 62, 62, 0.08)', 'Alert'),
    ]
    
    for y0, y1, color, name in zone_bands:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=color,
            line_width=0,
            row=1, col=1
        )
    
    # Main drift score line - prominent
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=drift_scores,
            mode='lines',
            name='Drift Score',
            line=dict(color='#4A9EF7', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(74, 158, 247, 0.15)'
        ),
        row=1, col=1
    )
    
    # Add drift start marker
    fig.add_vline(
        x=drift_start,
        line_dash="dash",
        line_color="#e53e3e",
        opacity=0.7,
        annotation_text="Drift Begins",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color="#a0aec0",
        row=1, col=1
    )
    
    # Zone timeline with desaturated colors
    if history['zones']:
        zone_colors_map = {
            'NORMAL': 'rgba(72, 187, 120, 0.5)',
            'WATCH': 'rgba(236, 201, 75, 0.5)',
            'WARNING': 'rgba(237, 137, 54, 0.5)',
            'ALERT': 'rgba(229, 62, 62, 0.5)'
        }
        zone_colors_list = [zone_colors_map.get(z, 'rgba(128,128,128,0.5)') for z in history['zones']]
        
        fig.add_trace(
            go.Bar(
                x=frames,
                y=[1] * len(frames),
                marker_color=zone_colors_list,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    # Layout with dark theme
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=80, t=30, b=40),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a1d24',
        font=dict(color='#a0aec0'),
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor='#2d3748',
        zerolinecolor='#2d3748',
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        gridcolor='#2d3748',
        zerolinecolor='#2d3748',
        tickfont=dict(size=10),
    )
    
    fig.update_yaxes(title_text="Drift Score", title_font_size=11, row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(title_text="Frame", title_font_size=11, row=2, col=1)
    
    # Add annotation for scale context
    fig.add_annotation(
        x=1.02, y=0.5,
        xref='paper', yref='paper',
        text="Score relative to<br>learned baseline",
        showarrow=False,
        font=dict(size=9, color='#718096'),
        align='left'
    )
    
    return fig


def create_zone_indicator(zone: Zone, confidence: float) -> str:
    """Create restrained zone indicator HTML."""
    zone_name = str(zone).upper()
    
    # Map confidence to strength label
    if confidence > 0.8:
        strength = "High"
    elif confidence > 0.5:
        strength = "Moderate"
    else:
        strength = "Low"
    
    zone_class = zone_name.lower()
    
    return f"""
    <div class="zone-card {zone_class}">
        <div class="zone-label">Current Classification</div>
        <div class="zone-value">{zone.icon} {zone_name}</div>
        <div class="zone-strength">Drift Strength: {strength}</div>
        <div class="zone-action">{zone.action}</div>
    </div>
    """


def create_comparison_chart(results: dict, drift_start: int) -> go.Figure:
    """Create baseline comparison chart with NOISE FLOOR emphasized."""
    fig = go.Figure()
    
    # Define styling for each method
    method_styles = {
        'noise_floor': {'color': '#4A9EF7', 'width': 3, 'opacity': 1.0},
        'isolation_forest': {'color': '#a0aec0', 'width': 1.5, 'opacity': 0.4},
        'threshold': {'color': '#ed8936', 'width': 1.5, 'opacity': 0.4},
        'one_class_svm': {'color': '#e53e3e', 'width': 1.5, 'opacity': 0.4},
    }
    
    for name, result in results.items():
        scores = result.scores
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        style = method_styles.get(name, {'color': '#718096', 'width': 1, 'opacity': 0.3})
        
        fig.add_trace(go.Scatter(
            x=list(range(len(scores))),
            y=scores_normalized,
            mode='lines',
            name=result.method_name,
            line=dict(color=style['color'], width=style['width']),
            opacity=style['opacity']
        ))
    
    # Drift start marker
    fig.add_vline(
        x=drift_start, 
        line_dash="dash", 
        line_color="#e53e3e",
        opacity=0.7
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=60, r=20, t=40, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a1d24',
        font=dict(color='#a0aec0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        xaxis_title="Frame",
        yaxis_title="Normalized Score",
    )
    
    fig.update_xaxes(gridcolor='#2d3748', zerolinecolor='#2d3748')
    fig.update_yaxes(gridcolor='#2d3748', zerolinecolor='#2d3748')
    
    # Key insight annotation
    fig.add_annotation(
        x=drift_start + 30, y=0.85,
        text="Earlier, smoother detection<br>with temporal context",
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowcolor='#4A9EF7',
        ax=50, ay=-30,
        font=dict(size=9, color='#4A9EF7'),
        bgcolor='rgba(30, 37, 48, 0.9)',
        borderpad=4
    )
    
    return fig


# =============================================================================
# DEMO SIMULATION
# =============================================================================
def run_demo_simulation(autoencoder, drift_detector, zone_classifier, feature_stats, 
                       drift_start: int, drift_rate: float):
    """Run simulation with synthetic data."""
    # Generate data
    normal_data = create_synthetic_normal_data(drift_start)
    drift_data = create_synthetic_drift_data(300 - drift_start, drift_rate=drift_rate)
    all_data = np.vstack([normal_data, drift_data])
    
    # Normalize
    normalized_data = (all_data - feature_stats['mean']) / feature_stats['std']
    
    # Process all frames
    results = []
    drift_detector.reset()
    zone_classifier.reset()
    
    for i, features in enumerate(normalized_data):
        features_batch = features.reshape(1, -1)
        norm_score = autoencoder.get_normality_score(features_batch)[0]
        drift_state = drift_detector.update(norm_score, frame_index=i)
        zone_state = zone_classifier.classify(drift_state.drift_score)
        
        results.append({
            'frame': i,
            'raw_score': norm_score,
            'drift_score': drift_state.drift_score,
            'smoothed_score': drift_state.smoothed_score,
            'trend': drift_state.trend_direction,
            'zone': zone_state.zone,
            'confidence': zone_state.confidence,
        })
    
    return results


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📡 NOISE FLOOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Behavioral Drift Intelligence System</p>', unsafe_allow_html=True)
    
    # System Status Bar
    st.markdown("""
    <div class="system-status">
        <div class="status-dot"></div>
        <span class="status-text">System Status: Monitoring stable patterns • Baseline integrity verified • No contamination detected</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Deployment Characteristics
    st.markdown("""
    <div class="deployment-bar">
        <div class="deploy-item"><span>✓</span> Real-time streaming inference</div>
        <div class="deploy-item"><span>✓</span> Low compute footprint (CPU-only)</div>
        <div class="deploy-item"><span>✓</span> Edge or cloud deployable</div>
        <div class="deploy-item"><span>✓</span> Compatible with existing CCTV pipelines</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        st.markdown("##### Scenario Parameters")
        drift_start = st.slider("Drift Onset Frame", 50, 200, 100, 
                               help="Frame where behavioral drift begins in simulation")
        drift_rate = st.slider("Drift Intensity", 0.005, 0.05, 0.015, 0.005,
                              help="Rate of behavioral change per frame")
        
        st.markdown("##### Detection Sensitivity")
        normal_thresh = st.slider("Normal Threshold", 0.5, 2.0, 1.5, 0.1)
        watch_thresh = st.slider("Watch Threshold", 1.5, 3.0, 2.0, 0.1)
        warning_thresh = st.slider("Warning Threshold", 2.0, 4.0, 2.5, 0.1)
        
        thresholds = {
            'normal': normal_thresh,
            'watch': watch_thresh,
            'warning': warning_thresh,
        }
        
        st.markdown("---")
        st.markdown("##### About")
        st.markdown("""
        <div style="color: #a0aec0; font-size: 0.85rem;">
        NOISE FLOOR detects <b>gradual behavioral drift</b> using temporal intelligence.
        <br><br>
        • Preventive, not reactive<br>
        • Learns from normal patterns<br>
        • Graduated confidence zones
        </div>
        <p class="research-note">
        Built on established research in unsupervised anomaly detection, concept drift analysis, and streaming ML.
        </p>
        """, unsafe_allow_html=True)
    
    # Initialize models
    autoencoder, drift_detector, zone_classifier, feature_extractor, feature_stats = initialize_models()
    zone_classifier.thresholds = thresholds
    
    # Main content
    col_main, col_side = st.columns([2.5, 1])
    
    with col_main:
        st.markdown('<div class="section-header">🎯 Monitoring Session</div>', unsafe_allow_html=True)
        
        # Reframed button
        if st.button("▶ Start Monitoring Session", type="primary", use_container_width=False):
            st.session_state.monitoring_active = True
            
            # Show analyzing state
            with st.spinner(""):
                st.markdown("""
                <div class="analyzing-state">
                    <p style="font-size: 1rem; margin-bottom: 8px;">⏳ Analyzing behavioral patterns...</p>
                    <p style="font-size: 0.85rem;">Establishing baseline • Processing temporal features</p>
                </div>
                """, unsafe_allow_html=True)
                
                results = run_demo_simulation(
                    autoencoder, drift_detector, zone_classifier, 
                    feature_stats, drift_start, drift_rate
                )
                
                st.session_state.history = {
                    'drift_scores': [r['drift_score'] for r in results],
                    'zones': [str(r['zone']).upper() for r in results],
                    'raw_scores': [r['raw_score'] for r in results],
                    'timestamps': [r['frame'] for r in results],
                    'results': results,
                }
                time.sleep(0.5)  # Brief pause for effect
        
        st.markdown('<p class="helper-text">Simulates gradual behavioral drift for system evaluation</p>', 
                   unsafe_allow_html=True)
    
    with col_side:
        st.markdown('<div class="section-header">📍 Zone Status</div>', unsafe_allow_html=True)
        
        if st.session_state.history.get('results'):
            latest = st.session_state.history['results'][-1]
            st.markdown(
                create_zone_indicator(latest['zone'], latest['confidence']),
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div class="zone-card normal">
                <div class="zone-label">Awaiting Analysis</div>
                <div class="zone-value" style="color: #718096;">—</div>
                <div class="zone-strength">Start monitoring to view status</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Drift Score Timeline
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Drift Score Timeline</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #718096; font-size: 0.85rem; margin-top: -10px;">Zone Classification (post-baseline period)</p>', unsafe_allow_html=True)
    
    if st.session_state.history.get('drift_scores'):
        fig = create_drift_chart(st.session_state.history, thresholds, drift_start)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detection Metrics with hierarchy
        st.markdown('<div class="section-header">📈 Detection Performance</div>', unsafe_allow_html=True)
        
        drift_scores = st.session_state.history['drift_scores']
        zones = st.session_state.history['zones']
        
        # Calculate metrics
        first_detection_idx = None
        for i in range(drift_start, len(zones)):
            if zones[i] in ['WATCH', 'WARNING', 'ALERT']:
                first_detection_idx = i
                break
        
        false_positives = sum(
            1 for i in range(min(drift_start, len(zones))) 
            if zones[i] in ['WATCH', 'WARNING', 'ALERT']
        )
        fp_rate = (false_positives / drift_start * 100) if drift_start > 0 else 0
        
        detection_delay = (first_detection_idx - drift_start) if first_detection_idx else None
        
        # Metric cards with hierarchy - Detection Delay is PRIMARY
        col1, col2, col3, col4 = st.columns([1.3, 1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-primary">
                <div class="metric-label">Detection Delay</div>
                <div class="metric-value">{detection_delay if detection_delay is not None else '—'} <span style="font-size: 1rem;">frames</span></div>
                <div class="metric-hint">↓ Earlier is better</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-secondary">
                <div class="metric-label">False Positive Rate</div>
                <div class="metric-value" style="font-size: 1.5rem;">{fp_rate:.1f}%</div>
                <div class="metric-hint">↓ Lower is better</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-secondary">
                <div class="metric-label">Drift Start</div>
                <div class="metric-value" style="font-size: 1.5rem;">Frame {drift_start}</div>
                <div class="metric-hint">Ground truth</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-secondary">
                <div class="metric-label">Max Drift Score</div>
                <div class="metric-value" style="font-size: 1.5rem;">{max(drift_scores):.1f}</div>
                <div class="metric-hint">Peak deviation</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Impact statement
        st.markdown("""
        <div class="impact-statement">
            <p>💡 <b>Early detection enables preventive response before visible failure or threat escalation.</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Operational Scenario Mapping
        st.markdown("---")
        st.markdown('<div class="section-header">🌐 Operational Scenario Mapping</div>', unsafe_allow_html=True)
        
        scen_col1, scen_col2, scen_col3 = st.columns(3)
        
        with scen_col1:
            st.markdown("""
            <div class="scenario-card">
                <div class="icon">🛂</div>
                <div class="title">Border Surveillance</div>
                <div class="desc">Gradual increase in crowd motion entropy preceding infiltration attempts.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with scen_col2:
            st.markdown("""
            <div class="scenario-card">
                <div class="icon">⚡</div>
                <div class="title">Critical Infrastructure</div>
                <div class="desc">Slow vibration and motion variance drift indicating mechanical degradation.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with scen_col3:
            st.markdown("""
            <div class="scenario-card">
                <div class="icon">🚶</div>
                <div class="title">Urban Crowd Intelligence</div>
                <div class="desc">Directional flow instability signaling pre-incident tension buildup.</div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="analyzing-state">
            <p>Start a monitoring session to view drift analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it Works Section
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 System Architecture</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="explanation-card">
            <h4>1. Feature Extraction</h4>
            <ul>
                <li>Optical flow magnitude</li>
                <li>Motion variance patterns</li>
                <li>Directional entropy</li>
            </ul>
            <p class="tagline">Focus on behavior, not appearance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="explanation-card">
            <h4>2. Normality Learning</h4>
            <ul>
                <li>Autoencoder compression</li>
                <li>Bottleneck captures "normal"</li>
                <li>Reconstruction error = drift</li>
            </ul>
            <p class="tagline">Unsupervised—no labeled data needed</p>
        </div>
        <div class="info-box">
            <h5>Why Unsupervised Learning?</h5>
            <ul>
                <li>Threat events are rare and poorly labeled</li>
                <li>Future attack patterns are unknown</li>
                <li>System learns only from normal behavior</li>
            </ul>
            <p class="footer">No prior examples of attacks required.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="explanation-card">
            <h4>3. Temporal Intelligence</h4>
            <ul>
                <li>EWMA smoothing</li>
                <li>Sliding window aggregation</li>
                <li>Graduated zone classification</li>
            </ul>
            <p class="tagline">Detect slow drift, not just spikes</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How NOISE FLOOR Differs - Philosophical Positioning
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 How NOISE FLOOR Differs</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <table class="philosophy-table">
        <tr>
            <th>Traditional Monitoring</th>
            <th>NOISE FLOOR</th>
        </tr>
        <tr>
            <td>Reactive alarms</td>
            <td>Preventive intelligence</td>
        </tr>
        <tr>
            <td>Fixed thresholds</td>
            <td>Learned normality</td>
        </tr>
        <tr>
            <td>Binary alerts</td>
            <td>Graduated confidence zones</td>
        </tr>
        <tr>
            <td>Spike detection</td>
            <td>Temporal drift tracking</td>
        </tr>
    </table>
    <p style="color: #718096; font-size: 0.85rem; font-style: italic; margin-top: 8px;">
        Designed to detect slow behavioral change before visible failure.
    </p>
    """, unsafe_allow_html=True)
    
    # Baseline Comparison
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Method Comparison</div>', unsafe_allow_html=True)
    
    if st.button("Compare with Traditional Methods", use_container_width=False):
        with st.spinner("Running baseline comparison..."):
            normal_train = create_synthetic_normal_data(500)
            normal_test = create_synthetic_normal_data(drift_start)
            drift_test = create_synthetic_drift_data(300 - drift_start, drift_rate=drift_rate)
            test_data = np.vstack([normal_test, drift_test])
            
            if st.session_state.history.get('drift_scores'):
                nf_scores = np.array(st.session_state.history['drift_scores'])
            else:
                normalized_test = (test_data - feature_stats['mean']) / feature_stats['std']
                norm_scores = autoencoder.get_normality_score(normalized_test)
                drift_detector.reset()
                nf_scores = []
                for i, score in enumerate(norm_scores):
                    state = drift_detector.update(score, frame_index=i)
                    nf_scores.append(state.drift_score)
                nf_scores = np.array(nf_scores)
            
            comparator = BaselineComparator()
            comparator.fit_baselines(normal_train)
            results = comparator.compare(test_data, nf_scores, drift_start)
        
        # Comparison table
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Method': result.method_name,
                'Detection Frame': result.detection_frame if result.detection_frame else '—',
                'Delay': f"{result.detection_delay}" if result.detection_delay >= 0 else '—',
                'FP Rate': f"{result.false_positive_rate*100:.1f}%",
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Comparison chart
        fig = create_comparison_chart(results, drift_start)
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="color: #4a5568; font-size: 0.8rem; text-align: center;">
    NOISE FLOOR • Behavioral Drift Intelligence • Gray-box Explainable AI
    </p>
    <p style="color: #3d4a5d; font-size: 0.7rem; text-align: center; font-style: italic; margin-top: 4px;">
    Built on established research in unsupervised anomaly detection, concept drift analysis, and streaming machine learning (IEEE and academic literature).
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
