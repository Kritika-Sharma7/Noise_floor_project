"""
NOISE FLOOR - Premium Enterprise Dashboard
============================================
Defence-grade behavioral drift intelligence interface.

Redesigned with:
• Premium dark theme with glassmorphism
• Enterprise-grade visual hierarchy
• Intuitive real-time monitoring
• Palantir/Darktrace-inspired aesthetics

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
from src.ai_insights import insights_engine, get_drift_context, OPENAI_AVAILABLE

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
# PREMIUM DESIGN SYSTEM CSS
# =============================================================================
st.markdown("""
<style>
    /* ========================================
       DESIGN TOKENS
       ======================================== */
    :root {
        --bg-primary: #0F141B;
        --bg-secondary: #151A22;
        --bg-card: rgba(21, 26, 34, 0.8);
        --bg-card-hover: rgba(26, 32, 42, 0.9);
        --bg-glass: rgba(21, 26, 34, 0.6);
        
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-medium: rgba(255, 255, 255, 0.1);
        --border-accent: rgba(56, 189, 248, 0.3);
        
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-tertiary: #64748B;
        --text-muted: #475569;
        
        --accent-primary: #38BDF8;
        --accent-primary-glow: rgba(56, 189, 248, 0.15);
        --accent-secondary: #F59E0B;
        --accent-success: #10B981;
        --accent-success-glow: rgba(16, 185, 129, 0.2);
        --accent-warning: #F97316;
        --accent-critical: #EF4444;
        --accent-critical-soft: #F87171;
        
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 20px rgba(56, 189, 248, 0.1);
        
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 32px;
        
        --transition-fast: 0.15s ease;
        --transition-normal: 0.25s ease;
        --transition-slow: 0.4s ease;
    }

    /* ========================================
       GLOBAL STYLES
       ======================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding but keep sidebar toggle */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: auto !important;
    }
    
    /* Ensure sidebar collapse button is always visible and styled */
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
        z-index: 999999 !important;
    }
    
    [data-testid="stSidebarCollapsedControl"]:hover {
        background: var(--bg-glass) !important;
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: currentColor !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-medium);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-tertiary);
    }

    /* ========================================
       GLASSMORPHISM CARDS
       ======================================== */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        transition: all var(--transition-normal);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }
    
    .glass-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-medium);
        box-shadow: var(--shadow-lg);
    }
    
    .glass-card-elevated {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-medium);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        box-shadow: var(--shadow-md);
    }

    /* ========================================
       SYSTEM INTELLIGENCE BAR
       ======================================== */
    .intel-bar {
        background: linear-gradient(90deg, rgba(15, 20, 27, 0.95), rgba(21, 26, 34, 0.95));
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: var(--spacing-md) var(--spacing-lg);
        margin-bottom: var(--spacing-lg);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: var(--spacing-md);
    }
    
    .intel-bar-left {
        display: flex;
        align-items: center;
        gap: var(--spacing-lg);
    }
    
    .intel-bar-logo {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
    }
    
    .intel-bar-logo-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, var(--accent-primary), #0EA5E9);
        border-radius: var(--radius-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }
    
    .intel-bar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    .intel-bar-subtitle {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        font-weight: 400;
        margin-top: 2px;
    }
    
    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        transition: all var(--transition-normal);
    }
    
    .status-pill.live {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: var(--accent-success);
    }
    
    .status-pill.stable {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: var(--accent-primary);
    }
    
    .status-pill.drift {
        background: rgba(249, 115, 22, 0.15);
        border: 1px solid rgba(249, 115, 22, 0.3);
        color: var(--accent-warning);
    }
    
    .status-pill.alert {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: var(--accent-critical-soft);
    }
    
    /* Pulse Animation */
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        position: relative;
    }
    
    .pulse-dot::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: currentColor;
        animation: pulse-ring 2s ease-out infinite;
    }
    
    @keyframes pulse-ring {
        0% { transform: translate(-50%, -50%) scale(1); opacity: 0.8; }
        100% { transform: translate(-50%, -50%) scale(2.5); opacity: 0; }
    }
    
    /* Meta Info */
    .intel-meta {
        display: flex;
        align-items: center;
        gap: var(--spacing-lg);
        font-size: 0.75rem;
        color: var(--text-tertiary);
    }
    
    .intel-meta-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .intel-meta-label {
        color: var(--text-muted);
    }
    
    .intel-meta-value {
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* ========================================
       HERO MONITORING CARD
       ======================================== */
    .hero-card {
        background: linear-gradient(135deg, rgba(21, 26, 34, 0.9), rgba(15, 20, 27, 0.95));
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: var(--spacing-xl);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all var(--transition-normal);
    }
    
    .hero-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
        opacity: 0.5;
    }
    
    .hero-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-glow);
    }
    
    .hero-icon {
        width: 72px;
        height: 72px;
        margin: 0 auto var(--spacing-md);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        position: relative;
    }
    
    .hero-icon.idle {
        background: linear-gradient(135deg, rgba(100, 116, 139, 0.2), rgba(71, 85, 105, 0.2));
        border: 2px solid var(--text-tertiary);
        color: var(--text-secondary);
    }
    
    .hero-icon.running {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2));
        border: 2px solid var(--accent-success);
        color: var(--accent-success);
        animation: icon-glow 2s ease-in-out infinite;
    }
    
    @keyframes icon-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
        50% { box-shadow: 0 0 30px rgba(16, 185, 129, 0.4); }
    }
    
    .hero-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-xs);
    }
    
    .hero-subtitle {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        margin-bottom: var(--spacing-lg);
        line-height: 1.5;
    }

    /* ========================================
       PRIMARY CTA BUTTON
       ======================================== */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), #0EA5E9) !important;
        border: none !important;
        color: var(--bg-primary) !important;
        font-weight: 600 !important;
        padding: 14px 32px !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
        transition: all var(--transition-normal) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3) !important;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-medium) !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-card) !important;
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
    }

    /* ========================================
       CONFIDENCE METER (VERTICAL)
       ======================================== */
    .confidence-meter {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        height: 100%;
    }
    
    .meter-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--spacing-md);
    }
    
    .meter-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .meter-container {
        position: relative;
        height: 200px;
        display: flex;
        justify-content: center;
        margin: var(--spacing-md) 0;
    }
    
    .meter-bar {
        width: 32px;
        height: 100%;
        border-radius: 16px;
        background: linear-gradient(to top, 
            var(--accent-success) 0%, 
            var(--accent-success) 25%,
            var(--accent-secondary) 25%,
            var(--accent-secondary) 50%,
            var(--accent-warning) 50%,
            var(--accent-warning) 75%,
            var(--accent-critical) 75%,
            var(--accent-critical) 100%
        );
        position: relative;
        opacity: 0.3;
        overflow: hidden;
    }
    
    .meter-indicator {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        transition: all var(--transition-slow);
        z-index: 10;
    }
    
    .meter-indicator.normal {
        background: var(--accent-success);
        box-shadow: 0 0 24px rgba(16, 185, 129, 0.5);
        bottom: 0%;
    }
    
    .meter-indicator.watch {
        background: var(--accent-secondary);
        box-shadow: 0 0 24px rgba(245, 158, 11, 0.5);
        bottom: 33%;
    }
    
    .meter-indicator.warning {
        background: var(--accent-warning);
        box-shadow: 0 0 24px rgba(249, 115, 22, 0.5);
        bottom: 60%;
    }
    
    .meter-indicator.alert {
        background: var(--accent-critical);
        box-shadow: 0 0 24px rgba(239, 68, 68, 0.5);
        bottom: 85%;
    }
    
    .meter-labels {
        position: absolute;
        right: -80px;
        top: 0;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 8px 0;
    }
    
    .meter-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .meter-label.active {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .meter-explanation {
        text-align: center;
        padding-top: var(--spacing-md);
        border-top: 1px solid var(--border-subtle);
        margin-top: var(--spacing-md);
    }
    
    .meter-zone-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-xs);
    }
    
    .meter-zone-desc {
        font-size: 0.8rem;
        color: var(--text-tertiary);
        line-height: 1.4;
    }

    /* ========================================
       SECTION HEADERS
       ======================================== */
    .section-header {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        margin-bottom: var(--spacing-md);
        padding-bottom: var(--spacing-sm);
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .section-header-icon {
        width: 28px;
        height: 28px;
        border-radius: var(--radius-sm);
        background: var(--accent-primary-glow);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    
    .section-header-text {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: 0.3px;
    }

    /* ========================================
       METRIC CARDS
       ======================================== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: var(--spacing-md);
    }
    
    @media (max-width: 768px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    .metric-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        text-align: center;
        transition: all var(--transition-normal);
    }
    
    .metric-card:hover {
        border-color: var(--border-medium);
        transform: translateY(-2px);
    }
    
    .metric-card.primary {
        border-color: var(--accent-primary);
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(56, 189, 248, 0.05));
    }
    
    .metric-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: var(--spacing-xs);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-value.primary {
        color: var(--accent-primary);
    }
    
    .metric-unit {
        font-size: 0.9rem;
        font-weight: 400;
        color: var(--text-secondary);
    }
    
    .metric-hint {
        font-size: 0.65rem;
        color: var(--text-tertiary);
        margin-top: var(--spacing-xs);
    }

    /* ========================================
       EMPTY STATE
       ======================================== */
    .empty-state {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px dashed var(--border-medium);
        border-radius: var(--radius-lg);
        padding: var(--spacing-xl) var(--spacing-lg);
        text-align: center;
    }
    
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: var(--spacing-md);
        opacity: 0.5;
    }
    
    .empty-state-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: var(--spacing-xs);
    }
    
    .empty-state-desc {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        line-height: 1.5;
        max-width: 400px;
        margin: 0 auto;
    }

    /* ========================================
       PIPELINE VISUALIZATION
       ======================================== */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        padding: var(--spacing-md) 0;
        flex-wrap: wrap;
    }
    
    .pipeline-step {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: var(--spacing-md) var(--spacing-lg);
        text-align: center;
        transition: all var(--transition-normal);
        cursor: pointer;
        min-width: 180px;
    }
    
    .pipeline-step:hover {
        border-color: var(--accent-primary);
        background: var(--bg-card-hover);
        transform: translateY(-4px);
        box-shadow: var(--shadow-glow);
    }
    
    .pipeline-step-icon {
        width: 40px;
        height: 40px;
        margin: 0 auto var(--spacing-sm);
        border-radius: var(--radius-sm);
        background: var(--accent-primary-glow);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        transition: all var(--transition-normal);
    }
    
    .pipeline-step:hover .pipeline-step-icon {
        background: var(--accent-primary);
        transform: scale(1.1);
    }
    
    .pipeline-step-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-xs);
    }
    
    .pipeline-step-desc {
        font-size: 0.7rem;
        color: var(--text-tertiary);
    }
    
    .pipeline-arrow {
        color: var(--text-muted);
        font-size: 1.5rem;
        padding: 0 var(--spacing-sm);
        animation: arrow-flow 1.5s ease-in-out infinite;
    }
    
    @keyframes arrow-flow {
        0%, 100% { opacity: 0.3; transform: translateX(0); }
        50% { opacity: 0.8; transform: translateX(4px); }
    }

    /* ========================================
       CALLOUT CARD
       ======================================== */
    .callout-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(56, 189, 248, 0.02));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        margin: var(--spacing-md) 0;
    }
    
    .callout-header {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        margin-bottom: var(--spacing-sm);
    }
    
    .callout-icon {
        font-size: 1.1rem;
    }
    
    .callout-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--accent-primary);
        letter-spacing: 0.3px;
    }
    
    .callout-content {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    .callout-content ul {
        margin: var(--spacing-sm) 0 0 0;
        padding-left: var(--spacing-lg);
    }
    
    .callout-content li {
        margin-bottom: var(--spacing-xs);
    }

    /* ========================================
       COMPARISON CARDS
       ======================================== */
    .comparison-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--spacing-lg);
    }
    
    @media (max-width: 768px) {
        .comparison-container {
            grid-template-columns: 1fr;
        }
    }
    
    .comparison-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        transition: all var(--transition-normal);
    }
    
    .comparison-card.highlighted {
        border-color: var(--accent-primary);
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), var(--bg-glass));
    }
    
    .comparison-card-header {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        margin-bottom: var(--spacing-md);
        padding-bottom: var(--spacing-sm);
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .comparison-card-icon {
        font-size: 1.2rem;
    }
    
    .comparison-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .comparison-card.highlighted .comparison-card-title {
        color: var(--accent-primary);
    }
    
    .comparison-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .comparison-list li {
        display: flex;
        align-items: flex-start;
        gap: var(--spacing-sm);
        padding: var(--spacing-sm) 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .comparison-list li::before {
        content: '→';
        color: var(--text-muted);
        flex-shrink: 0;
    }
    
    .comparison-card.highlighted .comparison-list li::before {
        content: '✓';
        color: var(--accent-success);
    }

    /* ========================================
       SIDEBAR STYLES
       ======================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-primary), var(--bg-secondary));
        border-right: 1px solid var(--border-subtle);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: var(--text-secondary);
    }
    
    .sidebar-section {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-md);
    }
    
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: var(--spacing-sm);
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background: var(--bg-card) !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        margin-top: var(--spacing-sm);
    }
    
    .stSlider [data-testid="stTickBar"] > div {
        background: var(--accent-primary) !important;
    }
    
    .slider-helper {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-top: var(--spacing-xs);
        line-height: 1.4;
    }

    /* ========================================
       SCENARIO CARDS
       ======================================== */
    .scenario-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: var(--spacing-md);
    }
    
    @media (max-width: 768px) {
        .scenario-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .scenario-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        transition: all var(--transition-normal);
    }
    
    .scenario-card:hover {
        border-color: var(--border-medium);
        transform: translateY(-2px);
    }
    
    .scenario-icon {
        font-size: 1.5rem;
        margin-bottom: var(--spacing-sm);
    }
    
    .scenario-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-xs);
    }
    
    .scenario-desc {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        line-height: 1.5;
    }

    /* ========================================
       FOOTER
       ======================================== */
    .footer {
        text-align: center;
        padding: var(--spacing-lg) 0;
        border-top: 1px solid var(--border-subtle);
        margin-top: var(--spacing-xl);
    }
    
    .footer-text {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    .footer-brand {
        color: var(--text-tertiary);
        font-weight: 500;
    }

    /* ========================================
       UTILITIES
       ======================================== */
    .divider {
        height: 1px;
        background: var(--border-subtle);
        margin: var(--spacing-lg) 0;
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .shimmer {
        background: linear-gradient(90deg, var(--bg-glass) 0%, var(--bg-card) 50%, var(--bg-glass) 100%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Hide default streamlit elements */
    .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background: var(--bg-glass) !important;
        border-radius: var(--radius-md) !important;
    }
    
    [data-testid="stDataFrame"] > div {
        background: transparent !important;
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
        st.session_state.system_status = 'stable'
        st.session_state.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.ai_insights_cache = {}
        st.session_state.chat_history = []


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
@st.cache_resource
def initialize_models():
    """Initialize and train models on synthetic normal data."""
    normal_data = create_synthetic_normal_data(1000)
    
    train_mean = np.mean(normal_data, axis=0)
    train_std = np.std(normal_data, axis=0)
    train_std[train_std == 0] = 1
    train_normalized = (normal_data - train_mean) / train_std
    
    autoencoder = NormalityAutoencoder(input_dim=normal_data.shape[1])
    autoencoder.compile()
    autoencoder.train(train_normalized, epochs=50, verbose=0)
    
    drift_detector = DriftDetector(baseline_frames=50)
    zone_classifier = WatchZoneClassifier()
    feature_extractor = FeatureExtractor()
    feature_stats = {'mean': train_mean, 'std': train_std}
    
    return autoencoder, drift_detector, zone_classifier, feature_extractor, feature_stats


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================
def create_drift_chart(history: dict, thresholds: dict, drift_start: int) -> go.Figure:
    """Create premium drift score chart with zone visualization."""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.85, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )
    
    if not history['drift_scores']:
        return fig
    
    frames = list(range(len(history['drift_scores'])))
    drift_scores = history['drift_scores']
    max_score = max(drift_scores) if drift_scores else 5
    
    # Zone bands with softer colors
    zone_bands = [
        (0, thresholds['normal'], 'rgba(16, 185, 129, 0.06)', 'Normal'),
        (thresholds['normal'], thresholds['watch'], 'rgba(245, 158, 11, 0.06)', 'Watch'),
        (thresholds['watch'], thresholds['warning'], 'rgba(249, 115, 22, 0.06)', 'Warning'),
        (thresholds['warning'], max_score * 1.3, 'rgba(239, 68, 68, 0.06)', 'Alert'),
    ]
    
    for y0, y1, color, name in zone_bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, row=1, col=1)
    
    # Gradient fill under the line
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=drift_scores,
            mode='lines',
            name='Drift Score',
            line=dict(color='#38BDF8', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(56, 189, 248, 0.1)',
            hovertemplate='<b>Frame %{x}</b><br>Drift Score: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Threshold lines
    threshold_styles = [
        (thresholds['normal'], '#10B981', 'Normal'),
        (thresholds['watch'], '#F59E0B', 'Watch'),
        (thresholds['warning'], '#F97316', 'Warning'),
    ]
    
    for thresh_val, color, label in threshold_styles:
        fig.add_hline(
            y=thresh_val, 
            line_dash="dot", 
            line_color=color, 
            opacity=0.4,
            row=1, col=1
        )
    
    # Drift start marker
    fig.add_vline(
        x=drift_start,
        line_dash="dash",
        line_color="#EF4444",
        opacity=0.6,
        annotation_text="Drift Onset",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color="#94A3B8",
        row=1, col=1
    )
    
    # Zone timeline
    if history['zones']:
        zone_colors_map = {
            'NORMAL': '#10B981',
            'WATCH': '#F59E0B',
            'WARNING': '#F97316',
            'ALERT': '#EF4444'
        }
        zone_colors_list = [zone_colors_map.get(z, '#64748B') for z in history['zones']]
        
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
        height=420,
        margin=dict(l=60, r=40, t=20, b=40),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 20, 27, 0.5)',
        font=dict(color='#94A3B8', family='Inter'),
        hovermode='x unified',
    )
    
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.03)',
        zerolinecolor='rgba(255,255,255,0.03)',
        tickfont=dict(size=10),
        showgrid=True,
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.03)',
        zerolinecolor='rgba(255,255,255,0.03)',
        tickfont=dict(size=10),
        showgrid=True,
    )
    
    fig.update_yaxes(title_text="Drift Score", title_font_size=11, row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(title_text="Frame", title_font_size=11, row=2, col=1)
    
    return fig


def create_comparison_chart(results: dict, drift_start: int) -> go.Figure:
    """Create baseline comparison chart."""
    fig = go.Figure()
    
    method_styles = {
        'noise_floor': {'color': '#38BDF8', 'width': 3, 'dash': 'solid'},
        'isolation_forest': {'color': '#64748B', 'width': 1.5, 'dash': 'dot'},
        'threshold': {'color': '#F59E0B', 'width': 1.5, 'dash': 'dot'},
        'one_class_svm': {'color': '#EF4444', 'width': 1.5, 'dash': 'dot'},
    }
    
    for name, result in results.items():
        scores = result.scores
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        style = method_styles.get(name, {'color': '#475569', 'width': 1, 'dash': 'dot'})
        
        fig.add_trace(go.Scatter(
            x=list(range(len(scores))),
            y=scores_normalized,
            mode='lines',
            name=result.method_name,
            line=dict(color=style['color'], width=style['width'], dash=style['dash']),
            opacity=0.9 if name == 'noise_floor' else 0.5
        ))
    
    fig.add_vline(x=drift_start, line_dash="dash", line_color="#EF4444", opacity=0.5)
    
    fig.update_layout(
        height=350,
        margin=dict(l=60, r=20, t=30, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 20, 27, 0.5)',
        font=dict(color='#94A3B8', family='Inter'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis_title="Frame",
        yaxis_title="Normalized Score",
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.03)')
    
    return fig


def get_zone_info(zone_name: str) -> dict:
    """Get zone display information."""
    zones = {
        'NORMAL': {
            'icon': '🟢', 
            'color': 'var(--accent-success)',
            'desc': 'System operating within learned behavioral patterns.',
            'action': 'Continue monitoring'
        },
        'WATCH': {
            'icon': '🟡', 
            'color': 'var(--accent-secondary)',
            'desc': 'Minor deviation detected. Early-stage drift possible.',
            'action': 'Increase observation frequency'
        },
        'WARNING': {
            'icon': '🟠', 
            'color': 'var(--accent-warning)',
            'desc': 'Significant behavioral drift in progress.',
            'action': 'Prepare intervention protocols'
        },
        'ALERT': {
            'icon': '🔴', 
            'color': 'var(--accent-critical)',
            'desc': 'Critical drift threshold exceeded.',
            'action': 'Immediate review recommended'
        },
    }
    return zones.get(zone_name, zones['NORMAL'])


# =============================================================================
# DEMO SIMULATION
# =============================================================================
def run_demo_simulation(autoencoder, drift_detector, zone_classifier, feature_stats, 
                       drift_start: int, drift_rate: float):
    """Run simulation with synthetic data."""
    normal_data = create_synthetic_normal_data(drift_start)
    drift_data = create_synthetic_drift_data(300 - drift_start, drift_rate=drift_rate)
    all_data = np.vstack([normal_data, drift_data])
    
    normalized_data = (all_data - feature_stats['mean']) / feature_stats['std']
    
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
    
    # Initialize models
    autoencoder, drift_detector, zone_classifier, feature_extractor, feature_stats = initialize_models()
    
    # =========================================================================
    # SYSTEM INTELLIGENCE BAR
    # =========================================================================
    # Determine current status
    if st.session_state.history.get('results'):
        latest_zone = str(st.session_state.history['results'][-1]['zone']).upper()
        if latest_zone == 'ALERT':
            status_class = 'alert'
            status_text = 'Alert Active'
        elif latest_zone in ['WARNING', 'WATCH']:
            status_class = 'drift'
            status_text = 'Drift Detected'
        else:
            status_class = 'live'
            status_text = 'Live · Stable'
    else:
        status_class = 'stable'
        status_text = 'Ready'
    
    st.markdown(f"""
    <div class="intel-bar">
        <div class="intel-bar-left">
            <div class="intel-bar-logo">
                <div class="intel-bar-logo-icon">📡</div>
                <div>
                    <div class="intel-bar-title">NOISE FLOOR</div>
                    <div class="intel-bar-subtitle">Behavioral Drift Intelligence</div>
                </div>
            </div>
            <div class="status-pill {status_class}">
                <div class="pulse-dot"></div>
                {status_text}
            </div>
        </div>
        <div class="intel-meta">
            <div class="intel-meta-item">
                <span class="intel-meta-label">Model</span>
                <span class="intel-meta-value">v1.0.0</span>
            </div>
            <div class="intel-meta-item">
                <span class="intel-meta-label">Baseline</span>
                <span class="intel-meta-value">{st.session_state.last_trained}</span>
            </div>
            <div class="intel-meta-item">
                <span class="intel-meta-label">AI</span>
                <span class="intel-meta-value" style="color: {'#10B981' if OPENAI_AVAILABLE else '#EF4444'};">{'● Active' if OPENAI_AVAILABLE else '○ Inactive'}</span>
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
            <div style="font-size: 0.7rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;">Control Panel</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Drift Simulation Section
        st.markdown("""<div class="sidebar-section">
            <div class="sidebar-section-title">Drift Simulation</div>
        </div>""", unsafe_allow_html=True)
        
        drift_start = st.slider(
            "Drift Onset Frame", 
            50, 200, 100,
            help="Frame where behavioral drift begins"
        )
        st.markdown('<p class="slider-helper">Earlier onset = longer normal baseline period</p>', unsafe_allow_html=True)
        
        drift_rate = st.slider(
            "Drift Intensity", 
            0.005, 0.05, 0.015, 0.005,
            help="Rate of behavioral change per frame"
        )
        intensity_label = "Low" if drift_rate < 0.01 else "Medium" if drift_rate < 0.03 else "High"
        st.markdown(f'<p class="slider-helper">Intensity: <strong>{intensity_label}</strong> · Faster = more aggressive drift</p>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sensitivity Thresholds Section
        st.markdown("""<div class="sidebar-section">
            <div class="sidebar-section-title">Sensitivity Thresholds</div>
        </div>""", unsafe_allow_html=True)
        
        normal_thresh = st.slider("Normal → Watch", 0.5, 2.0, 1.5, 0.1)
        watch_thresh = st.slider("Watch → Warning", 1.5, 3.0, 2.0, 0.1)
        warning_thresh = st.slider("Warning → Alert", 2.0, 4.0, 2.5, 0.1)
        
        st.markdown('<p class="slider-helper">Lower thresholds = higher sensitivity</p>', unsafe_allow_html=True)
        
        thresholds = {
            'normal': normal_thresh,
            'watch': watch_thresh,
            'warning': warning_thresh,
        }
        
        zone_classifier.thresholds = thresholds
    
    # =========================================================================
    # MAIN CONTENT AREA
    # =========================================================================
    
    # Hero Section - Monitoring Card & Zone Status
    col_hero, col_meter = st.columns([2, 1])
    
    with col_hero:
        is_running = st.session_state.history.get('results') is not None
        icon_class = "running" if is_running else "idle"
        icon_emoji = "📊" if is_running else "🎯"
        
        st.markdown(f"""
        <div class="hero-card fade-in">
            <div class="hero-icon {icon_class}">{icon_emoji}</div>
            <div class="hero-title">{'Monitoring Active' if is_running else 'Start Monitoring Session'}</div>
            <div class="hero-subtitle">
                {'Real-time drift analysis in progress. Temporal patterns are being evaluated against the learned baseline.' 
                 if is_running else 
                 'Initialize behavioral drift detection. The system will simulate gradual anomaly injection to demonstrate early-warning capabilities.'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_btn, col_spacer = st.columns([1, 2])
        with col_btn:
            btn_label = "🔄 Reset Session" if is_running else "▶ Begin Analysis"
            if st.button(btn_label, type="primary", use_container_width=True):
                if is_running:
                    st.session_state.history = {
                        'drift_scores': [], 'zones': [], 'raw_scores': [], 'timestamps': []
                    }
                    st.rerun()
                else:
                    with st.spinner(""):
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
                        time.sleep(0.3)
                    st.rerun()
    
    with col_meter:
        # Confidence Meter
        if st.session_state.history.get('results'):
            latest = st.session_state.history['results'][-1]
            zone_name = str(latest['zone']).upper()
            zone_info = get_zone_info(zone_name)
            meter_class = zone_name.lower()
        else:
            zone_name = "—"
            zone_info = {'icon': '○', 'desc': 'Start monitoring to view zone status', 'action': ''}
            meter_class = "normal"
        
        # Calculate position percentage for indicator
        position_map = {'normal': '5%', 'watch': '35%', 'warning': '60%', 'alert': '88%'}
        indicator_pos = position_map.get(meter_class, '5%')
        
        st.markdown(f"""
        <div class="confidence-meter">
            <div class="meter-header">
                <span class="meter-title">Zone Status</span>
            </div>
            <div class="meter-container">
                <div class="meter-bar"></div>
                <div class="meter-indicator {meter_class}" style="bottom: {indicator_pos};">
                    {zone_info['icon'] if zone_name != '—' else '○'}
                </div>
            </div>
            <div class="meter-explanation">
                <div class="meter-zone-name">{zone_name}</div>
                <div class="meter-zone-desc">{zone_info['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI-powered zone insight
        if st.session_state.history.get('results') and OPENAI_AVAILABLE:
            latest = st.session_state.history['results'][-1]
            zone_insight = insights_engine.generate_zone_insight(
                str(latest['zone']).upper(),
                latest['drift_score'],
                latest['confidence']
            )
            st.markdown(f"""
            <div class="callout-card" style="margin-top: 12px; padding: 12px;">
                <div class="callout-header" style="margin-bottom: 4px;">
                    <span class="callout-icon">🤖</span>
                    <span class="callout-title" style="font-size: 0.75rem;">AI Insight</span>
                </div>
                <div class="callout-content" style="font-size: 0.8rem;">{zone_insight}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # =========================================================================
    # DRIFT SCORE TIMELINE
    # =========================================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📈</div>
        <span class="section-header-text">Drift Score Timeline</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.history.get('drift_scores'):
        fig = create_drift_chart(st.session_state.history, thresholds, drift_start)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # AI-powered graph summary
        drift_context = get_drift_context(st.session_state.history, drift_start, thresholds)
        graph_summary = insights_engine.generate_graph_summary(drift_context)
        st.markdown(f"""
        <div class="callout-card" style="margin-top: -8px; margin-bottom: 16px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.02)); border-color: rgba(16, 185, 129, 0.2);">
            <div class="callout-header">
                <span class="callout-icon">🤖</span>
                <span class="callout-title" style="color: #10B981;">AI Graph Analysis</span>
            </div>
            <div class="callout-content">{graph_summary}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detection Metrics
        st.markdown("""
        <div class="section-header" style="margin-top: 24px;">
            <div class="section-header-icon">📊</div>
            <span class="section-header-text">Detection Performance</span>
        </div>
        """, unsafe_allow_html=True)
        
        drift_scores = st.session_state.history['drift_scores']
        zones = st.session_state.history['zones']
        
        # Calculate metrics
        first_detection_idx = None
        for i in range(drift_start, len(zones)):
            if zones[i] in ['WATCH', 'WARNING', 'ALERT']:
                first_detection_idx = i
                break
        
        false_positives = sum(1 for i in range(min(drift_start, len(zones))) if zones[i] in ['WATCH', 'WARNING', 'ALERT'])
        fp_rate = (false_positives / drift_start * 100) if drift_start > 0 else 0
        detection_delay = (first_detection_idx - drift_start) if first_detection_idx else None
        
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card primary">
                <div class="metric-label">Detection Delay</div>
                <div class="metric-value primary">{detection_delay if detection_delay is not None else '—'}<span class="metric-unit"> frames</span></div>
                <div class="metric-hint">↓ Earlier is better</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">False Positive Rate</div>
                <div class="metric-value">{fp_rate:.1f}<span class="metric-unit">%</span></div>
                <div class="metric-hint">↓ Lower is better</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Drift Onset</div>
                <div class="metric-value">Frame {drift_start}</div>
                <div class="metric-hint">Ground truth</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak Drift Score</div>
                <div class="metric-value">{max(drift_scores):.1f}</div>
                <div class="metric-hint">Max deviation</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI-powered drift explanation
        drift_context = get_drift_context(st.session_state.history, drift_start, thresholds)
        drift_explanation = insights_engine.generate_drift_explanation(drift_context)
        metrics_summary = insights_engine.generate_metrics_summary(
            detection_delay if detection_delay else 0,
            fp_rate,
            max(drift_scores),
            drift_start
        )
        
        st.markdown(f"""
        <div class="callout-card">
            <div class="callout-header">
                <span class="callout-icon">🤖</span>
                <span class="callout-title">AI Analysis Summary</span>
            </div>
            <div class="callout-content">
                <p><strong>Drift Status:</strong> {drift_explanation}</p>
                <p style="margin-top: 8px;"><strong>Performance:</strong> {metrics_summary}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-title">No Data Yet</div>
            <div class="empty-state-desc">
                Start a monitoring session to visualize drift scores over time. 
                The timeline will show behavioral deviation from the learned baseline, 
                with zone transitions highlighted as they occur.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # =========================================================================
    # SYSTEM ARCHITECTURE - PIPELINE VISUALIZATION
    # =========================================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🔧</div>
        <span class="section-header-text">System Architecture</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="pipeline-step-icon">👁️</div>
            <div class="pipeline-step-title">Feature Extraction</div>
            <div class="pipeline-step-desc">Motion, flow, entropy</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">🧠</div>
            <div class="pipeline-step-title">Normality Learning</div>
            <div class="pipeline-step-desc">Autoencoder baseline</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">⏱️</div>
            <div class="pipeline-step-title">Temporal Intelligence</div>
            <div class="pipeline-step-desc">EWMA smoothing, zones</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Why Unsupervised Learning - Callout Card
    st.markdown("""
    <div class="callout-card" style="margin-top: 24px;">
        <div class="callout-header">
            <span class="callout-icon">🎓</span>
            <span class="callout-title">Design Rationale: Why Unsupervised Learning?</span>
        </div>
        <div class="callout-content">
            <ul>
                <li><strong>Threat events are rare</strong> — labeled anomaly data is scarce and often unrepresentative</li>
                <li><strong>Future patterns are unknown</strong> — supervised models can't predict novel attack vectors</li>
                <li><strong>Normal is learnable</strong> — consistent baseline behavior provides robust deviation signal</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # =========================================================================
    # HOW NOISE FLOOR DIFFERS - COMPARISON CARDS
    # =========================================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">⚡</div>
        <span class="section-header-text">How NOISE FLOOR Differs</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="comparison-container">
        <div class="comparison-card">
            <div class="comparison-card-header">
                <span class="comparison-card-icon">📊</span>
                <span class="comparison-card-title">Traditional Monitoring</span>
            </div>
            <ul class="comparison-list">
                <li>Reactive alarms after threshold breach</li>
                <li>Fixed thresholds require manual tuning</li>
                <li>Binary alerts: normal or anomaly</li>
                <li>Spike detection misses gradual drift</li>
            </ul>
        </div>
        <div class="comparison-card highlighted">
            <div class="comparison-card-header">
                <span class="comparison-card-icon">📡</span>
                <span class="comparison-card-title">NOISE FLOOR</span>
            </div>
            <ul class="comparison-list">
                <li>Preventive intelligence before failure</li>
                <li>Learned normality adapts to context</li>
                <li>Graduated confidence zones</li>
                <li>Temporal drift tracking with EWMA</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # =========================================================================
    # OPERATIONAL SCENARIOS
    # =========================================================================
    if st.session_state.history.get('results'):
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🌐</div>
            <span class="section-header-text">Operational Scenarios</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="scenario-grid">
            <div class="scenario-card">
                <div class="scenario-icon">🛂</div>
                <div class="scenario-title">Border Surveillance</div>
                <div class="scenario-desc">Detect gradual crowd motion entropy changes preceding infiltration attempts.</div>
            </div>
            <div class="scenario-card">
                <div class="scenario-icon">⚡</div>
                <div class="scenario-title">Critical Infrastructure</div>
                <div class="scenario-desc">Monitor vibration and motion variance drift indicating mechanical degradation.</div>
            </div>
            <div class="scenario-card">
                <div class="scenario-icon">🏙️</div>
                <div class="scenario-title">Urban Intelligence</div>
                <div class="scenario-desc">Track directional flow instability signaling pre-incident tension buildup.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # =========================================================================
    # METHOD COMPARISON
    # =========================================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🔬</div>
        <span class="section-header-text">Baseline Method Comparison</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Run Comparison Analysis", use_container_width=False):
        with st.spinner("Evaluating baseline methods..."):
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
        
        # Results table
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
        
        fig = create_comparison_chart(results, drift_start)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # AI-powered comparison insight
        comparison_insight = insights_engine.generate_comparison_insight(results, drift_start)
        st.markdown(f"""
        <div class="callout-card" style="margin-top: 16px;">
            <div class="callout-header">
                <span class="callout-icon">🤖</span>
                <span class="callout-title">AI Comparison Analysis</span>
            </div>
            <div class="callout-content">{comparison_insight}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # =========================================================================
    # AI ASSISTANT CHAT
    # =========================================================================
    if OPENAI_AVAILABLE and st.session_state.history.get('results'):
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">💬</div>
            <span class="section-header-text">AI Assistant</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="callout-card" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(139, 92, 246, 0.02)); border-color: rgba(139, 92, 246, 0.2); margin-bottom: 16px;">
            <div class="callout-content" style="font-size: 0.85rem; color: #A78BFA;">
                Ask questions about the current monitoring session, drift patterns, or system behavior.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input(
            "Ask the AI assistant",
            placeholder="e.g., Why did drift increase at frame 120?",
            label_visibility="collapsed"
        )
        
        if user_question:
            drift_context = get_drift_context(st.session_state.history, drift_start, thresholds)
            with st.spinner("Analyzing..."):
                ai_response = insights_engine.chat_query(user_question, drift_context)
            
            st.markdown(f"""
            <div class="glass-card" style="margin-top: 12px;">
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="font-size: 1.5rem;">🤖</div>
                    <div>
                        <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 4px;">AI Assistant</div>
                        <div style="color: #F1F5F9; font-size: 0.9rem; line-height: 1.6;">{ai_response}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("""
    <div class="footer">
        <div class="footer-text">
            <span class="footer-brand">NOISE FLOOR</span> · Behavioral Drift Intelligence · Gray-box Explainable AI
            <span style="margin-left: 16px; color: #475569;">Powered by OpenAI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
