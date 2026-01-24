"""
NOISE FLOOR - Configuration Module
===================================
Central configuration for all system parameters.
Gray-box design: All hyperparameters are exposed and documented.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# =============================================================================
# VIDEO PROCESSING CONFIGURATION
# =============================================================================
# Frame extraction settings
FRAME_WIDTH = 224           # Resize width for processing
FRAME_HEIGHT = 224          # Resize height for processing
FRAME_SKIP = 2              # Process every Nth frame (reduces computation)
MAX_FRAMES = None           # None = process all frames

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION (Gray-Box: Behavioral Features)
# =============================================================================
"""
WHY THESE FEATURES?
-------------------
We extract MOTION-based features, not appearance-based features.
This means we learn HOW things move, not WHAT they look like.

1. Optical Flow Magnitude: Measures speed of movement
   - High magnitude = fast motion
   - Normal range established during training
   
2. Motion Variance: Measures consistency of movement
   - Low variance = uniform motion (e.g., regular walking)
   - High variance = chaotic motion (e.g., running, fighting)
   
3. Directional Entropy: Measures diversity of movement directions
   - Low entropy = everyone moving same direction
   - High entropy = random/scattered movement patterns
   
4. Motion Energy: Total amount of movement in frame
   - Represents overall activity level
"""

# Optical Flow parameters (Farneback method)
OPTICAL_FLOW_PARAMS = {
    'pyr_scale': 0.5,       # Pyramid scale
    'levels': 3,            # Number of pyramid levels
    'winsize': 15,          # Averaging window size
    'iterations': 3,        # Iterations at each pyramid level
    'poly_n': 5,            # Size of pixel neighborhood
    'poly_sigma': 1.2,      # Gaussian standard deviation
}

# Feature vector dimension after extraction
FEATURE_DIM = 64            # Compressed feature dimension

# =============================================================================
# AUTOENCODER CONFIGURATION (Gray-Box: Normality Model)
# =============================================================================
"""
WHY AUTOENCODER?
----------------
The autoencoder learns to COMPRESS and RECONSTRUCT normal behavior.
- During training: Learns efficient representation of normal patterns
- During inference: High reconstruction error = deviation from normal

Architecture is intentionally simple for explainability:
- Encoder: Compresses input to bottleneck
- Bottleneck: Compact representation of "normality"
- Decoder: Reconstructs from bottleneck

RECONSTRUCTION ERROR = How different current behavior is from learned normal
"""

AUTOENCODER_CONFIG = {
    'input_dim': FEATURE_DIM,
    'encoding_dims': [48, 32, 16],   # Encoder layer sizes
    'bottleneck_dim': 8,              # Compressed representation size
    'activation': 'relu',
    'output_activation': 'linear',
    'dropout_rate': 0.1,
}

# Training parameters
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'validation_split': 0.15,
    'early_stopping_patience': 10,
    'min_delta': 0.0001,
}

# =============================================================================
# DRIFT DETECTION CONFIGURATION (Gray-Box: Temporal Analysis)
# =============================================================================
"""
WHY DRIFT DETECTION, NOT ANOMALY DETECTION?
-------------------------------------------
Traditional anomaly detection asks: "Is this instant abnormal?"
Drift detection asks: "Is behavior gradually changing over time?"

Key insight: Threats often manifest as SLOW changes, not sudden spikes.
- Insider threats: Gradual behavioral shift
- Equipment degradation: Slow performance decline
- Crowd dynamics: Gradual tension buildup

EWMA (Exponentially Weighted Moving Average):
- Gives more weight to recent observations
- Smooths out noise while tracking trends
- Alpha parameter controls responsiveness

SLIDING WINDOW:
- Aggregates behavior over time window
- Reduces false positives from momentary spikes
- Window size defines "memory" of the system
"""

DRIFT_CONFIG = {
    # Sliding window for temporal aggregation
    'window_size': 30,              # Number of frames in sliding window
    'window_step': 5,               # Step size for sliding window
    
    # EWMA parameters
    'ewma_alpha': 0.1,              # Smoothing factor (0.1 = slow adaptation)
    'ewma_span': 20,                # Alternative: span-based calculation
    
    # Trend detection
    'trend_window': 50,             # Window for trend slope calculation
    'trend_threshold': 0.02,        # Minimum slope to consider as drift
    
    # Baseline establishment
    'baseline_frames': 200,         # Frames to establish normal baseline
}

# =============================================================================
# WATCH ZONE CONFIGURATION (Gray-Box: Alert Levels)
# =============================================================================
"""
WHY WATCH ZONES, NOT BINARY ALERTS?
-----------------------------------
Binary alerts (normal/abnormal) cause:
- Alert fatigue from false positives
- Delayed response (waiting for "certain" anomaly)
- No early warning capability

Watch Zones provide:
- Graduated confidence levels
- Early warning for preventive action
- Reduced false positive impact

Zone thresholds are percentile-based:
- Established during training on normal data
- Adapt to the specific environment
- Explainable as "X standard deviations from normal"
"""

ZONE_CONFIG = {
    # Zone thresholds (in standard deviations from mean)
    'normal_threshold': 1.5,        # Below this = definitely normal
    'watch_threshold': 2.0,         # Above this = start watching
    'warning_threshold': 2.5,       # Above this = warning
    'alert_threshold': 3.0,         # Above this = alert
    
    # Zone persistence (prevents flickering)
    'min_zone_duration': 10,        # Minimum frames to confirm zone change
    
    # Colors for visualization
    'zone_colors': {
        'normal': '#00C851',        # Green
        'watch': '#FFBB33',         # Yellow
        'warning': '#FF8800',       # Orange
        'alert': '#FF4444',         # Red
    }
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
DASHBOARD_CONFIG = {
    'refresh_rate': 100,            # Milliseconds between updates
    'graph_history': 500,           # Number of points to show in graph
    'video_fps': 15,                # Playback FPS
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
"""
UCSD PEDESTRIAN DATASET
-----------------------
A real surveillance dataset commonly used for anomaly detection research.
- Contains normal pedestrian walking footage
- Test set includes anomalies (bikes, skaters, etc.)
- Ideal for demonstrating drift detection

Download: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
"""

DATASET_CONFIG = {
    'name': 'UCSD_Pedestrian',
    'url': 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz',
    'train_folder': 'Train',
    'test_folder': 'Test',
}
