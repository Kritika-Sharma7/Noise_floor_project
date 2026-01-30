"""
NOISE FLOOR - Configuration Module
===================================
Defense-grade behavioral drift intelligence system.
Designed for border surveillance and high-security perimeters.

TECHNOLOGY READINESS LEVEL: TRL-4
Lab-validated prototype for decision-support intelligence.
This is NOT an autonomous system - AI assists operators, it does NOT replace them.

SYSTEM PHILOSOPHY:
- "Defense systems manage CONFIDENCE, not panic."
- "AI assists operators, it does NOT replace them."
- "Baseline adaptation is human-gated."

Gray-box design: All hyperparameters are exposed and documented.
This is NOT a toy ML demo - this is mission-critical configuration.

ARCHITECTURE:
    Video/Sensor Ingestion → Context Augmentation → Feature Extraction
    → LSTM-VAE → Drift Intelligence → Risk Zones → Explainability
    → Human-in-the-Loop Feedback
"""

import os
from pathlib import Path

# =============================================================================
# DATA MODE CONFIGURATION
# =============================================================================
"""
DATA_MODE: Switch between synthetic and real video sources.

"synthetic" - Controlled testing with generated behavioral data
            - Clean demonstrations for judges
            - Benchmarking detection delay
            - Reproducible experiments

"real_video" - Real surveillance footage processing
             - UCSD Anomaly Detection Dataset as proxy for border CCTV
             - OpenCV optical flow feature extraction
             - Production-realistic demonstration

"The intelligence pipeline is feature-driven, not data-source-driven."
Both modes produce identical 24-feature vectors for LSTM-VAE processing.
"""

DATA_MODE = "real_video"  # "synthetic" | "real_video"

# Real video data paths - UCSD Anomaly Detection Dataset (Local)
UCSD_DATASET_PATH = r"C:\Users\raghav\NoiseFloor_IPEC\Noise_floor_project\data\UCSD_Anomaly_Dataset.v1p2"
UCSD_SUBSET = "ped1"                        # "ped1" or "ped2"
CUSTOM_VIDEO_PATH = None                    # Path to custom surveillance video

# =============================================================================
# BASELINE FREEZE CONFIGURATION
# =============================================================================
"""
BASELINE FREEZE: Human-gated baseline management.

"Baseline adaptation is human-gated."

The baseline defines what is NORMAL. Protecting this from corruption
is critical for system integrity. This prevents:
- Data poisoning attacks
- Drift normalization during slow-burn intrusions
- Autonomous baseline corruption

Adaptation occurs ONLY with explicit operator approval.
"""

BASELINE_FREEZE_CONFIG = {
    # Learning phase
    'learning_window': 200,             # Frames for initial baseline establishment
    'freeze_after_learning': True,      # Auto-freeze after learning (recommended)
    
    # Adaptation controls
    'adaptation_learning_rate': 0.01,   # Very slow adaptation (safety)
    'max_adaptation_rate': 0.05,        # Cap on single update magnitude
    'min_samples_for_adaptation': 10,   # Require multiple operator confirmations
    
    # Storage
    'baseline_storage_path': "./baseline_data",
    'auto_save': True,
    'auto_load': True,
}

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
FEEDBACK_DIR = BASE_DIR / "feedback_data"
BASELINE_DIR = BASE_DIR / "baseline_data"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, FEEDBACK_DIR, BASELINE_DIR]:
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

# =============================================================================
# LSTM-VAE CONFIGURATION (Temporal Normality Learning)
# =============================================================================
"""
WHY LSTM-VAE?
-------------
Traditional autoencoders process single frames independently.
LSTM-VAE learns TEMPORAL patterns - how behavior evolves over time.

- LSTM layers capture sequential dependencies
- VAE provides probabilistic latent space
- Trained on NORMAL data only (unsupervised)
- Reconstruction error + KL divergence = normality score

For border surveillance: Threats emerge GRADUALLY.
We need to detect slow behavioral shifts, not just spikes.
"""

LSTM_VAE_CONFIG = {
    'input_dim': 24,                # Number of behavioral features
    'hidden_dim': 64,               # LSTM hidden state size
    'latent_dim': 16,               # VAE latent space dimension
    'num_layers': 2,                # Number of LSTM layers
    'sequence_length': 10,          # Temporal window (frames)
    'dropout': 0.1,
    
    # Training
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'kl_weight': 0.1,               # KL divergence weight in loss
}

# =============================================================================
# DRIFT INTELLIGENCE CONFIGURATION
# =============================================================================
"""
DRIFT INTELLIGENCE LAYER
------------------------
Converts raw ML outputs into decision-ready intelligence.

KL Divergence: Measures how current distribution differs from baseline
EWMA Smoothing: Removes noise while tracking gradual trends
Rolling Z-Score: Contextualizes current deviation
Trend Persistence: Confirms sustained drift vs momentary noise

OUTPUT: Threat Deviation Index (TDI) 0-100
- 0-25: Normal
- 25-50: Watch
- 50-75: Warning
- 75-100: Critical
"""

DRIFT_INTELLIGENCE_CONFIG = {
    # EWMA (Exponentially Weighted Moving Average)
    'ewma_alpha': 0.1,              # Smoothing factor (lower = smoother)
    
    # Rolling statistics window
    'rolling_window': 50,           # Frames for rolling stats
    
    # Trend detection
    'trend_window': 20,             # Frames for trend calculation
    'trend_threshold_rising': 0.05, # Slope to consider "rising"
    'trend_threshold_falling': -0.03, # Slope to consider "falling"
    'trend_persistence_required': 5, # Consecutive frames to confirm trend
    
    # TDI calculation weights
    'weight_kl_divergence': 0.4,
    'weight_z_score': 0.3,
    'weight_trend_persistence': 0.2,
    'weight_ewma_deviation': 0.1,
}

# =============================================================================
# RISK ZONE CONFIGURATION
# =============================================================================
"""
RISK ZONES
----------
Graduated response system - NOT binary alerts.

NORMAL (Green): Operating within learned baseline
WATCH (Yellow): Early deviation detected - increase observation
WARNING (Orange): Significant drift - prepare response protocols
CRITICAL (Red): Threshold exceeded - immediate action required

HYSTERESIS: Prevents zone flickering
- Requires sustained deviation to enter higher zone
- Easier to exit than enter (safety margin)
"""

RISK_ZONE_CONFIG = {
    # TDI thresholds for zone transitions
    'watch_threshold': 0.25,        # Enter WATCH above 25
    'warning_threshold': 0.50,      # Enter WARNING above 50
    'critical_threshold': 0.75,     # Enter CRITICAL above 75
    
    # Exit thresholds (with hysteresis)
    'watch_exit': 0.20,
    'warning_exit': 0.40,
    'critical_exit': 0.65,
    
    # Minimum duration in zone before transition (frames)
    'min_duration_watch': 5,
    'min_duration_warning': 3,
    'min_duration_critical': 2,
    
    # Colors for visualization
    'zone_colors': {
        'NORMAL': '#22c55e',        # Green
        'WATCH': '#eab308',         # Yellow
        'WARNING': '#f97316',       # Orange
        'CRITICAL': '#ef4444',      # Red
    }
}

# =============================================================================
# BEHAVIORAL FEATURES CONFIGURATION
# =============================================================================
"""
24 BEHAVIORAL FEATURES
----------------------
Motion-based features extracted from temporal windows.
We learn HOW things move, not WHAT they look like.

Categories:
1. Motion Energy (frames 1-4): Overall activity levels
2. Flow Statistics (frames 5-10): Optical flow metrics
3. Directional (frames 11-14): Movement direction patterns
4. Variance (frames 15-18): Consistency of motion
5. Temporal (frames 19-24): Time-based patterns
"""

BEHAVIORAL_FEATURES_CONFIG = {
    'temporal_window': 5,           # Frames per temporal window
    'normalize': True,              # Normalize features to [0, 1]
    
    # Optical flow parameters
    'optical_flow': {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
    },
    
    # Grid-based features
    'grid_size': (4, 4),            # Spatial grid for localized features
}

# =============================================================================
# CONTEXT AUGMENTATION CONFIGURATION
# =============================================================================
"""
SCENE CONTEXT
-------------
Normality is CONTEXT-DEPENDENT.

- Dawn rush hour ≠ midnight behavior
- Checkpoint queue ≠ open field
- Clear day ≠ foggy visibility

Context factors adjust baseline expectations.
"""

CONTEXT_CONFIG = {
    # Time of day periods
    'time_periods': {
        'dawn': (5, 8),             # 5:00 - 8:00
        'day': (8, 17),             # 8:00 - 17:00
        'dusk': (17, 20),           # 17:00 - 20:00
        'night': (20, 5),           # 20:00 - 5:00
    },
    
    # Camera zones (pre-configured per deployment)
    'camera_zones': ['fence_line', 'road', 'open_field', 'checkpoint'],
    
    # Patrol schedule integration
    'patrol_schedule_enabled': False,
}

# =============================================================================
# HUMAN-IN-THE-LOOP CONFIGURATION
# =============================================================================
"""
OPERATOR FEEDBACK
-----------------
"AI assists operators, it does NOT replace them."

Feedback types:
- CONFIRM: Drift is concerning
- BENIGN: False alarm / expected behavior
- INVESTIGATE: Flag for review

Baseline adaptation:
- Gradual, controlled updates
- Requires threshold of consistent feedback
- Always reversible
"""

FEEDBACK_CONFIG = {
    # Minimum feedback before baseline adaptation
    'min_feedback_for_update': 10,
    
    # Learning rate for baseline shifts
    'update_learning_rate': 0.05,
    
    # Consistency threshold (% agreement needed)
    'consistency_threshold': 0.8,
    
    # Maximum shift per update (standard deviations)
    'max_shift_per_update': 0.5,
    
    # Storage
    'storage_path': str(FEEDBACK_DIR),
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
"""
SUPPORTED DATASETS
------------------
- UCSD Pedestrian: Standard surveillance benchmark
- Avenue: Campus surveillance with various anomalies
- UMN: Crowd panic and escape scenarios

For production: Replace with actual surveillance feeds.
"""

DATASET_CONFIG = {
    'ucsd': {
        'name': 'UCSD_Pedestrian',
        'url': 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz',
        'train_folder': 'Train',
        'test_folder': 'Test',
    },
    'avenue': {
        'name': 'Avenue',
        'train_folder': 'training_videos',
        'test_folder': 'testing_videos',
    },
    'umn': {
        'name': 'UMN',
        'scenes': ['lawn', 'indoor', 'plaza'],
    },
}

# =============================================================================
# EXPLAINABILITY CONFIGURATION
# =============================================================================
"""
XAI CONFIGURATION
-----------------
Every alert must be explainable.
Operators need to understand WHY the system flagged something.
"""

EXPLAINABILITY_CONFIG = {
    # Top features to report
    'top_k_features': 5,
    
    # Explanation detail level
    'detail_level': 'standard',     # 'minimal', 'standard', 'detailed'
    
    # Generate natural language explanations
    'natural_language': True,
}
