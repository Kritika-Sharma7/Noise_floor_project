# NOISE FLOOR

## Defense-Grade Early Warning Intelligence System

> **Turning background noise into preventive defense insight.**
>
> *This system is designed for border surveillance and high-security perimeters where threats emerge gradually.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![TRL-4](https://img.shields.io/badge/TRL--4%20Lab%20Validated-green.svg)](#technology-readiness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## System Philosophy

> **"Defense systems manage CONFIDENCE, not panic."**
>
> **"AI assists operators, it does NOT replace them."**
>
> **"Baseline adaptation is human-gated."**

---

## Quick Start

### Prerequisites
- **Python 3.10 or higher** (tested up to Python 3.14)
- **Git** for cloning the repository
- **OpenCV** for real video processing (installed automatically via requirements.txt)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/Kritika-Sharma7/Noise_floor_project.git
cd Noise_floor_project
```

#### 2. Create Virtual Environment
```bash
python -m venv .venv
```

#### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 5. (Optional) Set Up OpenAI API for AI Insights
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

---

## Running the Application

### Dashboard (Recommended)
```bash
# Make sure virtual environment is activated, then:
streamlit run dashboard/app.py
```
The dashboard will open at `http://localhost:8501`

### Intelligence-Enhanced Dashboard
```bash
streamlit run dashboard/app_intelligence.py
```

### CLI Demo
```bash
python main.py --demo
```

---

## Project Structure

```
Noise_floor_project/
├── dashboard/
│   ├── app.py                    # Main Streamlit dashboard
│   ├── app_intelligence.py       # Intelligence-enhanced dashboard
│   └── app_backup.py             # Backup dashboard version
├── src/
│   ├── __init__.py
│   ├── lstm_vae.py               # LSTM-VAE temporal normality model
│   ├── drift_intelligence.py     # Drift scoring & TDI computation
│   ├── risk_zones.py             # Confidence-based zone classifier
│   ├── explainability.py         # XAI attribution & explanations
│   ├── feedback_system.py        # Human-in-the-loop feedback
│   ├── video_features.py         # Real video optical flow extraction
│   ├── behavioral_features.py    # 24 behavioral feature definitions
│   ├── baseline_freeze.py        # Baseline management
│   ├── context_augmentation.py   # Environmental context integration
│   ├── data_ingestion.py         # Multi-source data loading
│   ├── feature_extraction.py     # Synthetic feature extraction
│   ├── autoencoder.py            # Pure NumPy normality autoencoder
│   ├── drift_detection.py        # EWMA + sliding window detection
│   ├── watch_zones.py            # Graduated alert classification
│   ├── baseline_comparison.py    # Compare vs traditional methods
│   └── utils.py                  # Utility functions
├── scripts/
│   └── generate_sample_video.py  # Generate synthetic test videos
├── data/
│   ├── baseline_data/            # Frozen baseline snapshots
│   └── UCSD_Anomaly_Dataset.v1p2/  # (Download separately - see below)
├── feedback_data/                # Operator feedback storage
├── models/                       # Saved model weights
├── output/                       # Generated outputs
├── config.py                     # All configuration settings
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Configuration

### Data Mode
Edit `config.py` to switch between synthetic and real video:

```python
DATA_MODE = "real_video"  # Options: "synthetic" | "real_video"
```

### Detection Sensitivity
```python
# In config.py
BASELINE_FRAMES = 50        # Frames to establish baseline
EWMA_ALPHA = 0.3            # Smoothing factor (0-1)
TREND_WINDOW = 20           # Frames for trend analysis

# Zone thresholds
NORMAL_THRESHOLD = 1.5
WATCH_THRESHOLD = 2.0
WARNING_THRESHOLD = 2.5
```

---

## Using Real Video Data (UCSD Dataset)

### Download the Dataset
1. Download from [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)
   - Or from [Kaggle](https://www.kaggle.com/datasets/antoinelamb/ucsd-anomaly-detection-dataset)
2. Extract to `data/UCSD_Anomaly_Dataset.v1p2/`

### Directory Structure After Download
```
data/
└── UCSD_Anomaly_Dataset.v1p2/
    ├── UCSDped1/
    │   ├── Train/
    │   │   ├── Train001/
    │   │   ├── Train002/
    │   │   └── ...
    │   └── Test/
    └── UCSDped2/
        ├── Train/
        └── Test/
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NOISE FLOOR PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Video Input  │───>│   Feature    │───>│  LSTM-VAE Normality  │  │
│  │  (CCTV/IoT)  │    │  Extraction  │    │      Learning        │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                             │                        │              │
│                             v                        v              │
│                    ┌────────────────────────────────────┐          │
│                    │     Behavioral Feature Vector      │          │
│                    │  - Optical Flow Magnitude          │          │
│                    │  - Motion Variance                 │          │
│                    │  - Directional Entropy             │          │
│                    │  - Temporal Gradients              │          │
│                    └────────────────────────────────────┘          │
│                                      │                              │
│                                      v                              │
│                    ┌────────────────────────────────────┐          │
│                    │     Drift Intelligence Engine      │          │
│                    │  - KL Divergence from baseline     │          │
│                    │  - EWMA-smoothed deviation scores  │          │
│                    │  - Trend persistence tracking      │          │
│                    │  - Threat Deviation Index (TDI)    │          │
│                    └────────────────────────────────────┘          │
│                                      │                              │
│                                      v                              │
│                    ┌────────────────────────────────────┐          │
│                    │      Risk Zone Classifier          │          │
│                    │  NORMAL | WATCH | WARNING | CRITICAL          │
│                    └────────────────────────────────────┘          │
│                                      │                              │
│                                      v                              │
│                    ┌────────────────────────────────────┐          │
│                    │   Explainability & Dashboard       │          │
│                    │   Human-in-the-Loop Feedback       │          │
│                    └────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Problem

Current monitoring systems are **reactive and threshold-based**. They wait for something to be obviously wrong before alerting.

**Traditional Anomaly Detection:**
- Reacts only to sudden spikes
- High false positive rate
- Misses gradual changes
- Binary alerts (normal/anomaly)

**Real-world threats often manifest as GRADUAL changes:**
- Insider threats: Slow behavioral shift over weeks
- Equipment degradation: Progressive performance decline
- Crowd dynamics: Tension building before incidents
- Infrastructure decay: Slow operational drift

---

## Our Solution: NOISE FLOOR

NOISE FLOOR learns what "normal" looks like and detects when behavior **slowly drifts away** from that baseline.

**Key Innovation: Drift Detection, Not Anomaly Detection**

```
Traditional: "Is this instant abnormal?"       -> Reactive, noisy
NOISE FLOOR: "Is behavior gradually changing?" -> Proactive, stable
```

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Unsupervised Learning** | Train only on normal behavior, no labeled anomalies needed |
| **LSTM-VAE Architecture** | Temporal normality modeling with uncertainty quantification |
| **Drift Intelligence** | EWMA smoothing, trend analysis, KL divergence tracking |
| **Confidence-Based Zones** | Graduated risk levels (Normal -> Watch -> Warning -> Critical) |
| **Explainable AI (XAI)** | Feature attribution explaining WHY drift is detected |
| **Human-in-the-Loop** | Operator feedback for baseline adaptation |
| **Real-Time Dashboard** | Live monitoring with drift visualization |

---

## Troubleshooting

### Common Issues

**1. Streamlit not found**
```bash
pip install streamlit
```

**2. OpenCV errors**
```bash
pip install opencv-python
# Or for full version:
pip install opencv-contrib-python
```

**3. Module not found errors**
```bash
# Make sure you're in the project root directory
cd Noise_floor_project

# And virtual environment is activated
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux/Mac:
source .venv/bin/activate
```

**4. Port 8501 already in use**
```bash
streamlit run dashboard/app.py --server.port 8502
```

**5. Permission denied on Windows PowerShell**
```powershell
# Run this once to allow script execution:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Technical Details

### Why LSTM-VAE?
- **LSTM**: Captures temporal dependencies in behavioral sequences
- **VAE**: Provides probabilistic latent space for uncertainty quantification
- **Unsupervised**: Learns only from NORMAL data

### Why Pure NumPy (No TensorFlow)?
- Maximum Python version compatibility (3.10-3.14)
- Lightweight deployment (~50MB vs ~500MB)
- CPU-only inference for edge devices
- Simpler dependency management

### 24 Behavioral Features
The system extracts 24 features from video including:
- Motion energy and variance
- Optical flow statistics
- Direction consistency and entropy
- Scene complexity metrics
- Velocity and acceleration patterns
- Activity state transitions

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Team

- Kritika Sharma
- Contributors welcome!

---

<p align="center">
  <b>NOISE FLOOR</b> - Behavioral Drift Intelligence - Gray-box Explainable AI
</p>
