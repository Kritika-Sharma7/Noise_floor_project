# 🛡️ NOISE FLOOR

## Defense-Grade Early Warning Intelligence System

> **"Turning background noise into preventive defense insight."**
>
> *Designed for border surveillance and high-security perimeters where threats emerge gradually.*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/TRL--4-Lab%20Validated-green.svg" alt="TRL">
  <img src="https://img.shields.io/badge/UCSD-Dataset-orange.svg" alt="Dataset">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## 🎯 The Problem

**Traditional surveillance systems are REACTIVE:**
- They alert you **AFTER** something bad happens
- Binary alerts: "NORMAL" or "ALARM" 
- High false positive rate causes **alert fatigue**
- **Misses gradual threats** that evolve slowly

**Real-world threats often emerge GRADUALLY:**
- Border infiltrations happen in stages
- Insider threats develop over weeks
- Equipment fails progressively
- Crowd tension builds before incidents

---

## 💡 Our Solution

**NOISE FLOOR detects threats BEFORE they become obvious.**

Instead of asking *"Is this instant abnormal?"* (reactive), we ask:
> *"Is behavior **gradually changing** from what's normal?"* (proactive)

### Key Innovation: Drift Detection, Not Anomaly Detection

```
Traditional: "Is this frame abnormal?"     → Reactive, noisy, late
NOISE FLOOR: "Is behavior slowly drifting?" → Proactive, stable, early warning
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NOISE FLOOR PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 📹 VIDEO     │───→│ 🔬 FEATURE   │───→│ 🧠 LSTM-VAE          │  │
│  │  INGESTION   │    │  EXTRACTION  │    │  TEMPORAL LEARNING   │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                       │              │
│         │           24 Behavioral Features           │              │
│         │           • Motion Energy                  │              │
│         │           • Optical Flow                   │              │
│         │           • Scene Entropy                  │              │
│         │           • Direction Patterns             │              │
│         │                                            │              │
│         │                    ↓                       ↓              │
│         │         ┌────────────────────────────────────┐            │
│         │         │     📊 DRIFT INTELLIGENCE          │            │
│         │         │  • Threat Deviation Index (TDI)    │            │
│         │         │  • KL Divergence Analysis          │            │
│         │         │  • EWMA Smoothed Scoring           │            │
│         │         │  • Trend Detection (↑ → ↓)         │            │
│         │         └────────────────────────────────────┘            │
│         │                         │                                 │
│         │                         ↓                                 │
│         │         ┌────────────────────────────────────┐            │
│         │         │     🎯 RISK ZONE CLASSIFIER        │            │
│         │         │  🟢 NORMAL  → Standard monitoring  │            │
│         │         │  🟡 WATCH   → Increase attention   │            │
│         │         │  🟠 WARNING → Alert response team  │            │
│         │         │  🔴 CRITICAL→ Immediate action     │            │
│         │         └────────────────────────────────────┘            │
│         │                         │                                 │
│         │                         ↓                                 │
│         │         ┌────────────────────────────────────┐            │
│         │         │     🖥️ INTELLIGENCE DASHBOARD      │            │
│         │         │  • Real-time TDI visualization     │            │
│         │         │  • Feature attribution (XAI)       │            │
│         │         │  • Human-in-the-loop feedback      │            │
│         │         └────────────────────────────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎓 Unsupervised Learning** | Train only on normal behavior - no labeled anomalies needed |
| **🧠 LSTM-VAE Architecture** | Pure NumPy implementation - temporal normality modeling |
| **� Ensemble Detection** | LSTM-VAE + Isolation Forest + One-Class SVM + LOF |
| **📊 Threat Deviation Index** | 0-100 scale for intuitive operator understanding |
| **🎯 4-Tier Risk Zones** | Graduated alerts reduce fatigue (Normal → Watch → Warning → Critical) |
| **🏷️ Anomaly Classification** | Categorize threats (Loitering, Intrusion, Crowd, etc.) |
| **🌌 3D Latent Visualization** | Visualize behavioral trajectories in latent space |
| **🔍 Explainable AI (XAI)** | Shows WHICH features are causing drift |
| **🚨 Incident Logging** | Track, export, and analyze all alerts |
| **🔮 TDI Forecasting** | Predict future threat levels |
| **👤 Human-in-the-Loop** | Operators provide feedback for baseline adaptation |
| **📹 Real Video Support** | Works with UCSD dataset and custom surveillance footage |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/Noise_floor_project.git
cd Noise_floor_project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch the main dashboard (RECOMMENDED)
streamlit run dashboard/app_main.py
```

The dashboard opens at `http://localhost:8501`

---

## 🖥️ Dashboard Guide

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **📊 Intelligence Dashboard** | Main TDI display, risk zones, feature attribution, AI explanations |
| **🧠 AI Ensemble** | Multi-model detection votes, 3D latent space, anomaly classification |
| **📹 Camera Grid** | 6-camera surveillance view with live UCSD frames |
| **🚨 Incident Log** | Full history of all alerts with export capability |
| **📈 Analytics** | Session statistics, zone distribution, data export |

### Operating Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **📹 UCSD Real Video** | Processes actual surveillance footage from UCSD dataset | Production demo, validation |
| **🔬 Synthetic Demo** | Uses generated data with controlled drift | Quick testing, concept demo |

### Understanding the Display

#### Threat Deviation Index (TDI)
- **0-25**: 🟢 NORMAL - All good
- **25-50**: 🟡 WATCH - Something's slightly off
- **50-75**: 🟠 WARNING - Confirmed drift, pay attention
- **75-100**: 🔴 CRITICAL - Take action immediately

#### Drift Trend
- **↑ RISING** - Threat is increasing
- **→ STABLE** - No significant change
- **↓ FALLING** - Returning to normal

#### Anomaly Categories
- 🧍 **Loitering** - Prolonged stationary activity
- ⚠️ **Intrusion** - Boundary crossing detected
- 👥 **Crowd Formation** - Unusual gathering
- 🌀 **Erratic Movement** - Abnormal motion patterns
- 🎯 **Coordinated Activity** - Synchronized movement
- ⚡ **Speed Anomaly** - Unusual velocity
- ↩️ **Direction Anomaly** - Unusual direction pattern

---

## 📁 Project Structure

```
Noise_floor_project/
├── 📂 dashboard/
│   └── app_main.py            # Main dashboard (USE THIS)
│
├── 📂 src/
│   ├── lstm_vae.py            # LSTM-VAE temporal model
│   ├── drift_intelligence.py  # TDI computation engine
│   ├── ensemble_detector.py   # Multi-model ensemble
│   ├── advanced_ai.py         # Anomaly classification
│   ├── risk_zones.py          # 4-tier zone classifier
│   ├── behavioral_features.py # 24 feature definitions
│   ├── video_features.py      # Real video processing
│   ├── explainability.py      # XAI attribution
│   ├── incident_logger.py     # Incident tracking
│   ├── feedback_system.py     # Human-in-the-loop
│   └── utils.py               # Utility functions
│
├── 📂 data/
│   └── UCSD_Anomaly_Dataset.v1p2/
│       ├── UCSDped1/
│       │   ├── Train/         # Normal pedestrian videos
│       │   └── Test/          # Contains anomalies
│       └── UCSDped2/
│
├── 📂 incident_logs/          # Logged incidents
├── 📂 feedback_data/          # Operator feedback logs
├── 📂 baseline_data/          # Frozen baseline snapshots
│
├── config.py                  # Configuration settings
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🔬 How It Works

### Phase 1: Learning Normal Behavior
```
UCSD Train Data (Normal pedestrians)
    → Optical Flow Extraction
    → 24 Behavioral Features
    → LSTM-VAE Training
    → Ensemble Detector Fitting
    → Baseline Established ✓
```

### Phase 2: Monitoring & Detection
```
UCSD Test Data (Contains bikes, carts, etc.)
    → Feature Extraction
    → LSTM-VAE Inference
    → Ensemble Voting (IF, SVM, LOF)
    → Compute TDI
    → Classify Risk Zone
    → Classify Anomaly Type
    → Log Incident
    → Generate Explanation
```

### Phase 3: Operator Response
```
Dashboard displays:
    → Current TDI (e.g., 67)
    → Risk Zone (🟠 WARNING)
    → Trend (↑ RISING)
    → Top Features causing drift
    → AI explanation
    
Operator can:
    → Acknowledge alert
    → Mark as false positive
    → Request investigation
    → Update baseline (human-gated)
```

---

## 🎓 System Philosophy

> **"Defense systems manage CONFIDENCE, not panic."**
> 
> **"AI assists operators, it does NOT replace them."**
> 
> **"Baseline adaptation is human-gated."**

These three principles guide every design decision:

1. **Graduated Risk Zones** - Reduce alert fatigue with progressive warnings
2. **Explainable AI** - Operators understand WHY alerts occur
3. **Human-in-the-Loop** - Critical decisions remain with humans
4. **Baseline Protection** - Prevents adversarial manipulation

---

## 📊 Dataset Information

### UCSD Anomaly Detection Dataset

Used as proxy for border surveillance footage.

| Subset | Train | Test | Anomalies |
|--------|-------|------|-----------|
| **Ped1** | 34 clips | 36 clips | Bikes, skateboards, carts |
| **Ped2** | 16 clips | 12 clips | Bikes, skateboards |

**How we use it:**
- **Train folder** → Learn NORMAL pedestrian behavior
- **Test folder** → Detect DRIFT when anomalies appear (labels ignored - unsupervised)

---

## 🛠️ Technical Specifications

| Component | Specification |
|-----------|---------------|
| **ML Model** | LSTM-VAE (Pure NumPy, no TensorFlow/PyTorch) |
| **Features** | 24 behavioral metrics from optical flow |
| **Latent Dim** | 8-dimensional latent space |
| **Sequence Length** | 10 frames temporal window |
| **Smoothing** | EWMA with α=0.15 |
| **Technology Readiness** | TRL-4 (Lab Validated) |

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Data source
DATA_MODE = "real_video"  # "synthetic" | "real_video"

# UCSD Dataset
UCSD_SUBSET = "ped1"      # "ped1" or "ped2"

# Baseline protection
BASELINE_FREEZE_CONFIG = {
    'learning_window': 200,
    'freeze_after_learning': True,
    'adaptation_learning_rate': 0.01,
}
```

---

## 🤝 Use Cases

| Domain | Application |
|--------|-------------|
| **Border Security** | Detect infiltration patterns at perimeter fences |
| **Airport Security** | Monitor crowd behavior at checkpoints |
| **Critical Infrastructure** | Surveillance of power plants, data centers |
| **Military Installations** | Base perimeter monitoring |
| **Corporate Security** | Campus and facility protection |

---

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Detection Delay** | Frames between actual drift start and system detection |
| **False Positive Rate** | Alerts during confirmed normal periods |
| **Peak TDI** | Maximum threat deviation observed |
| **Zone Transitions** | History of risk zone changes |

---

## 🐛 Troubleshooting

### Common Issues

**1. "Dataset not found"**
```bash
# Ensure UCSD dataset is in correct location:
data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/
```

**2. "Module not found"**
```bash
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**3. "Streamlit not starting"**
```bash
pip install streamlit --upgrade
streamlit run dashboard/app_pro_v2.py
```

---

## 👥 Team

- **Project**: NOISE FLOOR - Defense Intelligence System
- **Event**: SnowHack Hackathon

---

## 📄 License

MIT License - See LICENSE file for details.

---

<p align="center">
  <b>🛡️ NOISE FLOOR - Because early warning saves lives. 🛡️</b>
</p>
