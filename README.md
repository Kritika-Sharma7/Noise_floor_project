# NOISE FLOOR �️

## Defense-Grade Early Warning Intelligence System

> **Turning background noise into preventive defense insight.**
>
> *This system is designed for border surveillance and high-security perimeters where threats emerge gradually.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![TRL-4](https://img.shields.io/badge/TRL--4%20Lab%20Validated-green.svg)](#technology-readiness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎖️ System Philosophy

> **"Defense systems manage CONFIDENCE, not panic."**
>
> **"AI assists operators, it does NOT replace them."**
>
> **"Baseline adaptation is human-gated."**

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10 or higher
- OpenCV (for real video processing)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher (tested up to Python 3.14)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Kritika-Sharma7/Noise_floor_project.git
cd Noise_floor_project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# Make sure virtual environment is activated, then:
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

### Running CLI Demo

```bash
python main.py --demo
```

---

## 🎯 The Problem

Current monitoring systems are **reactive and threshold-based**. They wait for something to be obviously wrong before alerting.

**Traditional Anomaly Detection:**
- ❌ Reacts only to sudden spikes
- ❌ High false positive rate
- ❌ Misses gradual changes
- ❌ Binary alerts (normal/anomaly)

**Real-world threats often manifest as GRADUAL changes:**
- 🕵️ Insider threats: Slow behavioral shift over weeks
- ⚙️ Equipment degradation: Progressive performance decline
- 👥 Crowd dynamics: Tension building before incidents
- 🏭 Infrastructure decay: Slow operational drift

---

## 💡 Our Solution: NOISE FLOOR

NOISE FLOOR learns what "normal" looks like and detects when behavior **slowly drifts away** from that baseline.

**Key Innovation: Drift Detection, Not Anomaly Detection**

```
Traditional: "Is this instant abnormal?"      → Reactive, noisy
NOISE FLOOR: "Is behavior gradually changing?" → Proactive, stable
```

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Unsupervised Learning** | Train only on normal behavior, no labeled anomalies needed |
| **Temporal Aggregation** | EWMA smoothing, sliding windows, trend analysis |
| **Gray-Box Explainable** | Every feature and score has clear interpretation |
| **Watch Zones** | Graduated alert levels (Normal → Watch → Warning → Alert) |
| **Real-Time Dashboard** | Live monitoring with drift visualization |

---

## 🏗️ Project Structure

```
Noise_floor_project/
├── dashboard/
│   └── app.py              # Streamlit dashboard (main UI)
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py   # Behavioral feature extraction
│   ├── autoencoder.py          # Pure NumPy normality autoencoder
│   ├── drift_detection.py      # EWMA + sliding window drift detection
│   ├── watch_zones.py          # Graduated alert classification
│   ├── baseline_comparison.py  # Compare vs traditional methods
│   └── utils.py                # Utility functions
├── config.py               # Configuration settings
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NOISE FLOOR PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Video Input  │───▶│   Feature    │───▶│  Normality Learning  │  │
│  │  (CCTV/IoT)  │    │  Extraction  │    │   (Autoencoder)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                             │                        │              │
│                             ▼                        ▼              │
│                    ┌────────────────────────────────────┐          │
│                    │     Behavioral Feature Vector      │          │
│                    │  • Optical Flow Magnitude          │          │
│                    │  • Motion Variance                 │          │
│                    │  • Directional Entropy             │          │
│                    │  • Temporal Gradients              │          │
│                    └────────────────────────────────────┘          │
│                                      │                              │
│                                      ▼                              │
│                    ┌────────────────────────────────────┐          │
│                    │       Drift Detection Engine       │          │
│                    │  • EWMA Smoothing                  │          │
│                    │  • Sliding Window Aggregation      │          │
│                    │  • Trend Slope Analysis            │          │
│                    └────────────────────────────────────┘          │
│                                      │                              │
│                                      ▼                              │
│                    ┌────────────────────────────────────┐          │
│                    │        Watch Zone Classifier       │          │
│                    │  🟢 Normal  🟡 Watch  🟠 Warning  🔴 Alert   │  │
│                    └────────────────────────────────────┘          │
│                                      │                              │
│                                      ▼                              │
│                    ┌────────────────────────────────────┐          │
│                    │      Real-Time Dashboard           │          │
│                    └────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dashboard Features

The Streamlit dashboard provides:

1. **System Status Bar** - Real-time monitoring health
2. **Deployment Characteristics** - CPU-only, edge-deployable, CCTV-compatible
3. **Drift Score Timeline** - Visual drift progression with zone bands
4. **Detection Metrics** - Detection delay, false positive rate
5. **Operational Scenarios** - Border surveillance, infrastructure, crowd intel
6. **Method Comparison** - NOISE FLOOR vs Isolation Forest, SVM, Threshold
7. **System Architecture** - Explainable pipeline visualization

---

## 🔧 Configuration

Edit `config.py` to adjust:

```python
# Detection sensitivity
BASELINE_FRAMES = 50        # Frames to establish baseline
EWMA_ALPHA = 0.3            # Smoothing factor (0-1)
TREND_WINDOW = 20           # Frames for trend analysis

# Zone thresholds
NORMAL_THRESHOLD = 1.5
WATCH_THRESHOLD = 2.0
WARNING_THRESHOLD = 2.5
```

---

## 🧪 Running Tests

```bash
# Run the demo simulation
python main.py --demo

# Expected output:
# Detection Frame: ~115 (drift at 100)
# Detection Delay: ~15 frames
# False Positive Rate: 0.0%
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📋 Troubleshooting

### Common Issues

**1. Streamlit not found**
```bash
pip install streamlit
```

**2. Module not found errors**
```bash
# Make sure you're in the project root directory
cd Noise_floor_project
# And virtual environment is activated
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

**3. Port 8501 already in use**
```bash
streamlit run dashboard/app.py --server.port 8502
```

---

## 📚 Technical Details

### Why Unsupervised Learning?
- Threat events are rare and poorly labeled
- Future attack patterns are unknown
- System learns only from normal behavior
- **No prior examples of attacks required**

### Why Pure NumPy (No TensorFlow)?
- Maximum Python version compatibility (3.10-3.14)
- Lightweight deployment (~50MB vs ~500MB)
- CPU-only inference for edge devices
- Simpler dependency management

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Team

- Kritika Sharma
- Contributors welcome!

---

<p align="center">
  <b>NOISE FLOOR</b> • Behavioral Drift Intelligence • Gray-box Explainable AI
</p>
