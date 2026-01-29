"""
NOISE FLOOR - Defense-Grade Intelligence System
================================================
Border surveillance and high-security perimeter monitoring.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

TECHNOLOGY READINESS LEVEL: TRL-4
Lab-validated prototype for decision-support intelligence.
This is NOT an autonomous system - AI assists operators, it does NOT replace them.

SYSTEM PHILOSOPHY:
- "Defense systems manage CONFIDENCE, not panic."
- "AI assists operators, it does NOT replace them."
- "Baseline adaptation is human-gated."

ARCHITECTURE:
    Video/Sensor Ingestion → Context Augmentation → Behavioral Features
    → LSTM-VAE → Drift Intelligence → Risk Zones → Explainability
    → Human-in-the-Loop Feedback

OUTPUTS:
    • Threat Deviation Index (TDI): 0-100 scale
    • Risk Zone: Normal / Watch / Warning / Critical
    • Drift Trend: ↑ Rising / → Stable / ↓ Falling
    • Top Contributing Features
    • Natural Language Explanation

DATA MODES:
    • synthetic   - Controlled testing with generated data
    • real_video  - Real surveillance footage (UCSD dataset or custom)

Usage:
    python main.py --demo              # Synthetic data demo
    python main.py --real-demo         # Real video demo (UCSD dataset)
    python main.py --video PATH        # Process custom video
    python main.py --dashboard         # Launch defense dashboard
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import DATA_MODE, UCSD_DATASET_PATH, UCSD_SUBSET


def run_intelligence_demo(data_mode: str = "synthetic"):
    """
    Run the complete defense-grade intelligence pipeline.
    Demonstrates LSTM-VAE, drift intelligence, risk zones, and explainability.
    
    Args:
        data_mode: "synthetic" or "real_video"
    """
    print("=" * 70)
    print("NOISE FLOOR - Defense Intelligence System")
    print("Behavioral Drift Detection for Border Surveillance")
    print("=" * 70)
    print(f"\nSession: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Mode: {data_mode.upper()}")
    
    if data_mode == "real_video":
        print("\nNOTE: Public surveillance dataset used as proxy for border CCTV footage.")
    
    # Import intelligence modules
    from src.behavioral_features import (
        BEHAVIORAL_FEATURES,
        create_synthetic_normal_data, create_synthetic_drift_data
    )
    from src.lstm_vae import TemporalNormalityLSTMVAE
    from src.drift_intelligence import DriftIntelligenceEngine, DriftTrend
    from src.risk_zones import RiskZoneClassifier, RiskZone
    from src.explainability import DriftAttributor
    
    n_features = len(BEHAVIORAL_FEATURES)
    
    # =========================================================================
    # STEP 1: GENERATE TRAINING DATA (Normal behavior only)
    # =========================================================================
    print("\n" + "─" * 70)
    print("📊 PHASE 1: Baseline Learning")
    print("─" * 70)
    
    print("\n[1.1] Generating normal behavioral data...")
    train_data = create_synthetic_normal_data(num_samples=500, feature_dim=n_features)
    print(f"      Training samples: {len(train_data)}")
    print(f"      Features per sample: {n_features}")
    
    # Compute baseline statistics
    baseline_means = np.mean(train_data, axis=0)
    baseline_stds = np.std(train_data, axis=0) + 1e-6
    print(f"      Baseline mean range: [{baseline_means.min():.3f}, {baseline_means.max():.3f}]")
    
    # =========================================================================
    # STEP 2: TRAIN LSTM-VAE
    # =========================================================================
    print("\n[1.2] Training LSTM-VAE for temporal normality...")
    
    lstm_vae = TemporalNormalityLSTMVAE(
        input_dim=n_features,
        hidden_dim=32,
        latent_dim=8,
        seq_len=10,
    )
    
    # Create sequences
    sequences = []
    for i in range(len(train_data) - 10):
        seq = train_data[i:i+10]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    print(f"      Training sequences: {len(sequences)}")
    
    lstm_vae.train(sequences, epochs=50, verbose=False)
    print("      ✓ LSTM-VAE training complete")
    
    # =========================================================================
    # STEP 3: INITIALIZE INTELLIGENCE COMPONENTS
    # =========================================================================
    print("\n[1.3] Initializing intelligence components...")
    
    drift_engine = DriftIntelligenceEngine(
        baseline_frames=50,       # Use first 50 frames for baseline
        ewma_alpha=0.1,
        feature_names=BEHAVIORAL_FEATURES,
    )
    print("      ✓ Drift Intelligence Engine initialized")
    
    zone_classifier = RiskZoneClassifier()
    print("      ✓ Risk Zone Classifier initialized")
    
    attributor = DriftAttributor(
        feature_names=BEHAVIORAL_FEATURES,
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
    )
    print("      ✓ Drift Attributor initialized")
    
    # =========================================================================
    # STEP 4: GENERATE TEST DATA WITH DRIFT
    # =========================================================================
    print("\n" + "─" * 70)
    print("🎯 PHASE 2: Drift Simulation")
    print("─" * 70)
    
    drift_start = 100
    drift_magnitude = 1.5
    
    print(f"\n[2.1] Generating test data with injected drift...")
    print(f"      Normal period: frames 0-{drift_start-1}")
    print(f"      Drift period: frames {drift_start}-299")
    print(f"      Drift magnitude: {drift_magnitude}σ")
    
    normal_test = create_synthetic_normal_data(num_samples=drift_start, feature_dim=n_features)
    drift_test = create_synthetic_drift_data(num_samples=200, feature_dim=n_features, drift_rate=drift_magnitude * 0.02)
    test_data = np.vstack([normal_test, drift_test])
    
    print(f"      Total test frames: {len(test_data)}")
    
    # =========================================================================
    # STEP 5: PROCESS TEST DATA
    # =========================================================================
    print("\n" + "─" * 70)
    print("📈 PHASE 3: Intelligence Analysis")
    print("─" * 70)
    
    print("\n[3.1] Processing frames through intelligence pipeline...")
    
    results = []
    feature_buffer = []
    drift_engine.reset()
    zone_classifier.reset()
    
    zone_transitions = []
    current_zone = None
    drift_onset_detected = None
    
    for i, features in enumerate(test_data):
        # Add to feature buffer
        feature_buffer.append(features)
        if len(feature_buffer) > 10:
            feature_buffer.pop(0)
        
        # Compute metrics from LSTM-VAE
        if len(feature_buffer) >= 10:
            sequence = np.array(feature_buffer[-10:])
            seq_input = sequence.reshape(1, 10, -1)
            
            # Get detailed output from LSTM-VAE
            output = lstm_vae.forward(seq_input, training=False)
            reconstruction_loss = output.reconstruction_loss
            kl_divergence = output.kl_divergence
            latent_mean = output.latent_mean[0]
            latent_logvar = output.latent_log_var[0]
        else:
            # Simple fallback for early frames
            reconstruction_loss = np.mean((features - baseline_means) ** 2)
            kl_divergence = 0.0
            latent_mean = np.zeros(8)
            latent_logvar = np.zeros(8)
        
        # Process through drift intelligence engine
        intelligence = drift_engine.process(
            reconstruction_loss=reconstruction_loss,
            kl_divergence=kl_divergence,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            features=features,
            frame_index=i,
        )
        
        # Classify risk zone
        zone_state = zone_classifier.classify(
            threat_deviation_index=intelligence.threat_deviation_index,
            z_score=intelligence.z_score,
            trend_slope=intelligence.trend_slope,
            trend_persistence=intelligence.trend_persistence,
        )
        
        results.append({
            'frame': i,
            'tdi': intelligence.threat_deviation_index,
            'zone': zone_state.zone,
            'trend': intelligence.drift_trend,
            'confidence': intelligence.confidence,
        })
        
        # Track zone transitions
        if zone_state.zone != current_zone:
            zone_transitions.append((i, zone_state.zone))
            current_zone = zone_state.zone
            
            # First time leaving NORMAL after drift starts
            if drift_onset_detected is None and i >= drift_start and zone_state.zone != RiskZone.NORMAL:
                drift_onset_detected = i
    
    print(f"      Processed {len(test_data)} frames")
    
    # =========================================================================
    # STEP 6: RESULTS
    # =========================================================================
    print("\n" + "─" * 70)
    print("📊 PHASE 4: Intelligence Report")
    print("─" * 70)
    
    # Zone transitions
    print("\n[4.1] Zone Transitions:")
    for frame, zone in zone_transitions:
        marker = " ← DRIFT INJECTED" if frame == drift_start else ""
        if frame == drift_onset_detected:
            marker = " ← DRIFT DETECTED ⚡"
        
        zone_icon = {
            RiskZone.NORMAL: "🟢",
            RiskZone.WATCH: "🟡",
            RiskZone.WARNING: "🟠",
            RiskZone.CRITICAL: "🔴",
        }.get(zone, "⚪")
        
        print(f"      Frame {frame:4d}: {zone_icon} {zone.name}{marker}")
    
    # Detection performance
    print("\n[4.2] Detection Performance:")
    
    detection_delay = None
    if drift_onset_detected is not None:
        detection_delay = drift_onset_detected - drift_start
        print(f"      ✓ Drift detected at frame {drift_onset_detected}")
        print(f"      Detection delay: {detection_delay} frames")
        
        if detection_delay < 20:
            print("      ⚡ EXCELLENT: Early warning achieved")
        elif detection_delay < 50:
            print("      ✓ GOOD: Reasonable detection time")
        else:
            print("      ⚠ SLOW: Detection delayed")
    else:
        print("      ⚠ Drift not detected in test period")
    
    # False positives
    fp_transitions = [t for t in zone_transitions if t[0] < drift_start and t[1] != RiskZone.NORMAL]
    fp_rate = (len(fp_transitions) / drift_start * 100) if drift_start > 0 else 0
    print(f"\n      False positives (pre-drift): {len(fp_transitions)} ({fp_rate:.1f}%)")
    
    # Final statistics
    tdi_values = [r['tdi'] for r in results]
    print(f"\n[4.3] TDI Statistics:")
    print(f"      Peak TDI: {max(tdi_values):.1f}")
    print(f"      Mean TDI (normal period): {np.mean(tdi_values[:drift_start]):.1f}")
    print(f"      Mean TDI (drift period): {np.mean(tdi_values[drift_start:]):.1f}")
    
    # Sample explanation
    print("\n[4.4] Sample Explanation (peak drift frame):")
    peak_frame = np.argmax(tdi_values)
    peak_features = test_data[peak_frame]
    peak_result = results[peak_frame]
    
    attributions = attributor.compute_feature_attributions(
        current_features=peak_features,
        top_k=5,
    )
    
    explanation_obj = attributor.generate_explanation(
        current_features=peak_features,
        threat_deviation_index=peak_result['tdi'],
        risk_zone=peak_result['zone'].name,
        trend_direction=peak_result['trend'].name,
        confidence=peak_result['confidence'],
    )
    
    print(f"\n      {explanation_obj.summary}")
    
    print("\n      Top Contributing Features:")
    for attr in attributions[:5]:
        bar = "█" * int(abs(attr.z_score) * 5)
        print(f"        {attr.feature_name}: {bar} (z={attr.z_score:.2f})")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTELLIGENCE SUMMARY")
    print("=" * 70)
    
    final_zone = results[-1]['zone']
    final_tdi = results[-1]['tdi']
    final_trend = results[-1]['trend']
    
    zone_icon = {
        RiskZone.NORMAL: "🟢",
        RiskZone.WATCH: "🟡",
        RiskZone.WARNING: "🟠",
        RiskZone.CRITICAL: "🔴",
    }.get(final_zone, "⚪")
    
    trend_arrow = {
        DriftTrend.INCREASING: "↑",
        DriftTrend.STABLE: "→",
        DriftTrend.DECREASING: "↓",
    }.get(final_trend, "→")
    
    print(f"""
    Current Status:
    ───────────────
    Threat Deviation Index: {final_tdi:.0f}/100
    Risk Zone: {zone_icon} {final_zone.name}
    Drift Trend: {trend_arrow} {final_trend.name}
    
    Detection Performance:
    ──────────────────────
    Detection Delay: {detection_delay if detection_delay else '—'} frames
    False Positive Rate: {fp_rate:.1f}%
    Peak TDI: {max(tdi_values):.1f}
    """)
    
    print("=" * 70)
    print("Demo complete!")
    print("Run 'streamlit run dashboard/app_intelligence.py' for interactive demo.")
    print("=" * 70)


def run_synthetic_demo():
    """
    Run complete pipeline with synthetic data.
    Demonstrates all components working together.
    """
    print("=" * 70)
    print("NOISE FLOOR - Synthetic Data Demo")
    print("=" * 70)
    
    from src.feature_extraction import create_synthetic_normal_data, create_synthetic_drift_data
    from src.autoencoder import NormalityAutoencoder
    from src.drift_detection import DriftDetector
    from src.watch_zones import WatchZoneClassifier
    from src.baseline_comparison import BaselineComparator
    
    # Step 1: Generate Data
    print("\n📊 Step 1: Generating synthetic behavioral data...")
    normal_train = create_synthetic_normal_data(500)
    normal_test = create_synthetic_normal_data(100)
    drift_test = create_synthetic_drift_data(200, drift_rate=0.015)
    
    test_data = np.vstack([normal_test, drift_test])
    drift_start = len(normal_test)
    
    print(f"   Training data: {len(normal_train)} samples")
    print(f"   Test data: {len(test_data)} samples")
    print(f"   Drift starts at frame: {drift_start}")
    
    # Step 2: Train Autoencoder (on normalized data)
    print("\n🧠 Step 2: Training normality autoencoder...")
    
    # Normalize training data first
    train_mean = np.mean(normal_train, axis=0)
    train_std = np.std(normal_train, axis=0)
    train_std[train_std == 0] = 1
    train_normalized = (normal_train - train_mean) / train_std
    
    autoencoder = NormalityAutoencoder(input_dim=normal_train.shape[1])
    autoencoder.compile()
    history = autoencoder.train(train_normalized, epochs=50, verbose=0)
    print(f"   Final training loss: {history['loss'][-1]:.6f}")
    
    # Step 3: Process Test Data
    print("\n🔍 Step 3: Processing test data...")
    
    # Normalize test data using training statistics
    test_normalized = (test_data - train_mean) / train_std
    
    # Get normality scores
    normality_scores = autoencoder.get_normality_score(test_normalized)
    
    # Step 4: Detect Drift
    print("\n📈 Step 4: Running drift detection...")
    # Use first 50 frames for baseline (well within normal_test portion of 100 frames)
    detector = DriftDetector(baseline_frames=50)
    states = detector.process_batch(normality_scores)
    drift_scores = np.array([s.drift_score for s in states])
    
    # Step 5: Classify Zones
    print("\n🚦 Step 5: Classifying watch zones...")
    classifier = WatchZoneClassifier()
    
    zone_transitions = []
    current_zone = None
    
    for i, state in enumerate(states):
        zone_state = classifier.classify(state.drift_score)
        if zone_state.zone != current_zone:
            zone_transitions.append((i, zone_state.zone))
            current_zone = zone_state.zone
    
    print("\n   Zone Transitions:")
    for frame, zone in zone_transitions:
        marker = "← DRIFT STARTS" if frame == drift_start else ""
        print(f"      Frame {frame:4d}: {zone.icon} {zone} {marker}")
    
    # Step 6: Compare with Baselines
    print("\n📊 Step 6: Comparing with baseline methods...")
    comparator = BaselineComparator()
    comparator.fit_baselines(normal_train)
    results = comparator.compare(test_data, drift_scores, drift_start)
    comparator.print_comparison(results, drift_start)
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    
    # FIX 3: Detection is ONLY valid AFTER drift starts
    # Find first warning/alert that occurs AFTER drift_start
    first_detection = None
    for frame, zone in zone_transitions:
        if frame >= drift_start and zone.value in ['watch', 'warning', 'alert']:
            first_detection = frame
            break
    
    if first_detection is not None:
        detection_delay = first_detection - drift_start
        print(f"\n✅ NOISE FLOOR detected drift at frame {first_detection}")
        print(f"   Detection delay: {detection_delay} frames after drift started")
        
        if detection_delay < 30:
            print(f"   ⚡ Early detection achieved!")
    else:
        print("\n⚠️  No warning/alert triggered during test (after drift started)")
    
    # FIX 4: False positives = alerts BEFORE drift starts only
    # Count zones that triggered BEFORE the drift began
    fp_count = sum(1 for f, z in zone_transitions 
                   if f < drift_start and z.value in ['watch', 'warning', 'alert'])
    fp_rate = (fp_count / drift_start * 100) if drift_start > 0 else 0
    print(f"\n📉 False positives before drift: {fp_count} ({fp_rate:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Demo complete! Run 'streamlit run dashboard/app.py' for interactive demo.")
    print("=" * 70)


def launch_dashboard(intelligence: bool = False):
    """Launch the Streamlit dashboard."""
    import subprocess
    
    if intelligence:
        dashboard_path = PROJECT_ROOT / "dashboard" / "app_intelligence.py"
        print("Launching Defense Intelligence Dashboard...")
    else:
        dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
        print("Launching Standard Dashboard...")
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        return
    
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def train_on_video(video_path: str, output_dir: str = "models"):
    """Train the system on a video file."""
    from src.feature_extraction import FeatureExtractor
    from src.autoencoder import NormalityAutoencoder
    
    print(f"Training on video: {video_path}")
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_from_video(video_path)
    feature_matrix = extractor.features_to_matrix(features)
    
    print(f"Extracted {len(features)} frames with {feature_matrix.shape[1]} features each")
    
    # Normalize
    normalized = extractor.normalize_features(feature_matrix, fit=True)
    
    # Train autoencoder
    autoencoder = NormalityAutoencoder(input_dim=feature_matrix.shape[1])
    autoencoder.compile()
    autoencoder.train(normalized)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    autoencoder.save(str(output_path / "autoencoder"))
    
    print(f"Model saved to {output_path}")


def analyze_video(video_path: str, model_dir: str = "models"):
    """Analyze a video for drift using trained model."""
    from src.feature_extraction import FeatureExtractor
    from src.autoencoder import NormalityAutoencoder
    from src.drift_detection import DriftDetector
    from src.watch_zones import WatchZoneClassifier
    
    print(f"Analyzing video: {video_path}")
    
    # Load model
    autoencoder = NormalityAutoencoder()
    autoencoder.load(str(Path(model_dir) / "autoencoder"))
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_from_video(video_path)
    feature_matrix = extractor.features_to_matrix(features)
    normalized = extractor.normalize_features(feature_matrix)
    
    # Get normality scores
    normality_scores = autoencoder.get_normality_score(normalized)
    
    # Detect drift
    detector = DriftDetector()
    classifier = WatchZoneClassifier()
    
    print("\nAnalysis Results:")
    print("-" * 50)
    
    current_zone = None
    for i, score in enumerate(normality_scores):
        drift_state = detector.update(score, frame_index=i)
        zone_state = classifier.classify(drift_state.drift_score)
        
        if zone_state.zone != current_zone:
            print(f"Frame {i:5d}: {zone_state.zone.icon} {zone_state.zone} "
                  f"(drift={drift_state.drift_score:.2f})")
            current_zone = zone_state.zone


def run_real_video_demo(video_path: str = None):
    """
    Run intelligence pipeline on real surveillance video.
    
    "The intelligence pipeline is feature-driven, not data-source-driven."
    
    Args:
        video_path: Path to video file, or None to use UCSD dataset
    """
    print("=" * 70)
    print("NOISE FLOOR - Real Video Intelligence Demo")
    print("Border Surveillance Early Warning System")
    print("=" * 70)
    print(f"\nSession: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNOTE: Public surveillance dataset used as proxy for border CCTV footage.")
    print("Labels intentionally ignored - unsupervised learning only.\n")
    
    # Import modules
    from src.video_features import RealVideoFeatureExtractor, UCSDDatasetLoader
    from src.lstm_vae import TemporalNormalityLSTMVAE
    from src.drift_intelligence import DriftIntelligenceEngine, DriftTrend
    from src.risk_zones import RiskZoneClassifier, RiskZone
    from src.explainability import DriftAttributor
    from src.baseline_freeze import BaselineFreezeManager
    from src.behavioral_features import BEHAVIORAL_FEATURES
    
    n_features = len(BEHAVIORAL_FEATURES)
    
    # =========================================================================
    # STEP 1: LOAD REAL VIDEO DATA
    # =========================================================================
    print("─" * 70)
    print("📹 PHASE 1: Real Video Processing")
    print("─" * 70)
    
    feature_extractor = RealVideoFeatureExtractor(
        frame_size=(224, 224),
        sample_interval=3,
    )
    
    if video_path and Path(video_path).exists():
        # Custom video
        print(f"\n[1.1] Processing custom video: {video_path}")
        features, metadata = feature_extractor.extract_from_video(
            video_path, max_frames=500
        )
        train_features = features[:int(len(features) * 0.6)]
        test_features = features[int(len(features) * 0.6):]
        
    else:
        # UCSD Dataset
        print(f"\n[1.1] Loading UCSD Anomaly Detection Dataset...")
        print(f"      Path: {UCSD_DATASET_PATH}")
        print(f"      Subset: {UCSD_SUBSET}")
        
        loader = UCSDDatasetLoader(UCSD_DATASET_PATH, UCSD_SUBSET)
        
        if not loader.is_available():
            print("\n⚠️  UCSD dataset not found!")
            print(f"    Expected path: {UCSD_DATASET_PATH}/UCSD{UCSD_SUBSET}")
            print("\n    To download:")
            print("    http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm")
            print("\n    Falling back to synthetic data demo...")
            run_intelligence_demo(data_mode="synthetic")
            return
        
        # Extract training features (NORMAL behavior only)
        print("\n[1.2] Extracting features from NORMAL sequences...")
        train_features = feature_extractor.extract_from_frames(
            loader.load_train_frames(max_sequences=10),
            max_frames=1000
        )
        
        # Extract test features (contains anomalies - labels IGNORED)
        print("\n[1.3] Extracting features from TEST sequences...")
        print("      (Anomaly labels intentionally ignored)")
        feature_extractor.reset()
        test_features = feature_extractor.extract_from_frames(
            loader.load_test_frames(max_sequences=5),
            max_frames=500
        )
    
    print(f"\n      Training samples: {len(train_features)}")
    print(f"      Test samples: {len(test_features)}")
    print(f"      Features per sample: {n_features}")
    
    # =========================================================================
    # STEP 2: INITIALIZE BASELINE FREEZE MANAGER
    # =========================================================================
    print("\n[1.4] Initializing baseline freeze manager...")
    baseline_manager = BaselineFreezeManager(
        feature_dim=n_features,
        learning_window=min(200, len(train_features)),
    )
    
    # =========================================================================
    # STEP 3: COMPUTE BASELINE (FROZEN)
    # =========================================================================
    print("\n" + "─" * 70)
    print("📊 PHASE 2: Baseline Learning (Will be FROZEN)")
    print("─" * 70)
    
    baseline_means = np.mean(train_features, axis=0)
    baseline_stds = np.std(train_features, axis=0) + 1e-6
    
    print(f"\n[2.1] Baseline statistics computed from {len(train_features)} NORMAL samples")
    print(f"      Feature mean range: [{baseline_means.min():.4f}, {baseline_means.max():.4f}]")
    
    # Feed to baseline manager
    for features in train_features[:baseline_manager.learning_window]:
        baseline_manager.add_learning_sample(features)
    
    print(f"\n      ✓ Baseline FROZEN (human-gated adaptation only)")
    
    # =========================================================================
    # STEP 4: TRAIN LSTM-VAE
    # =========================================================================
    print("\n[2.2] Training LSTM-VAE on NORMAL behavior...")
    
    lstm_vae = TemporalNormalityLSTMVAE(
        input_dim=n_features,
        hidden_dim=32,
        latent_dim=8,
        seq_len=10,
    )
    
    # Create sequences
    sequences = []
    for i in range(len(train_features) - 10):
        seq = train_features[i:i+10]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    print(f"      Training sequences: {len(sequences)}")
    
    lstm_vae.train(sequences, epochs=50, verbose=False)
    print("      ✓ LSTM-VAE training complete")
    
    # =========================================================================
    # STEP 5: INITIALIZE INTELLIGENCE COMPONENTS
    # =========================================================================
    print("\n[2.3] Initializing intelligence components...")
    
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
    
    print("      ✓ All components initialized")
    
    # =========================================================================
    # STEP 6: PROCESS TEST DATA
    # =========================================================================
    print("\n" + "─" * 70)
    print("📈 PHASE 3: Intelligence Analysis on TEST data")
    print("─" * 70)
    print("\nNOTE: Absolute values may differ from synthetic demo.")
    print("      Key insight: Drift TRENDS remain consistently detectable.\n")
    
    results = []
    feature_buffer = []
    drift_engine.reset()
    zone_classifier.reset()
    
    zone_transitions = []
    current_zone = None
    first_watch_frame = None
    
    for i, features in enumerate(test_features):
        # Add to feature buffer
        feature_buffer.append(features)
        if len(feature_buffer) > 10:
            feature_buffer.pop(0)
        
        # Compute metrics from LSTM-VAE
        if len(feature_buffer) >= 10:
            sequence = np.array(feature_buffer[-10:])
            seq_input = sequence.reshape(1, 10, -1)
            
            output = lstm_vae.forward(seq_input, training=False)
            reconstruction_loss = output.reconstruction_loss
            kl_divergence = output.kl_divergence
            latent_mean = output.latent_mean[0]
            latent_logvar = output.latent_log_var[0]
        else:
            reconstruction_loss = np.mean((features - baseline_means) ** 2)
            kl_divergence = 0.0
            latent_mean = np.zeros(8)
            latent_logvar = np.zeros(8)
        
        # Process through drift intelligence engine
        intelligence = drift_engine.process(
            reconstruction_loss=reconstruction_loss,
            kl_divergence=kl_divergence,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            features=features,
            frame_index=i,
        )
        
        # Classify risk zone
        zone_state = zone_classifier.classify(
            threat_deviation_index=intelligence.threat_deviation_index,
            z_score=intelligence.z_score,
            trend_slope=intelligence.trend_slope,
            trend_persistence=intelligence.trend_persistence,
        )
        
        results.append({
            'frame': i,
            'tdi': intelligence.threat_deviation_index,
            'zone': zone_state.zone,
            'trend': intelligence.drift_trend,
            'confidence': intelligence.confidence,
        })
        
        # Track zone transitions
        if zone_state.zone != current_zone:
            zone_transitions.append((i, zone_state.zone))
            current_zone = zone_state.zone
            
            if first_watch_frame is None and zone_state.zone != RiskZone.NORMAL:
                first_watch_frame = i
    
    print(f"[3.1] Processed {len(test_features)} test frames")
    
    # =========================================================================
    # STEP 7: RESULTS
    # =========================================================================
    print("\n" + "─" * 70)
    print("📊 PHASE 4: Intelligence Report")
    print("─" * 70)
    
    # Zone transitions
    print("\n[4.1] Zone Transitions (first 15):")
    for frame, zone in zone_transitions[:15]:
        marker = " ← FIRST DRIFT DETECTED" if frame == first_watch_frame else ""
        
        zone_icon = {
            RiskZone.NORMAL: "🟢",
            RiskZone.WATCH: "🟡",
            RiskZone.WARNING: "🟠",
            RiskZone.CRITICAL: "🔴",
        }.get(zone, "⚪")
        
        print(f"      Frame {frame:4d}: {zone_icon} {zone.name}{marker}")
    
    if len(zone_transitions) > 15:
        print(f"      ... and {len(zone_transitions) - 15} more transitions")
    
    # Statistics
    tdi_values = [r['tdi'] for r in results]
    print(f"\n[4.2] TDI Statistics:")
    print(f"      Mean TDI: {np.mean(tdi_values):.1f}")
    print(f"      Max TDI: {max(tdi_values):.1f}")
    print(f"      Std TDI: {np.std(tdi_values):.1f}")
    
    # Zone distribution
    zone_counts = {}
    for r in results:
        zone = r['zone'].name
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    
    print(f"\n[4.3] Time in Zones:")
    for zone, count in zone_counts.items():
        pct = count / len(results) * 100
        print(f"      {zone}: {pct:.1f}%")
    
    # Sample explanation at peak drift
    print("\n[4.4] Sample Explanation (peak drift frame):")
    peak_frame = np.argmax(tdi_values)
    peak_features = test_features[peak_frame]
    peak_result = results[peak_frame]
    
    attributions = attributor.compute_feature_attributions(
        current_features=peak_features,
        top_k=5,
    )
    
    explanation_obj = attributor.generate_explanation(
        current_features=peak_features,
        threat_deviation_index=peak_result['tdi'],
        risk_zone=peak_result['zone'].name,
        trend_direction=peak_result['trend'].name,
        confidence=peak_result['confidence'],
    )
    
    print(f"\n      {explanation_obj.summary}")
    
    print("\n      Top Contributing Features:")
    for attr in attributions[:5]:
        bar = "█" * int(min(10, abs(attr.z_score) * 2))
        print(f"        {attr.feature_name}: {bar} (z={attr.z_score:.2f})")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("REAL VIDEO INTELLIGENCE SUMMARY")
    print("=" * 70)
    
    print(f"""
    Data Source: {"UCSD Anomaly Detection Dataset" if not video_path else video_path}
    Positioning: Public surveillance dataset used as proxy for border CCTV
    
    Key Results:
    ────────────
    Total Frames Analyzed: {len(test_features)}
    Zone Transitions: {len(zone_transitions)}
    Peak TDI: {max(tdi_values):.1f}
    First Elevated Zone: Frame {first_watch_frame if first_watch_frame else 'N/A'}
    
    IMPORTANT NOTES:
    ────────────────
    • "Absolute values differ, but drift trends remain consistently detectable."
    • Labels were intentionally ignored - unsupervised detection only
    • Baseline is FROZEN - human-gated adaptation required for updates
    • This is a TRL-4 lab-validated prototype for decision-support
    
    System Philosophy:
    ──────────────────
    "Defense systems manage CONFIDENCE, not panic."
    "AI assists operators, it does NOT replace them."
    """)
    
    print("=" * 70)
    print("Real video demo complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="NOISE FLOOR - Defense-Grade Behavioral Drift Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo              Run synthetic data intelligence demo
  python main.py --real-demo         Run real video demo (UCSD dataset)
  python main.py --video PATH        Process custom surveillance video
  python main.py --dashboard         Launch defense intelligence dashboard
  python main.py --legacy-demo       Run original synthetic demo
  
Data Modes:
  synthetic   - Controlled testing with generated behavioral data
  real_video  - Real surveillance footage processing

Technology Readiness: TRL-4 (Lab-validated prototype)
"This system is designed for border surveillance and high-security perimeters."
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run synthetic data intelligence demo')
    parser.add_argument('--real-demo', action='store_true',
                       help='Run real video demo (UCSD dataset)')
    parser.add_argument('--video', type=str, metavar='PATH',
                       help='Process custom surveillance video')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch defense intelligence dashboard')
    parser.add_argument('--legacy-demo', action='store_true',
                       help='Run original synthetic demo')
    parser.add_argument('--legacy-dashboard', action='store_true',
                       help='Launch original dashboard')
    parser.add_argument('--train', type=str, metavar='VIDEO',
                       help='Train on video file')
    parser.add_argument('--analyze', type=str, metavar='VIDEO',
                       help='Analyze video for drift')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory for model storage')
    
    args = parser.parse_args()
    
    if args.demo:
        run_intelligence_demo(data_mode="synthetic")
    elif args.real_demo:
        run_real_video_demo()
    elif args.video:
        run_real_video_demo(video_path=args.video)
    elif args.dashboard:
        launch_dashboard(intelligence=True)
    elif args.legacy_demo:
        run_synthetic_demo()
    elif args.legacy_dashboard:
        launch_dashboard(intelligence=False)
    elif args.train:
        train_on_video(args.train, args.model_dir)
    elif args.analyze:
        analyze_video(args.analyze, args.model_dir)
    else:
        # Default: run synthetic demo
        print("\nNo action specified. Running synthetic intelligence demo...")
        print("Use --help to see available options.\n")
        run_intelligence_demo(data_mode="synthetic")


if __name__ == "__main__":
    main()
