"""
NOISE FLOOR - Main Entry Point
================================
Run the complete pipeline or individual components.

Usage:
    python main.py --demo          # Run synthetic demo
    python main.py --dashboard     # Launch Streamlit dashboard
    python main.py --train VIDEO   # Train on video file
    python main.py --analyze VIDEO # Analyze video for drift
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


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


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
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


def main():
    parser = argparse.ArgumentParser(
        description="NOISE FLOOR - Unsupervised Behavioral Drift Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo              Run synthetic data demonstration
  python main.py --dashboard         Launch interactive Streamlit dashboard
  python main.py --train video.mp4   Train on video file
  python main.py --analyze video.mp4 Analyze video for drift
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run synthetic data demo')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch Streamlit dashboard')
    parser.add_argument('--train', type=str, metavar='VIDEO',
                       help='Train on video file')
    parser.add_argument('--analyze', type=str, metavar='VIDEO',
                       help='Analyze video for drift')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory for model storage')
    
    args = parser.parse_args()
    
    if args.demo:
        run_synthetic_demo()
    elif args.dashboard:
        launch_dashboard()
    elif args.train:
        train_on_video(args.train, args.model_dir)
    elif args.analyze:
        analyze_video(args.analyze, args.model_dir)
    else:
        # Default: run demo
        print("No action specified. Running demo...")
        print("Use --help to see available options.\n")
        run_synthetic_demo()


if __name__ == "__main__":
    main()
