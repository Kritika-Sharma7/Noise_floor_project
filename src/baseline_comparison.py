"""
NOISE FLOOR - Baseline Comparison Module
=========================================
Comparison with traditional anomaly detection methods.

GRAY-BOX EXPLANATION:
---------------------
This module implements traditional anomaly detection methods
to demonstrate why NOISE FLOOR's drift detection is superior.

TRADITIONAL METHODS (Baseline):
1. Isolation Forest: Tree-based anomaly detection
2. Threshold Alerting: Simple statistical thresholds
3. One-Class SVM: Boundary-based anomaly detection

PROBLEMS WITH TRADITIONAL METHODS:
- React only to INSTANT anomalies
- High false positive rate
- Miss GRADUAL drift
- No temporal context
- Binary (normal/anomaly) output

NOISE FLOOR ADVANTAGES:
- Detects GRADUAL changes
- Uses temporal aggregation
- Graduated alert levels
- Lower false positives
- Earlier detection of slow threats
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from baseline comparison."""
    method_name: str
    detection_frame: Optional[int]      # First frame with detection
    total_alerts: int                   # Total number of alert frames
    false_positive_rate: float          # Alerts during normal period
    detection_delay: int                # Frames after drift starts
    scores: np.ndarray                  # Raw scores from method


class IsolationForestBaseline:
    """
    Isolation Forest anomaly detection baseline.
    
    HOW IT WORKS:
    -------------
    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies are easier to isolate (fewer splits needed).
    
    LIMITATIONS:
    - No temporal context (treats each frame independently)
    - Binary output (anomaly or not)
    - Misses gradual drift (no trend detection)
    - High false positive rate on noise
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X_normal: np.ndarray):
        """Fit on normal data."""
        X_scaled = self.scaler.fit_transform(X_normal)
        self.model.fit(X_scaled)
        self.fitted = True
        logger.info(f"IsolationForest fitted on {len(X_normal)} samples")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Returns:
            labels: 1 for normal, -1 for anomaly
            scores: Anomaly scores (lower = more anomalous)
        """
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        return labels, scores


class ThresholdBaseline:
    """
    Simple threshold-based alerting baseline.
    
    HOW IT WORKS:
    -------------
    Alert if any feature exceeds mean ± k*std.
    
    LIMITATIONS:
    - No learning of complex patterns
    - Very high false positive rate
    - Cannot adapt to multimodal distributions
    - Misses subtle multi-feature drift
    """
    
    def __init__(self, k: float = 3.0):
        self.k = k
        self.means = None
        self.stds = None
    
    def fit(self, X_normal: np.ndarray):
        """Compute mean and std from normal data."""
        self.means = np.mean(X_normal, axis=0)
        self.stds = np.std(X_normal, axis=0)
        self.stds[self.stds == 0] = 1  # Prevent division by zero
        logger.info(f"Threshold baseline fitted with k={self.k}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on threshold.
        
        Returns:
            labels: 1 for normal, -1 for anomaly
            scores: Maximum z-score per sample
        """
        z_scores = np.abs((X - self.means) / self.stds)
        max_z_scores = np.max(z_scores, axis=1)
        labels = np.where(max_z_scores > self.k, -1, 1)
        return labels, -max_z_scores  # Negative to match IF convention


class OneClassSVMBaseline:
    """
    One-Class SVM anomaly detection baseline.
    
    HOW IT WORKS:
    -------------
    Learns a boundary around normal data. Points outside
    the boundary are classified as anomalies.
    
    LIMITATIONS:
    - Sensitive to hyperparameters
    - Slow on large datasets
    - No temporal modeling
    - Binary output
    """
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf'):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma='scale')
        self.scaler = StandardScaler()
    
    def fit(self, X_normal: np.ndarray):
        """Fit on normal data."""
        X_scaled = self.scaler.fit_transform(X_normal)
        self.model.fit(X_scaled)
        logger.info(f"OneClassSVM fitted on {len(X_normal)} samples")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies."""
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        return labels, scores


class BaselineComparator:
    """
    Compare NOISE FLOOR with baseline methods.
    
    COMPARISON METRICS:
    -------------------
    1. Detection Frame: When was drift first detected?
    2. Detection Delay: How long after drift started?
    3. False Positive Rate: Alerts during known-normal period
    4. Total Alerts: How many frames triggered alerts?
    
    EXPECTED RESULTS:
    - NOISE FLOOR: Early detection, low false positives
    - Baselines: Late detection OR high false positives
    """
    
    def __init__(self):
        self.baselines = {
            'isolation_forest': IsolationForestBaseline(),
            'threshold': ThresholdBaseline(),
            'one_class_svm': OneClassSVMBaseline(),
        }
    
    def fit_baselines(self, X_normal: np.ndarray):
        """Fit all baseline methods on normal data."""
        for name, baseline in self.baselines.items():
            logger.info(f"Fitting {name}...")
            baseline.fit(X_normal)
    
    def compare(
        self,
        X_test: np.ndarray,
        noise_floor_drift_scores: np.ndarray,
        drift_start_frame: int,
        noise_floor_threshold: float = 2.0
    ) -> Dict[str, ComparisonResult]:
        """
        Compare NOISE FLOOR with baselines.
        
        Args:
            X_test: Test feature matrix
            noise_floor_drift_scores: Drift scores from NOISE FLOOR
            drift_start_frame: Frame where actual drift begins
            noise_floor_threshold: Threshold for NOISE FLOOR alerts
            
        Returns:
            Dictionary of comparison results for each method
            
        CRITICAL FIXES:
        - FP Rate = alerts BEFORE drift_start_frame only
        - Detection = first alert AFTER drift_start_frame
        """
        results = {}
        
        # NOISE FLOOR results
        nf_alerts = noise_floor_drift_scores > noise_floor_threshold
        
        # FIX: Detection must be AFTER drift starts
        nf_detection = None
        for i in range(drift_start_frame, len(nf_alerts)):
            if nf_alerts[i]:
                nf_detection = i
                break
        
        # FIX: FP = alerts BEFORE drift starts
        fp_before_drift = np.sum(nf_alerts[:drift_start_frame]) if drift_start_frame > 0 else 0
        fp_rate = float(fp_before_drift / drift_start_frame) if drift_start_frame > 0 else 0.0
        
        results['noise_floor'] = ComparisonResult(
            method_name='NOISE FLOOR',
            detection_frame=nf_detection,
            total_alerts=int(np.sum(nf_alerts)),
            false_positive_rate=fp_rate,
            detection_delay=nf_detection - drift_start_frame if nf_detection is not None else -1,
            scores=noise_floor_drift_scores,
        )
        
        # Baseline results
        for name, baseline in self.baselines.items():
            labels, scores = baseline.predict(X_test)
            alerts = labels == -1
            
            # FIX: Detection must be AFTER drift starts
            detection = None
            for i in range(drift_start_frame, len(alerts)):
                if alerts[i]:
                    detection = i
                    break
            
            # FIX: FP = alerts BEFORE drift starts
            baseline_fp_count = np.sum(alerts[:drift_start_frame]) if drift_start_frame > 0 else 0
            baseline_fp_rate = float(baseline_fp_count / drift_start_frame) if drift_start_frame > 0 else 0.0
            
            results[name] = ComparisonResult(
                method_name=name.replace('_', ' ').title(),
                detection_frame=detection,
                total_alerts=int(np.sum(alerts)),
                false_positive_rate=baseline_fp_rate,
                detection_delay=detection - drift_start_frame if detection is not None else -1,
                scores=-scores,  # Invert to match drift score convention
            )
        
        return results
    
    def print_comparison(self, results: Dict[str, ComparisonResult], drift_start_frame: int):
        """Print formatted comparison results."""
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON RESULTS")
        print("=" * 70)
        print(f"\nDrift starts at frame: {drift_start_frame}")
        print("-" * 70)
        print(f"{'Method':<25} {'Detection':<12} {'Delay':<10} {'FP Rate':<12} {'Total Alerts'}")
        print("-" * 70)
        
        for name, result in results.items():
            detection = result.detection_frame if result.detection_frame else "N/A"
            delay = result.detection_delay if result.detection_delay >= 0 else "N/A"
            
            print(f"{result.method_name:<25} {str(detection):<12} {str(delay):<10} "
                  f"{result.false_positive_rate*100:.1f}%{'':<8} {result.total_alerts}")
        
        print("-" * 70)
        
        # Highlight NOISE FLOOR advantages
        nf_result = results.get('noise_floor')
        if nf_result:
            print("\n📊 NOISE FLOOR ANALYSIS:")
            
            # Compare detection delay
            baseline_delays = [r.detection_delay for r in results.values() 
                             if r.detection_delay >= 0 and 'noise_floor' not in r.method_name.lower()]
            if baseline_delays and nf_result.detection_delay >= 0:
                avg_baseline_delay = np.mean(baseline_delays)
                if nf_result.detection_delay < avg_baseline_delay:
                    print(f"   ✅ Detected drift {avg_baseline_delay - nf_result.detection_delay:.0f} frames "
                          f"earlier than baseline average")
            
            # Compare false positive rate
            baseline_fps = [r.false_positive_rate for r in results.values() 
                          if 'noise_floor' not in r.method_name.lower()]
            if baseline_fps:
                avg_baseline_fp = np.mean(baseline_fps)
                if nf_result.false_positive_rate < avg_baseline_fp:
                    print(f"   ✅ {(avg_baseline_fp - nf_result.false_positive_rate)*100:.1f}% fewer "
                          f"false positives than baseline average")
        
        print("=" * 70 + "\n")


def run_comparison_demo():
    """
    Run a demonstration of baseline comparison.
    
    Creates synthetic data and compares all methods.
    """
    from .feature_extraction import create_synthetic_normal_data, create_synthetic_drift_data
    from .autoencoder import NormalityAutoencoder
    from .drift_detection import DriftDetector
    
    print("=" * 70)
    print("NOISE FLOOR vs TRADITIONAL METHODS - COMPARISON DEMO")
    print("=" * 70)
    
    # Create data
    print("\n1. Generating synthetic data...")
    normal_train = create_synthetic_normal_data(500)
    normal_test = create_synthetic_normal_data(100)
    drift_test = create_synthetic_drift_data(200, drift_rate=0.015)
    
    # Combine test data
    test_data = np.vstack([normal_test, drift_test])
    drift_start = len(normal_test)
    
    print(f"   Training data: {len(normal_train)} samples")
    print(f"   Test data: {len(test_data)} samples (drift starts at {drift_start})")
    
    # Train NOISE FLOOR
    print("\n2. Training NOISE FLOOR autoencoder...")
    autoencoder = NormalityAutoencoder(input_dim=normal_train.shape[1])
    autoencoder.compile()
    autoencoder.train(normal_train, epochs=30, verbose=0)
    
    # Get NOISE FLOOR drift scores
    print("\n3. Computing NOISE FLOOR drift scores...")
    normality_scores = autoencoder.get_normality_score(test_data)
    
    detector = DriftDetector(baseline_frames=50)
    states = detector.process_batch(normality_scores)
    drift_scores = np.array([s.drift_score for s in states])
    
    # Compare with baselines
    print("\n4. Running baseline comparisons...")
    comparator = BaselineComparator()
    comparator.fit_baselines(normal_train)
    
    results = comparator.compare(
        test_data,
        drift_scores,
        drift_start_frame=drift_start
    )
    
    # Print results
    comparator.print_comparison(results, drift_start)
    
    return results


if __name__ == "__main__":
    run_comparison_demo()
