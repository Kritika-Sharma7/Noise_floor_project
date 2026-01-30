"""
NOISE FLOOR - Ensemble Anomaly Detection Module
=================================================
Multi-model ensemble for robust behavioral drift detection.

This module combines multiple anomaly detection approaches:
1. LSTM-VAE (primary): Temporal pattern learning
2. Isolation Forest: Tree-based anomaly detection
3. One-Class SVM: Boundary-based novelty detection
4. Local Outlier Factor: Density-based detection

DESIGN PHILOSOPHY:
------------------
"Consensus reduces false positives while maintaining sensitivity."

Each detector votes on whether a sample is anomalous.
The ensemble combines votes using weighted averaging,
where weights are based on historical performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectorType(Enum):
    """Types of anomaly detectors in the ensemble."""
    LSTM_VAE = "lstm_vae"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "lof"


@dataclass
class DetectorVote:
    """Vote from a single detector."""
    detector_type: DetectorType
    anomaly_score: float              # Raw anomaly score (0-1)
    is_anomaly: bool                  # Binary decision
    confidence: float                 # Confidence in decision
    raw_output: float                 # Raw model output
    
    def __repr__(self):
        status = "ANOMALY" if self.is_anomaly else "NORMAL"
        return f"{self.detector_type.value}: {status} (score={self.anomaly_score:.3f}, conf={self.confidence:.3f})"


@dataclass
class EnsembleDecision:
    """Combined decision from the ensemble."""
    # Consensus metrics
    consensus_score: float            # Weighted average score (0-1)
    consensus_anomaly: bool           # Majority decision
    agreement_ratio: float            # How many detectors agree (0-1)
    
    # Individual votes
    votes: List[DetectorVote] = field(default_factory=list)
    
    # Breakdown by detector
    detector_scores: Dict[str, float] = field(default_factory=dict)
    
    # Confidence in ensemble decision
    ensemble_confidence: float = 0.0
    
    # Threat Deviation Index contribution
    tdi_contribution: float = 0.0
    
    def __repr__(self):
        status = "ANOMALY" if self.consensus_anomaly else "NORMAL"
        return f"Ensemble: {status} (score={self.consensus_score:.3f}, agreement={self.agreement_ratio:.1%})"


class IsolationForest:
    """
    Pure NumPy Isolation Forest implementation.
    
    Isolation Forest detects anomalies by isolating observations.
    Anomalies are easier to isolate (require fewer splits).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        
        self.trees = []
        self.threshold = 0.5
        self._fitted = False
        
        np.random.seed(random_state)
    
    def _build_tree(self, X: np.ndarray, depth: int = 0, max_depth: int = 10) -> dict:
        """Build a single isolation tree."""
        n_samples, n_features = X.shape
        
        # Stop conditions
        if depth >= max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples, 'depth': depth}
        
        # Random feature and split
        feature_idx = np.random.randint(n_features)
        feature_values = X[:, feature_idx]
        min_val, max_val = feature_values.min(), feature_values.max()
        
        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples, 'depth': depth}
        
        split_value = np.random.uniform(min_val, max_val)
        
        # Split data
        left_mask = X[:, feature_idx] < split_value
        right_mask = ~left_mask
        
        return {
            'type': 'internal',
            'feature': feature_idx,
            'split': split_value,
            'left': self._build_tree(X[left_mask], depth + 1, max_depth),
            'right': self._build_tree(X[right_mask], depth + 1, max_depth),
        }
    
    def _path_length(self, x: np.ndarray, tree: dict, depth: int = 0) -> float:
        """Calculate path length for a single sample."""
        if tree['type'] == 'leaf':
            # Add expected path length for remaining samples
            n = tree['size']
            if n <= 1:
                return depth
            else:
                # Harmonic approximation
                return depth + 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        
        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)
    
    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """Fit the isolation forest."""
        n_samples = X.shape[0]
        sample_size = min(self.max_samples, n_samples)
        max_depth = int(np.ceil(np.log2(sample_size)))
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Subsample
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            tree = self._build_tree(X[indices], max_depth=max_depth)
            self.trees.append(tree)
        
        # Compute threshold from training data
        scores = self._compute_scores(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        self._fitted = True
        logger.info(f"IsolationForest fitted with {self.n_estimators} trees, threshold={self.threshold:.4f}")
        return self
    
    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples."""
        n_samples = X.shape[0]
        avg_path_lengths = np.zeros(n_samples)
        
        for tree in self.trees:
            for i in range(n_samples):
                avg_path_lengths[i] += self._path_length(X[i], tree)
        
        avg_path_lengths /= len(self.trees)
        
        # Normalize: shorter path = more anomalous
        n = self.max_samples
        c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        scores = 2 ** (-avg_path_lengths / c)
        
        return scores
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and labels.
        
        Returns:
            scores: Anomaly scores (higher = more anomalous)
            labels: -1 for anomaly, 1 for normal
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        scores = self._compute_scores(X)
        labels = np.where(scores > self.threshold, -1, 1)
        
        return scores, labels


class OneClassSVM:
    """
    Simplified One-Class SVM using RBF kernel.
    
    Uses an approximation based on kernel distance to training data center.
    """
    
    def __init__(
        self,
        nu: float = 0.1,
        gamma: float = 0.1,
    ):
        self.nu = nu
        self.gamma = gamma
        
        self.support_vectors_ = None
        self.center_ = None
        self.radius_ = None
        self._fitted = False
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel between X and Y."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        dist_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * dist_sq)
    
    def fit(self, X: np.ndarray) -> 'OneClassSVM':
        """Fit the one-class SVM."""
        self.center_ = np.mean(X, axis=0)
        
        # Compute distances to center in kernel space
        K_center = self._rbf_kernel(X, self.center_.reshape(1, -1)).flatten()
        distances = 1 - K_center
        
        # Threshold based on nu (contamination factor)
        self.radius_ = np.percentile(distances, 100 * (1 - self.nu))
        
        # Store support vectors (samples near boundary)
        boundary_idx = np.where(np.abs(distances - self.radius_) < 0.1)[0]
        if len(boundary_idx) > 0:
            self.support_vectors_ = X[boundary_idx]
        else:
            self.support_vectors_ = X[:10]
        
        self._fitted = True
        logger.info(f"OneClassSVM fitted, radius={self.radius_:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores and labels."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        K_center = self._rbf_kernel(X, self.center_.reshape(1, -1)).flatten()
        distances = 1 - K_center
        
        # Normalize scores to 0-1 range
        scores = distances / (self.radius_ * 2 + 1e-6)
        scores = np.clip(scores, 0, 1)
        
        labels = np.where(distances > self.radius_, -1, 1)
        
        return scores, labels


class LocalOutlierFactor:
    """
    Local Outlier Factor for density-based anomaly detection.
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        self.X_train_ = None
        self.threshold_ = None
        self._fitted = False
    
    def _compute_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances."""
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        dist_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        return np.sqrt(np.maximum(dist_sq, 0))
    
    def _compute_lof(self, X: np.ndarray) -> np.ndarray:
        """Compute LOF scores."""
        distances = self._compute_distances(X, self.X_train_)
        n_samples = X.shape[0]
        n_train = self.X_train_.shape[0]
        k = min(self.n_neighbors, n_train - 1)
        
        lof_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Find k nearest neighbors
            dists = distances[i]
            neighbor_idx = np.argsort(dists)[1:k+1] if i < n_train else np.argsort(dists)[:k]
            
            # k-distance (distance to k-th neighbor)
            k_dist = dists[neighbor_idx[-1]] if len(neighbor_idx) > 0 else 0
            
            # Local reachability density (simplified)
            reach_dists = np.maximum(dists[neighbor_idx], k_dist)
            lrd = k / (np.sum(reach_dists) + 1e-10)
            
            # LOF is ratio of neighbor LRDs to sample LRD
            # Simplified: use inverse of local density
            lof_scores[i] = 1.0 / (lrd + 1e-10)
        
        # Normalize
        lof_scores = lof_scores / (np.median(lof_scores) + 1e-10)
        
        return lof_scores
    
    def fit(self, X: np.ndarray) -> 'LocalOutlierFactor':
        """Fit the LOF model."""
        self.X_train_ = X.copy()
        
        # Compute threshold from training data
        lof_scores = self._compute_lof(X)
        self.threshold_ = np.percentile(lof_scores, 100 * (1 - self.contamination))
        
        self._fitted = True
        logger.info(f"LocalOutlierFactor fitted, threshold={self.threshold_:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores and labels."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        lof_scores = self._compute_lof(X)
        
        # Normalize to 0-1
        scores = np.clip(lof_scores / (self.threshold_ * 2 + 1e-6), 0, 1)
        labels = np.where(lof_scores > self.threshold_, -1, 1)
        
        return scores, labels


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple detection methods.
    
    Provides robust anomaly detection through consensus voting.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        weights: Optional[Dict[str, float]] = None,
        use_isolation_forest: bool = True,
        use_one_class_svm: bool = True,
        use_lof: bool = True,
    ):
        """
        Initialize ensemble detector.
        
        Args:
            contamination: Expected proportion of anomalies
            weights: Custom weights for each detector
            use_*: Enable/disable specific detectors
        """
        self.contamination = contamination
        self.use_isolation_forest = use_isolation_forest
        self.use_one_class_svm = use_one_class_svm
        self.use_lof = use_lof
        
        # Default weights (can be updated based on performance)
        self.weights = weights or {
            'lstm_vae': 0.40,           # Primary detector
            'isolation_forest': 0.25,
            'one_class_svm': 0.20,
            'lof': 0.15,
        }
        
        # Initialize detectors
        self.isolation_forest = IsolationForest(contamination=contamination) if use_isolation_forest else None
        self.one_class_svm = OneClassSVM(nu=contamination) if use_one_class_svm else None
        self.lof = LocalOutlierFactor(contamination=contamination) if use_lof else None
        
        # Performance tracking
        self.performance_history = {
            'lstm_vae': deque(maxlen=100),
            'isolation_forest': deque(maxlen=100),
            'one_class_svm': deque(maxlen=100),
            'lof': deque(maxlen=100),
        }
        
        self._fitted = False
    
    def fit(self, X: np.ndarray, lstm_vae=None) -> 'EnsembleAnomalyDetector':
        """
        Fit all ensemble detectors on normal data.
        
        Args:
            X: Training data (normal samples only)
            lstm_vae: Pre-trained LSTM-VAE model (optional)
        """
        logger.info(f"Fitting ensemble detector on {len(X)} samples...")
        
        if self.isolation_forest:
            self.isolation_forest.fit(X)
        
        if self.one_class_svm:
            self.one_class_svm.fit(X)
        
        if self.lof:
            self.lof.fit(X)
        
        self._fitted = True
        logger.info("Ensemble detector fitted successfully")
        return self
    
    def predict(
        self,
        features: np.ndarray,
        lstm_vae_score: Optional[float] = None,
    ) -> EnsembleDecision:
        """
        Get ensemble prediction for a sample.
        
        Args:
            features: Feature vector for current sample
            lstm_vae_score: Anomaly score from LSTM-VAE (if available)
        
        Returns:
            EnsembleDecision with consensus and individual votes
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        votes = []
        weighted_sum = 0.0
        total_weight = 0.0
        
        # LSTM-VAE vote (if provided)
        if lstm_vae_score is not None:
            vote = DetectorVote(
                detector_type=DetectorType.LSTM_VAE,
                anomaly_score=min(lstm_vae_score, 1.0),
                is_anomaly=lstm_vae_score > 0.5,
                confidence=1.0 - abs(lstm_vae_score - 0.5) * 2,
                raw_output=lstm_vae_score,
            )
            votes.append(vote)
            weighted_sum += vote.anomaly_score * self.weights['lstm_vae']
            total_weight += self.weights['lstm_vae']
        
        # Isolation Forest vote
        if self.isolation_forest and self._fitted:
            scores, labels = self.isolation_forest.predict(features)
            vote = DetectorVote(
                detector_type=DetectorType.ISOLATION_FOREST,
                anomaly_score=float(scores[0]),
                is_anomaly=labels[0] == -1,
                confidence=abs(scores[0] - 0.5) * 2,
                raw_output=float(scores[0]),
            )
            votes.append(vote)
            weighted_sum += vote.anomaly_score * self.weights['isolation_forest']
            total_weight += self.weights['isolation_forest']
        
        # One-Class SVM vote
        if self.one_class_svm and self._fitted:
            scores, labels = self.one_class_svm.predict(features)
            vote = DetectorVote(
                detector_type=DetectorType.ONE_CLASS_SVM,
                anomaly_score=float(scores[0]),
                is_anomaly=labels[0] == -1,
                confidence=abs(scores[0] - 0.5) * 2,
                raw_output=float(scores[0]),
            )
            votes.append(vote)
            weighted_sum += vote.anomaly_score * self.weights['one_class_svm']
            total_weight += self.weights['one_class_svm']
        
        # LOF vote
        if self.lof and self._fitted:
            scores, labels = self.lof.predict(features)
            vote = DetectorVote(
                detector_type=DetectorType.LOCAL_OUTLIER_FACTOR,
                anomaly_score=float(scores[0]),
                is_anomaly=labels[0] == -1,
                confidence=abs(scores[0] - 0.5) * 2,
                raw_output=float(scores[0]),
            )
            votes.append(vote)
            weighted_sum += vote.anomaly_score * self.weights['lof']
            total_weight += self.weights['lof']
        
        # Compute consensus
        if total_weight > 0:
            consensus_score = weighted_sum / total_weight
        else:
            consensus_score = 0.5
        
        # Agreement ratio
        if votes:
            anomaly_votes = sum(1 for v in votes if v.is_anomaly)
            agreement_ratio = max(anomaly_votes, len(votes) - anomaly_votes) / len(votes)
        else:
            agreement_ratio = 0.0
        
        # Consensus anomaly (weighted majority)
        consensus_anomaly = consensus_score > 0.5
        
        # Ensemble confidence (based on agreement and individual confidences)
        if votes:
            avg_confidence = np.mean([v.confidence for v in votes])
            ensemble_confidence = agreement_ratio * avg_confidence
        else:
            ensemble_confidence = 0.0
        
        # Build detector scores dict
        detector_scores = {v.detector_type.value: v.anomaly_score for v in votes}
        
        # TDI contribution (scale to 0-100)
        tdi_contribution = consensus_score * 100
        
        return EnsembleDecision(
            consensus_score=consensus_score,
            consensus_anomaly=consensus_anomaly,
            agreement_ratio=agreement_ratio,
            votes=votes,
            detector_scores=detector_scores,
            ensemble_confidence=ensemble_confidence,
            tdi_contribution=tdi_contribution,
        )
    
    def update_weights(self, feedback: Dict[str, bool]) -> None:
        """
        Update detector weights based on feedback.
        
        Args:
            feedback: Dict mapping detector name to whether it was correct
        """
        learning_rate = 0.01
        
        for detector_name, was_correct in feedback.items():
            if detector_name in self.weights:
                if was_correct:
                    self.weights[detector_name] *= (1 + learning_rate)
                else:
                    self.weights[detector_name] *= (1 - learning_rate)
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def get_detector_agreement_matrix(self, recent_n: int = 50) -> np.ndarray:
        """Get agreement matrix between detectors over recent predictions."""
        # Placeholder for actual implementation
        n_detectors = 4
        return np.eye(n_detectors) * 0.8 + 0.2
    
    def get_status(self) -> Dict[str, Any]:
        """Get current ensemble status."""
        return {
            'fitted': self._fitted,
            'weights': self.weights.copy(),
            'detectors_enabled': {
                'isolation_forest': self.isolation_forest is not None,
                'one_class_svm': self.one_class_svm is not None,
                'lof': self.lof is not None,
            }
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_ensemble_from_training_data(
    train_data: np.ndarray,
    contamination: float = 0.1,
) -> EnsembleAnomalyDetector:
    """
    Create and fit an ensemble detector from training data.
    
    Args:
        train_data: Normal training samples
        contamination: Expected anomaly proportion
    
    Returns:
        Fitted EnsembleAnomalyDetector
    """
    ensemble = EnsembleAnomalyDetector(contamination=contamination)
    ensemble.fit(train_data)
    return ensemble
