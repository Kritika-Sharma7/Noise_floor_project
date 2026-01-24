"""
NOISE FLOOR - Autoencoder Module
=================================
Gray-box autoencoder for learning "normality" from behavioral features.
Pure NumPy implementation for maximum compatibility.

GRAY-BOX EXPLANATION:
---------------------
The autoencoder is the core of our normality learning system.

HOW IT WORKS:
1. ENCODER: Compresses input features to a small "bottleneck"
   - Forces the model to learn efficient representation
   - Captures the essential patterns of normal behavior
   
2. BOTTLENECK: The compressed representation
   - Contains only the most important information
   - Represents "what normal behavior looks like"
   
3. DECODER: Reconstructs the input from bottleneck
   - If input is similar to training data → low error
   - If input is different from training data → high error

RECONSTRUCTION ERROR = DEVIATION FROM NORMALITY
- Low error: Behavior matches learned normal patterns
- High error: Behavior differs from normal patterns

WHY THIS IS GRAY-BOX:
- Architecture is simple and interpretable
- Each layer's purpose is documented
- Reconstruction error has clear meaning
- Feature contributions to error can be analyzed
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NormalityAutoencoder:
    """
    Autoencoder for learning normal behavioral patterns.
    Pure NumPy implementation for maximum Python compatibility.
    
    ARCHITECTURE (Gray-Box Design):
    --------------------------------
    Input (14 features) → Encoder → Bottleneck (8) → Decoder → Output (14)
    
    The bottleneck forces compression, so only "normal" patterns
    that appear frequently in training data are learned efficiently.
    """
    
    def __init__(
        self,
        input_dim: int = 14,
        encoding_dims: List[int] = [48, 32, 16],
        bottleneck_dim: int = 8,
        dropout_rate: float = 0.1,
        name: str = "normality_autoencoder"
    ):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dims: Sizes of encoder layers
            bottleneck_dim: Size of compressed representation
            dropout_rate: Dropout for regularization
            name: Model name for saving/loading
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.name = name
        self.learning_rate = 0.001
        
        # Build weights
        self.weights = {}
        self._build_weights()
        
        # Training history
        self.history = None
        
        # Statistics for normality scoring
        self.reconstruction_stats = {
            'mean': 0.0,
            'std': 1.0,
            'threshold_percentiles': {}
        }
        
        logger.info(f"Autoencoder initialized: {input_dim} → {bottleneck_dim} → {input_dim}")
    
    def _build_weights(self):
        """Initialize weights for all layers using smaller initialization."""
        np.random.seed(42)
        
        # Encoder weights - use smaller initialization to prevent overflow
        dims = [self.input_dim] + self.encoding_dims + [self.bottleneck_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(1.0 / dims[i])  # Smaller init
            self.weights[f'enc_w{i}'] = np.random.randn(dims[i], dims[i+1]) * scale * 0.1
            self.weights[f'enc_b{i}'] = np.zeros(dims[i+1])
        
        # Decoder weights
        decoder_dims = [self.bottleneck_dim] + list(reversed(self.encoding_dims)) + [self.input_dim]
        for i in range(len(decoder_dims) - 1):
            scale = np.sqrt(1.0 / decoder_dims[i])  # Smaller init
            self.weights[f'dec_w{i}'] = np.random.randn(decoder_dims[i], decoder_dims[i+1]) * scale * 0.1
            self.weights[f'dec_b{i}'] = np.zeros(decoder_dims[i+1])
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def _forward(self, X: np.ndarray, training: bool = False) -> Tuple[np.ndarray, dict]:
        """Forward pass through the autoencoder."""
        cache = {'input': X}
        
        # Encoder
        h = X
        num_enc_layers = len(self.encoding_dims) + 1
        for i in range(num_enc_layers):
            z = h @ self.weights[f'enc_w{i}'] + self.weights[f'enc_b{i}']
            h = self._relu(z)
            cache[f'enc_z{i}'] = z
            cache[f'enc_h{i}'] = h
        
        bottleneck = h
        cache['bottleneck'] = bottleneck
        
        # Decoder
        num_dec_layers = len(self.encoding_dims) + 1
        for i in range(num_dec_layers - 1):
            z = h @ self.weights[f'dec_w{i}'] + self.weights[f'dec_b{i}']
            h = self._relu(z)
            cache[f'dec_z{i}'] = z
            cache[f'dec_h{i}'] = h
        
        # Final layer (linear activation)
        i = num_dec_layers - 1
        output = h @ self.weights[f'dec_w{i}'] + self.weights[f'dec_b{i}']
        cache['output'] = output
        
        return output, cache
    
    def _clip_gradient(self, grad: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        """Clip gradient to prevent exploding gradients."""
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * max_norm / norm
        return grad

    def _backward(self, cache: dict, learning_rate: float = 0.0001) -> float:
        """Backward pass for training with gradient clipping."""
        X = cache['input']
        output = cache['output']
        batch_size = X.shape[0]
        
        # Compute loss (handle NaN)
        diff = X - output
        if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
            return float('inf')
        
        loss = np.mean(diff ** 2)
        
        # Gradient of loss w.r.t. output
        dout = 2 * (output - X) / batch_size
        dout = self._clip_gradient(dout)
        
        # Backprop through decoder
        num_dec_layers = len(self.encoding_dims) + 1
        
        # Last layer (linear)
        i = num_dec_layers - 1
        if i == 0:
            h_prev = cache['bottleneck']
        else:
            h_prev = cache[f'dec_h{i-1}']
        
        dw = self._clip_gradient(h_prev.T @ dout)
        db = self._clip_gradient(np.sum(dout, axis=0))
        dh = dout @ self.weights[f'dec_w{i}'].T
        
        self.weights[f'dec_w{i}'] -= learning_rate * dw
        self.weights[f'dec_b{i}'] -= learning_rate * db
        
        # Remaining decoder layers
        for i in range(num_dec_layers - 2, -1, -1):
            dz = dh * self._relu_derivative(cache[f'dec_z{i}'])
            dz = self._clip_gradient(dz)
            
            if i == 0:
                h_prev = cache['bottleneck']
            else:
                h_prev = cache[f'dec_h{i-1}']
            
            dw = self._clip_gradient(h_prev.T @ dz)
            db = self._clip_gradient(np.sum(dz, axis=0))
            dh = dz @ self.weights[f'dec_w{i}'].T
            
            self.weights[f'dec_w{i}'] -= learning_rate * dw
            self.weights[f'dec_b{i}'] -= learning_rate * db
        
        # Backprop through encoder
        num_enc_layers = len(self.encoding_dims) + 1
        
        for i in range(num_enc_layers - 1, -1, -1):
            dz = dh * self._relu_derivative(cache[f'enc_z{i}'])
            dz = self._clip_gradient(dz)
            
            if i == 0:
                h_prev = cache['input']
            else:
                h_prev = cache[f'enc_h{i-1}']
            
            dw = self._clip_gradient(h_prev.T @ dz)
            db = self._clip_gradient(np.sum(dz, axis=0))
            dh = dz @ self.weights[f'enc_w{i}'].T
            
            self.weights[f'enc_w{i}'] -= learning_rate * dw
            self.weights[f'enc_b{i}'] -= learning_rate * db
        
        return loss

    def compile(self, learning_rate: float = 0.001):
        """Set learning rate."""
        self.learning_rate = learning_rate
        logger.info(f"Model compiled with learning_rate={learning_rate}")
    
    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.15,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> Dict:
        """
        Train the autoencoder on NORMAL data only.
        
        CRITICAL: Training data must contain ONLY normal behavior!
        """
        logger.info(f"Training on {len(X_train)} samples of NORMAL behavior")
        
        # Split data
        n_val = int(len(X_train) * validation_split)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_val = X_train[val_indices]
        X_tr = X_train[train_indices]
        
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(len(X_tr))
            X_shuffled = X_tr[perm]
            
            # Mini-batch training
            epoch_losses = []
            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i:i+batch_size]
                output, cache = self._forward(batch, training=True)
                loss = self._backward(cache, self.learning_rate)
                epoch_losses.append(loss)
            
            train_loss = np.mean(epoch_losses)
            
            # Validation loss
            val_output, _ = self._forward(X_val)
            val_loss = np.mean((X_val - val_output) ** 2)
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.copy() for k, v in self.weights.items()}
            else:
                patience_counter += 1
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best weights
        if best_weights:
            self.weights = best_weights
        
        # Compute reconstruction statistics on training data
        self._compute_reconstruction_stats(X_train)
        
        self.history = history
        logger.info(f"Training complete. Final loss: {history['loss'][-1]:.6f}")
        
        return history
    
    def _compute_reconstruction_stats(self, X_train: np.ndarray):
        """Compute statistics of reconstruction error on training data."""
        reconstructed, _ = self._forward(X_train)
        errors = np.mean((X_train - reconstructed) ** 2, axis=1)
        
        self.reconstruction_stats = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'threshold_percentiles': {
                '90': float(np.percentile(errors, 90)),
                '95': float(np.percentile(errors, 95)),
                '99': float(np.percentile(errors, 99)),
            }
        }
        
        logger.info(f"Reconstruction stats: mean={self.reconstruction_stats['mean']:.6f}, "
                   f"std={self.reconstruction_stats['std']:.6f}")
    
    def get_reconstruction_error(self, X: np.ndarray, per_feature: bool = False) -> np.ndarray:
        """Compute reconstruction error for input data."""
        reconstructed, _ = self._forward(X)
        squared_error = (X - reconstructed) ** 2
        
        if per_feature:
            return squared_error
        else:
            return np.mean(squared_error, axis=1)
    
    def get_normality_score(self, X: np.ndarray) -> np.ndarray:
        """Convert reconstruction error to normalized normality score."""
        errors = self.get_reconstruction_error(X)
        normalized = (errors - self.reconstruction_stats['mean']) / (self.reconstruction_stats['std'] + 1e-8)
        return normalized
    
    def get_feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """Get per-feature contribution to reconstruction error."""
        per_feature_error = self.get_reconstruction_error(X, per_feature=True)
        total_error = per_feature_error.sum(axis=1, keepdims=True)
        contributions = per_feature_error / (total_error + 1e-8)
        return contributions
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Get the bottleneck representation of input data."""
        _, cache = self._forward(X)
        return cache['bottleneck']
    
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct from bottleneck representation."""
        # Build decoder-only forward pass
        h = Z
        num_dec_layers = len(self.encoding_dims) + 1
        for i in range(num_dec_layers - 1):
            z = h @ self.weights[f'dec_w{i}'] + self.weights[f'dec_b{i}']
            h = self._relu(z)
        i = num_dec_layers - 1
        output = h @ self.weights[f'dec_w{i}'] + self.weights[f'dec_b{i}']
        return output
    
    def save(self, path: str):
        """Save model and statistics."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        np.savez(str(path / 'weights.npz'), **self.weights)
        
        # Save configuration and statistics
        config = {
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'bottleneck_dim': self.bottleneck_dim,
            'dropout_rate': self.dropout_rate,
            'reconstruction_stats': self.reconstruction_stats,
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and statistics."""
        path = Path(path)
        
        # Load configuration
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.reconstruction_stats = config['reconstruction_stats']
        
        # Load weights
        weights_data = np.load(str(path / 'weights.npz'))
        self.weights = {k: weights_data[k] for k in weights_data.files}
        
        logger.info(f"Model loaded from {path}")
    
    def summary(self):
        """Print model summary."""
        print("\n" + "=" * 60)
        print("NOISE FLOOR - Normality Autoencoder (NumPy)")
        print("=" * 60)
        print(f"\nArchitecture: {self.input_dim} → {self.encoding_dims} → "
              f"{self.bottleneck_dim} → {list(reversed(self.encoding_dims))} → {self.input_dim}")
        print(f"\nReconstruction Statistics:")
        print(f"  Mean error: {self.reconstruction_stats['mean']:.6f}")
        print(f"  Std error:  {self.reconstruction_stats['std']:.6f}")
        print(f"\nThreshold Percentiles:")
        for p, v in self.reconstruction_stats.get('threshold_percentiles', {}).items():
            print(f"  {p}th percentile: {v:.6f}")
        print("=" * 60 + "\n")


class EnsembleNormalityModel:
    """Ensemble of autoencoders for more robust normality learning."""
    
    def __init__(self, n_models: int = 3, **autoencoder_kwargs):
        self.n_models = n_models
        self.models = [
            NormalityAutoencoder(**autoencoder_kwargs, name=f'autoencoder_{i}')
            for i in range(n_models)
        ]
        
        for model in self.models:
            model.compile()
    
    def train(self, X_train: np.ndarray, **train_kwargs):
        """Train all models in the ensemble."""
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.n_models}")
            indices = np.random.choice(len(X_train), size=int(len(X_train) * 0.8), replace=False)
            model.train(X_train[indices], **train_kwargs)
    
    def get_normality_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble normality score with uncertainty."""
        scores = np.array([model.get_normality_score(X) for model in self.models])
        return np.mean(scores, axis=0), np.std(scores, axis=0)


if __name__ == "__main__":
    # Test autoencoder
    print("Testing Autoencoder Module")
    print("=" * 50)
    
    from feature_extraction import create_synthetic_normal_data, create_synthetic_drift_data
    
    normal_data = create_synthetic_normal_data(500)
    drift_data = create_synthetic_drift_data(200)
    
    print(f"Normal data shape: {normal_data.shape}")
    print(f"Drift data shape: {drift_data.shape}")
    
    # Create and train autoencoder
    autoencoder = NormalityAutoencoder(input_dim=normal_data.shape[1])
    autoencoder.compile()
    autoencoder.train(normal_data, epochs=50, verbose=0)
    
    # Get normality scores
    normal_scores = autoencoder.get_normality_score(normal_data)
    drift_scores = autoencoder.get_normality_score(drift_data)
    
    print(f"\nNormal data scores: mean={normal_scores.mean():.3f}, std={normal_scores.std():.3f}")
    print(f"Drift data scores: mean={drift_scores.mean():.3f}, std={drift_scores.std():.3f}")
    
    autoencoder.summary()
    
    print("\nAutoencoder module ready!")
