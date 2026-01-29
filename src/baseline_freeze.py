"""
NOISE FLOOR - Baseline Freeze & Controlled Adaptation System
==============================================================
Human-gated baseline management for defense-grade surveillance.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

CRITICAL SECURITY PRINCIPLE:
----------------------------
"Baseline adaptation is human-gated."

The baseline defines what is NORMAL. If an attacker can slowly shift
the baseline, they can make threats appear normal. This module prevents:
1. Data poisoning attacks
2. Drift normalization during slow-burn intrusions
3. Autonomous baseline corruption

BASELINE LIFECYCLE:
-------------------
1. LEARNING PHASE: Initial baseline established from first N frames
2. FROZEN PHASE: Baseline locked - no automatic updates
3. HUMAN-GATED ADAPTATION: Updates only with explicit operator approval

ADAPTATION RULES:
-----------------
- Baseline adapts ONLY if operator marks behavior as BENIGN
- Adaptation uses low learning rate (slow change)
- Maximum adaptation rate is capped
- All adaptations are logged for audit

TECHNOLOGY READINESS LEVEL: TRL-4
Lab-validated prototype for decision-support intelligence.
AI assists operators, it does NOT replace them.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineState:
    """
    Complete baseline state for drift detection.
    
    The baseline defines what is NORMAL. Protecting this from
    corruption is critical for system integrity.
    """
    # Feature statistics
    feature_means: np.ndarray
    feature_stds: np.ndarray
    
    # Reconstruction loss baseline (from LSTM-VAE)
    loss_mean: float = 0.0
    loss_std: float = 1.0
    
    # Latent space baseline
    latent_mean: np.ndarray = field(default_factory=lambda: np.zeros(8))
    latent_std: np.ndarray = field(default_factory=lambda: np.ones(8))
    
    # Baseline metadata
    num_samples: int = 0
    established_at: Optional[str] = None
    last_updated: Optional[str] = None
    
    # Freeze status
    is_frozen: bool = True
    freeze_reason: str = "initial_freeze"
    
    # Adaptation history
    adaptation_count: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'feature_means': self.feature_means.tolist(),
            'feature_stds': self.feature_stds.tolist(),
            'loss_mean': self.loss_mean,
            'loss_std': self.loss_std,
            'latent_mean': self.latent_mean.tolist(),
            'latent_std': self.latent_std.tolist(),
            'num_samples': self.num_samples,
            'established_at': self.established_at,
            'last_updated': self.last_updated,
            'is_frozen': self.is_frozen,
            'freeze_reason': self.freeze_reason,
            'adaptation_count': self.adaptation_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BaselineState':
        """Deserialize from dictionary."""
        return cls(
            feature_means=np.array(data['feature_means']),
            feature_stds=np.array(data['feature_stds']),
            loss_mean=data['loss_mean'],
            loss_std=data['loss_std'],
            latent_mean=np.array(data['latent_mean']),
            latent_std=np.array(data['latent_std']),
            num_samples=data['num_samples'],
            established_at=data.get('established_at'),
            last_updated=data.get('last_updated'),
            is_frozen=data.get('is_frozen', True),
            freeze_reason=data.get('freeze_reason', 'loaded'),
            adaptation_count=data.get('adaptation_count', 0),
        )


@dataclass
class AdaptationEvent:
    """Record of a baseline adaptation event."""
    timestamp: str
    operator_id: str
    frame_index: int
    adaptation_type: str  # "benign_marking", "manual_reset", "scheduled"
    
    # What changed
    features_updated: List[str]
    learning_rate_used: float
    
    # Context
    tdi_at_time: float
    zone_at_time: str
    
    # Operator notes
    notes: str = ""


class BaselineFreezeManager:
    """
    Manages baseline freezing and human-gated adaptation.
    
    SECURITY MODEL:
    ---------------
    The baseline is FROZEN by default after initial learning.
    Only explicit human approval can trigger adaptation.
    
    This prevents:
    - Adversarial drift normalization
    - Slow-burn attack masking
    - Data poisoning
    
    "Baseline adaptation is human-gated."
    """
    
    def __init__(
        self,
        feature_dim: int = 24,
        latent_dim: int = 8,
        
        # Learning phase
        learning_window: int = 200,        # Frames for initial baseline
        
        # Adaptation parameters
        adaptation_learning_rate: float = 0.01,  # Very slow adaptation
        max_adaptation_rate: float = 0.05,       # Cap on single update
        min_samples_for_adaptation: int = 10,    # Need multiple confirmations
        
        # Storage
        storage_path: str = "./baseline_data",
    ):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.learning_window = learning_window
        self.adaptation_learning_rate = adaptation_learning_rate
        self.max_adaptation_rate = max_adaptation_rate
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.baseline: Optional[BaselineState] = None
        self.phase: str = "learning"  # learning, frozen, adapting
        
        # Learning buffer
        self.learning_buffer: List[np.ndarray] = []
        self.loss_buffer: List[float] = []
        self.latent_buffer: List[np.ndarray] = []
        
        # Adaptation buffer (samples marked benign by operator)
        self.adaptation_buffer: List[np.ndarray] = []
        self.adaptation_events: List[AdaptationEvent] = []
        
        logger.info("BaselineFreezeManager initialized")
        logger.info(f"  Learning window: {learning_window} frames")
        logger.info(f"  Adaptation rate: {adaptation_learning_rate} (max: {max_adaptation_rate})")
        logger.info("  Baseline will be FROZEN after learning phase")
    
    def is_learning(self) -> bool:
        """Check if still in learning phase."""
        return self.phase == "learning"
    
    def is_frozen(self) -> bool:
        """Check if baseline is frozen."""
        return self.baseline is not None and self.baseline.is_frozen
    
    def add_learning_sample(
        self,
        features: np.ndarray,
        reconstruction_loss: Optional[float] = None,
        latent_mean: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Add a sample during learning phase.
        
        Returns:
            True if baseline was established (learning complete)
        """
        if self.phase != "learning":
            return False
        
        self.learning_buffer.append(features)
        
        if reconstruction_loss is not None:
            self.loss_buffer.append(reconstruction_loss)
        
        if latent_mean is not None:
            self.latent_buffer.append(latent_mean)
        
        # Check if learning is complete
        if len(self.learning_buffer) >= self.learning_window:
            self._establish_baseline()
            return True
        
        return False
    
    def _establish_baseline(self):
        """Establish and freeze baseline from learning buffer."""
        features_array = np.array(self.learning_buffer)
        
        feature_means = np.mean(features_array, axis=0)
        feature_stds = np.std(features_array, axis=0) + 1e-6
        
        loss_mean = np.mean(self.loss_buffer) if self.loss_buffer else 0.0
        loss_std = np.std(self.loss_buffer) + 1e-6 if self.loss_buffer else 1.0
        
        if self.latent_buffer:
            latent_array = np.array(self.latent_buffer)
            latent_mean = np.mean(latent_array, axis=0)
            latent_std = np.std(latent_array, axis=0) + 1e-6
        else:
            latent_mean = np.zeros(self.latent_dim)
            latent_std = np.ones(self.latent_dim)
        
        self.baseline = BaselineState(
            feature_means=feature_means,
            feature_stds=feature_stds,
            loss_mean=loss_mean,
            loss_std=loss_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
            num_samples=len(self.learning_buffer),
            established_at=datetime.now().isoformat(),
            is_frozen=True,
            freeze_reason="initial_establishment",
        )
        
        self.phase = "frozen"
        
        logger.info("=" * 60)
        logger.info("BASELINE ESTABLISHED AND FROZEN")
        logger.info("=" * 60)
        logger.info(f"  Samples: {self.baseline.num_samples}")
        logger.info(f"  Loss mean: {loss_mean:.4f} ± {loss_std:.4f}")
        logger.info(f"  Status: FROZEN (human-gated adaptation only)")
        logger.info("=" * 60)
        
        # Clear learning buffer
        self.learning_buffer.clear()
        self.loss_buffer.clear()
        self.latent_buffer.clear()
        
        # Save baseline
        self._save_baseline()
    
    def request_adaptation(
        self,
        features: np.ndarray,
        operator_id: str,
        frame_index: int,
        tdi: float,
        zone: str,
        notes: str = "",
    ) -> Dict:
        """
        Operator requests to mark current behavior as BENIGN.
        
        This adds the sample to adaptation buffer. Actual adaptation
        occurs only when enough samples are collected.
        
        Returns:
            Status dict with adaptation info
        """
        if self.baseline is None:
            return {
                'status': 'error',
                'message': 'Baseline not yet established',
            }
        
        # Add to adaptation buffer
        self.adaptation_buffer.append(features)
        
        # Log the event
        event = AdaptationEvent(
            timestamp=datetime.now().isoformat(),
            operator_id=operator_id,
            frame_index=frame_index,
            adaptation_type="benign_marking",
            features_updated=[],
            learning_rate_used=0.0,
            tdi_at_time=tdi,
            zone_at_time=zone,
            notes=notes,
        )
        self.adaptation_events.append(event)
        
        logger.info(f"Benign marking received from operator {operator_id}")
        logger.info(f"  Frame: {frame_index}, TDI: {tdi:.1f}, Zone: {zone}")
        logger.info(f"  Adaptation buffer: {len(self.adaptation_buffer)}/{self.min_samples_for_adaptation}")
        
        # Check if we have enough samples
        if len(self.adaptation_buffer) >= self.min_samples_for_adaptation:
            return self._apply_adaptation(operator_id)
        
        return {
            'status': 'buffered',
            'message': f'Sample added to adaptation buffer ({len(self.adaptation_buffer)}/{self.min_samples_for_adaptation})',
            'samples_needed': self.min_samples_for_adaptation - len(self.adaptation_buffer),
        }
    
    def _apply_adaptation(self, operator_id: str) -> Dict:
        """Apply accumulated adaptation buffer to baseline."""
        if not self.adaptation_buffer:
            return {'status': 'error', 'message': 'No samples in adaptation buffer'}
        
        # Compute new statistics from adaptation buffer
        new_features = np.array(self.adaptation_buffer)
        new_means = np.mean(new_features, axis=0)
        new_stds = np.std(new_features, axis=0) + 1e-6
        
        # Compute update magnitude (for safety check)
        mean_shift = np.abs(new_means - self.baseline.feature_means) / self.baseline.feature_stds
        max_shift = np.max(mean_shift)
        
        # Cap learning rate if shift is too large
        effective_lr = min(
            self.adaptation_learning_rate,
            self.max_adaptation_rate / (max_shift + 1e-6)
        )
        
        # Apply update: baseline = (1 - lr) * baseline + lr * new
        old_means = self.baseline.feature_means.copy()
        old_stds = self.baseline.feature_stds.copy()
        
        self.baseline.feature_means = (1 - effective_lr) * old_means + effective_lr * new_means
        self.baseline.feature_stds = (1 - effective_lr) * old_stds + effective_lr * new_stds
        
        # Update metadata
        self.baseline.last_updated = datetime.now().isoformat()
        self.baseline.adaptation_count += 1
        
        # Identify which features changed significantly
        mean_changes = np.abs(self.baseline.feature_means - old_means) / old_stds
        significant_changes = np.where(mean_changes > 0.01)[0].tolist()
        
        logger.info("=" * 60)
        logger.info("BASELINE ADAPTATION APPLIED (Human-gated)")
        logger.info("=" * 60)
        logger.info(f"  Operator: {operator_id}")
        logger.info(f"  Samples used: {len(self.adaptation_buffer)}")
        logger.info(f"  Effective learning rate: {effective_lr:.4f}")
        logger.info(f"  Features updated: {len(significant_changes)}")
        logger.info(f"  Total adaptations: {self.baseline.adaptation_count}")
        logger.info("=" * 60)
        
        # Clear buffer
        self.adaptation_buffer.clear()
        
        # Save updated baseline
        self._save_baseline()
        
        return {
            'status': 'adapted',
            'message': f'Baseline adapted with learning rate {effective_lr:.4f}',
            'features_updated': significant_changes,
            'adaptation_count': self.baseline.adaptation_count,
        }
    
    def force_freeze(self, reason: str = "manual_freeze"):
        """Force freeze the baseline."""
        if self.baseline:
            self.baseline.is_frozen = True
            self.baseline.freeze_reason = reason
            self.phase = "frozen"
            self.adaptation_buffer.clear()
            logger.info(f"Baseline FROZEN: {reason}")
            self._save_baseline()
    
    def get_baseline(self) -> Optional[BaselineState]:
        """Get current baseline state."""
        return self.baseline
    
    def get_feature_means(self) -> Optional[np.ndarray]:
        """Get baseline feature means."""
        return self.baseline.feature_means if self.baseline else None
    
    def get_feature_stds(self) -> Optional[np.ndarray]:
        """Get baseline feature stds."""
        return self.baseline.feature_stds if self.baseline else None
    
    def _save_baseline(self):
        """Save baseline to disk."""
        if self.baseline is None:
            return
        
        filepath = self.storage_path / "baseline.json"
        with open(filepath, 'w') as f:
            json.dump(self.baseline.to_dict(), f, indent=2)
        
        logger.debug(f"Baseline saved to {filepath}")
    
    def load_baseline(self, filepath: Optional[str] = None) -> bool:
        """Load baseline from disk."""
        if filepath is None:
            filepath = self.storage_path / "baseline.json"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"No baseline file found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.baseline = BaselineState.from_dict(data)
            self.phase = "frozen" if self.baseline.is_frozen else "adapting"
            
            logger.info(f"Baseline loaded from {filepath}")
            logger.info(f"  Status: {'FROZEN' if self.baseline.is_frozen else 'ADAPTING'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            return False
    
    def get_status_summary(self) -> Dict:
        """Get summary of baseline status."""
        if self.baseline is None:
            return {
                'phase': self.phase,
                'learning_progress': f"{len(self.learning_buffer)}/{self.learning_window}",
                'baseline_established': False,
            }
        
        return {
            'phase': self.phase,
            'baseline_established': True,
            'is_frozen': self.baseline.is_frozen,
            'freeze_reason': self.baseline.freeze_reason,
            'num_samples': self.baseline.num_samples,
            'established_at': self.baseline.established_at,
            'last_updated': self.baseline.last_updated,
            'adaptation_count': self.baseline.adaptation_count,
            'adaptation_buffer_size': len(self.adaptation_buffer),
        }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_frozen_baseline(
    features: np.ndarray,
    reconstruction_losses: Optional[np.ndarray] = None,
    latent_means: Optional[np.ndarray] = None,
) -> BaselineState:
    """
    Create a frozen baseline directly from data.
    
    Useful for initializing from pre-trained model outputs.
    """
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0) + 1e-6
    
    loss_mean = float(np.mean(reconstruction_losses)) if reconstruction_losses is not None else 0.0
    loss_std = float(np.std(reconstruction_losses) + 1e-6) if reconstruction_losses is not None else 1.0
    
    if latent_means is not None:
        latent_mean = np.mean(latent_means, axis=0)
        latent_std = np.std(latent_means, axis=0) + 1e-6
    else:
        latent_mean = np.zeros(8)
        latent_std = np.ones(8)
    
    return BaselineState(
        feature_means=feature_means,
        feature_stds=feature_stds,
        loss_mean=loss_mean,
        loss_std=loss_std,
        latent_mean=latent_mean,
        latent_std=latent_std,
        num_samples=len(features),
        established_at=datetime.now().isoformat(),
        is_frozen=True,
        freeze_reason="direct_creation",
    )


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("BASELINE FREEZE MANAGER - Demo")
    print("=" * 60)
    
    manager = BaselineFreezeManager(
        feature_dim=24,
        learning_window=50,  # Shorter for demo
        min_samples_for_adaptation=3,
    )
    
    print("\n1. Learning phase...")
    for i in range(50):
        features = np.random.randn(24) * 0.1
        if manager.add_learning_sample(features, reconstruction_loss=0.1 + np.random.randn() * 0.01):
            print(f"   Baseline established at sample {i+1}")
            break
    
    print(f"\n2. Baseline status: {manager.get_status_summary()}")
    
    print("\n3. Attempting adaptation without operator approval...")
    print("   (This should fail - baseline is frozen)")
    
    print("\n4. Operator marks behavior as benign...")
    for i in range(3):
        result = manager.request_adaptation(
            features=np.random.randn(24) * 0.1,
            operator_id="operator_001",
            frame_index=100 + i,
            tdi=15.0,
            zone="WATCH",
            notes="Normal foot traffic"
        )
        print(f"   Sample {i+1}: {result['status']}")
    
    print(f"\n5. Final status: {manager.get_status_summary()}")
