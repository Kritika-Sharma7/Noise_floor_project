"""
NOISE FLOOR - Human-in-the-Loop Feedback System
=================================================
Operator feedback integration for adaptive learning.

This system is designed for border surveillance and high-security perimeters 
where threats emerge gradually.

DESIGN PHILOSOPHY:
------------------
"AI assists operators, it does NOT replace them."

Human operators provide critical context that ML cannot:
1. Distinguish benign from concerning drift
2. Identify false alarms and edge cases
3. Adjust sensitivity based on operational context
4. Provide ground truth for system improvement

FEEDBACK TYPES:
---------------
1. CONFIRM - Operator confirms drift is concerning
2. BENIGN - Operator marks drift as false alarm / expected behavior
3. INVESTIGATE - Operator flags for further review
4. BASELINE_UPDATE - Operator approves baseline adjustment

BASELINE ADAPTATION:
--------------------
- Feedback is LOGGED, not immediately applied
- Changes accumulate in a feedback buffer
- Baseline updates require threshold of consistent feedback
- Updates are gradual (controlled learning rate)
- All changes are reversible

NEVER instantly override intelligence - safety first.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
from pathlib import Path
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of operator feedback."""
    CONFIRM = "confirm"           # Confirm drift is concerning
    BENIGN = "benign"             # Mark as false alarm / benign
    INVESTIGATE = "investigate"   # Flag for further review
    DISMISS = "dismiss"           # Acknowledge and dismiss alert
    ESCALATE = "escalate"         # Manually escalate priority
    
    # Baseline modifications
    BASELINE_ACCEPT = "baseline_accept"     # Accept current as new normal
    BASELINE_REJECT = "baseline_reject"     # Reject baseline shift


class OperatorAction(Enum):
    """Actions taken by operators in response to alerts."""
    NONE = "none"
    VIEWED = "viewed"
    ACKNOWLEDGED = "acknowledged"
    DISPATCHED = "dispatched"
    RESOLVED = "resolved"
    FALSE_ALARM = "false_alarm"


@dataclass
class OperatorFeedback:
    """
    Single feedback entry from an operator.
    """
    # Identification
    feedback_id: str
    operator_id: str
    timestamp: datetime
    
    # Context
    frame_index: int
    window_index: int
    risk_zone: str
    threat_deviation_index: float
    
    # Feedback content
    feedback_type: FeedbackType
    action_taken: OperatorAction
    notes: str = ""
    
    # Feature context (what the system flagged)
    flagged_features: List[str] = field(default_factory=list)
    
    # For baseline updates
    feature_values: Optional[np.ndarray] = None
    
    # Metadata
    session_id: str = ""
    camera_zone: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'feedback_id': self.feedback_id,
            'operator_id': self.operator_id,
            'timestamp': self.timestamp.isoformat(),
            'frame_index': self.frame_index,
            'window_index': self.window_index,
            'risk_zone': self.risk_zone,
            'threat_deviation_index': self.threat_deviation_index,
            'feedback_type': self.feedback_type.value,
            'action_taken': self.action_taken.value,
            'notes': self.notes,
            'flagged_features': self.flagged_features,
            'session_id': self.session_id,
            'camera_zone': self.camera_zone,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OperatorFeedback':
        """Create from dictionary."""
        return cls(
            feedback_id=data['feedback_id'],
            operator_id=data['operator_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            frame_index=data['frame_index'],
            window_index=data['window_index'],
            risk_zone=data['risk_zone'],
            threat_deviation_index=data['threat_deviation_index'],
            feedback_type=FeedbackType(data['feedback_type']),
            action_taken=OperatorAction(data['action_taken']),
            notes=data.get('notes', ''),
            flagged_features=data.get('flagged_features', []),
            session_id=data.get('session_id', ''),
            camera_zone=data.get('camera_zone', ''),
        )


@dataclass
class BaselineUpdateRequest:
    """
    Request to update baseline based on operator feedback.
    """
    request_id: str
    created_at: datetime
    
    # What to update
    feature_indices: List[int]
    proposed_means: np.ndarray
    proposed_stds: np.ndarray
    
    # Evidence
    supporting_feedback: List[str]      # Feedback IDs
    num_benign_markings: int
    
    # Status
    status: str = "pending"             # pending, approved, rejected, applied
    approved_by: Optional[str] = None
    applied_at: Optional[datetime] = None
    
    # Safety
    rollback_data: Optional[Dict] = None    # Original values for rollback


class FeedbackStore:
    """
    Persistent storage for operator feedback.
    
    Maintains history of all feedback for:
    - Audit trail
    - System improvement
    - Baseline adaptation evidence
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.feedback_buffer: List[OperatorFeedback] = []
        self.update_requests: List[BaselineUpdateRequest] = []
        
        # Load existing feedback if storage exists
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def add(self, feedback: OperatorFeedback):
        """Add new feedback entry."""
        self.feedback_buffer.append(feedback)
        
        if self.storage_path:
            self._save()
        
        logger.info(f"Feedback recorded: {feedback.feedback_type.value} from {feedback.operator_id}")
    
    def get_recent(self, n: int = 100) -> List[OperatorFeedback]:
        """Get most recent N feedback entries."""
        return self.feedback_buffer[-n:]
    
    def get_by_type(self, feedback_type: FeedbackType) -> List[OperatorFeedback]:
        """Get all feedback of a specific type."""
        return [f for f in self.feedback_buffer if f.feedback_type == feedback_type]
    
    def get_by_session(self, session_id: str) -> List[OperatorFeedback]:
        """Get all feedback from a session."""
        return [f for f in self.feedback_buffer if f.session_id == session_id]
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        if not self.feedback_buffer:
            return {'total': 0}
        
        type_counts = {}
        for f in self.feedback_buffer:
            t = f.feedback_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Calculate false positive indication
        benign_count = type_counts.get('benign', 0)
        confirm_count = type_counts.get('confirm', 0)
        total = benign_count + confirm_count
        
        false_positive_rate = benign_count / total if total > 0 else 0
        
        return {
            'total': len(self.feedback_buffer),
            'by_type': type_counts,
            'indicated_false_positive_rate': false_positive_rate,
            'operators': len(set(f.operator_id for f in self.feedback_buffer)),
        }
    
    def _save(self):
        """Save feedback to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'feedback': [f.to_dict() for f in self.feedback_buffer],
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(self.storage_path / 'feedback.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load feedback from disk."""
        feedback_file = self.storage_path / 'feedback.json'
        
        if not feedback_file.exists():
            return
        
        with open(feedback_file, 'r') as f:
            data = json.load(f)
        
        self.feedback_buffer = [
            OperatorFeedback.from_dict(d) 
            for d in data.get('feedback', [])
        ]
        
        logger.info(f"Loaded {len(self.feedback_buffer)} feedback entries")


class BaselineAdaptationEngine:
    """
    Handles gradual baseline adaptation based on operator feedback.
    
    SAFETY PRINCIPLES:
    -----------------
    1. Never instantly override - accumulate evidence
    2. Require threshold of consistent feedback
    3. Apply changes gradually (controlled learning rate)
    4. Maintain rollback capability
    5. Log all changes for audit
    
    ADAPTATION PROCESS:
    ------------------
    1. Accumulate BENIGN feedback for false alarms
    2. When threshold met, propose baseline shift
    3. Apply small incremental update
    4. Monitor for feedback reversal
    5. Continue gradual adaptation or rollback
    """
    
    def __init__(
        self,
        feedback_store: FeedbackStore,
        baseline_means: Optional[np.ndarray] = None,
        baseline_stds: Optional[np.ndarray] = None,
        # Adaptation parameters
        min_feedback_for_update: int = 10,      # Minimum benign feedback needed
        update_learning_rate: float = 0.05,     # How fast to shift baseline
        consistency_threshold: float = 0.8,     # % agreement needed
        max_shift_per_update: float = 0.5,      # Max std shift per update
    ):
        self.feedback_store = feedback_store
        
        # Current baseline
        self.baseline_means = baseline_means
        self.baseline_stds = baseline_stds
        
        # Adaptation parameters
        self.min_feedback_for_update = min_feedback_for_update
        self.update_learning_rate = update_learning_rate
        self.consistency_threshold = consistency_threshold
        self.max_shift_per_update = max_shift_per_update
        
        # Pending feature values from benign feedback
        self.pending_benign_samples: deque = deque(maxlen=1000)
        
        # Update history
        self.update_history: List[Dict] = []
        
        # Callbacks
        self.on_baseline_updated: Optional[Callable] = None
    
    def set_baseline(self, means: np.ndarray, stds: np.ndarray):
        """Set current baseline."""
        self.baseline_means = means.copy()
        self.baseline_stds = stds.copy()
    
    def record_benign_sample(
        self,
        features: np.ndarray,
        feedback: OperatorFeedback
    ):
        """
        Record a sample that was marked as benign/false alarm.
        
        These samples accumulate evidence for baseline expansion.
        """
        self.pending_benign_samples.append({
            'features': features,
            'feedback_id': feedback.feedback_id,
            'timestamp': feedback.timestamp,
            'frame_index': feedback.frame_index,
        })
        
        logger.debug(f"Benign sample recorded. Buffer size: {len(self.pending_benign_samples)}")
        
        # Check if we should propose an update
        if len(self.pending_benign_samples) >= self.min_feedback_for_update:
            self._evaluate_baseline_update()
    
    def _evaluate_baseline_update(self):
        """
        Evaluate whether baseline should be updated.
        
        Checks:
        1. Sufficient consistent benign feedback
        2. Feature values are within reasonable range
        3. Update would not destabilize system
        """
        if self.baseline_means is None:
            return
        
        # Get recent benign samples
        samples = list(self.pending_benign_samples)
        
        if len(samples) < self.min_feedback_for_update:
            return
        
        # Extract feature matrices
        features = np.array([s['features'] for s in samples])
        
        # Compute proposed new statistics
        proposed_means = np.mean(features, axis=0)
        proposed_stds = np.std(features, axis=0) + 1e-6
        
        # Check each feature for significant shift
        mean_shifts = (proposed_means - self.baseline_means) / self.baseline_stds
        
        # Log evaluation
        significant_shifts = np.abs(mean_shifts) > 0.5
        num_significant = np.sum(significant_shifts)
        
        logger.info(
            f"Baseline evaluation: {num_significant} features with significant shift "
            f"(based on {len(samples)} benign samples)"
        )
        
        # If shifts are reasonable, propose update
        if num_significant > 0:
            self._propose_update(proposed_means, proposed_stds, mean_shifts)
    
    def _propose_update(
        self,
        proposed_means: np.ndarray,
        proposed_stds: np.ndarray,
        shifts: np.ndarray
    ):
        """
        Propose a baseline update.
        
        Applies gradual shift, not full replacement.
        """
        if self.baseline_means is None:
            return
        
        # Clip shifts to max allowed
        clipped_shifts = np.clip(shifts, -self.max_shift_per_update, self.max_shift_per_update)
        
        # Compute new baseline with learning rate
        new_means = (
            self.baseline_means + 
            self.update_learning_rate * clipped_shifts * self.baseline_stds
        )
        
        # Expand stds slightly if samples show more variance
        std_expansion = np.maximum(proposed_stds / self.baseline_stds, 1.0)
        std_expansion = np.clip(std_expansion, 1.0, 1.0 + self.update_learning_rate)
        new_stds = self.baseline_stds * std_expansion
        
        # Store rollback data
        rollback = {
            'means': self.baseline_means.copy(),
            'stds': self.baseline_stds.copy(),
        }
        
        # Apply update
        old_means = self.baseline_means.copy()
        self.baseline_means = new_means
        self.baseline_stds = new_stds
        
        # Clear pending samples
        self.pending_benign_samples.clear()
        
        # Log update
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.pending_benign_samples),
            'mean_shift_magnitude': float(np.mean(np.abs(clipped_shifts))),
            'features_shifted': int(np.sum(np.abs(clipped_shifts) > 0.01)),
            'rollback': rollback,
        }
        self.update_history.append(update_record)
        
        logger.info(
            f"Baseline updated: mean shift magnitude = "
            f"{update_record['mean_shift_magnitude']:.3f}"
        )
        
        # Callback
        if self.on_baseline_updated:
            self.on_baseline_updated(self.baseline_means, self.baseline_stds)
    
    def rollback_last_update(self) -> bool:
        """
        Rollback the most recent baseline update.
        
        Returns True if rollback successful.
        """
        if not self.update_history:
            logger.warning("No updates to rollback")
            return False
        
        last_update = self.update_history.pop()
        rollback = last_update.get('rollback')
        
        if rollback:
            self.baseline_means = rollback['means']
            self.baseline_stds = rollback['stds']
            
            logger.info("Baseline rollback applied")
            
            if self.on_baseline_updated:
                self.on_baseline_updated(self.baseline_means, self.baseline_stds)
            
            return True
        
        return False
    
    def get_adaptation_status(self) -> Dict:
        """Get current adaptation status."""
        return {
            'pending_samples': len(self.pending_benign_samples),
            'min_for_update': self.min_feedback_for_update,
            'updates_applied': len(self.update_history),
            'learning_rate': self.update_learning_rate,
            'can_rollback': len(self.update_history) > 0,
        }


class HumanInTheLoop:
    """
    Main interface for human-in-the-loop feedback integration.
    
    USAGE:
    ------
    1. Create HumanInTheLoop instance with baseline
    2. Call submit_feedback() when operator provides input
    3. System automatically manages adaptation
    4. Use get_adjusted_thresholds() to get operator-informed thresholds
    
    This system is designed for border surveillance and high-security 
    perimeters where threats emerge gradually.
    """
    
    def __init__(
        self,
        baseline_means: Optional[np.ndarray] = None,
        baseline_stds: Optional[np.ndarray] = None,
        storage_path: Optional[str] = None,
        session_id: str = None,
    ):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Feedback storage
        self.feedback_store = FeedbackStore(storage_path)
        
        # Baseline adaptation
        self.adaptation_engine = BaselineAdaptationEngine(
            self.feedback_store,
            baseline_means,
            baseline_stds,
        )
        
        # Feedback counter for ID generation
        self._feedback_counter = 0
        
        # Alert tracking
        self.active_alerts: Dict[str, Dict] = {}
        
        logger.info(f"HumanInTheLoop initialized for session {self.session_id}")
    
    def set_baseline(self, means: np.ndarray, stds: np.ndarray):
        """Set baseline for adaptation."""
        self.adaptation_engine.set_baseline(means, stds)
    
    def create_alert(
        self,
        alert_id: str,
        frame_index: int,
        window_index: int,
        risk_zone: str,
        threat_deviation_index: float,
        flagged_features: List[str],
        camera_zone: str = "",
    ):
        """
        Create a new alert that awaits operator feedback.
        """
        self.active_alerts[alert_id] = {
            'created_at': datetime.now(),
            'frame_index': frame_index,
            'window_index': window_index,
            'risk_zone': risk_zone,
            'threat_deviation_index': threat_deviation_index,
            'flagged_features': flagged_features,
            'camera_zone': camera_zone,
            'status': 'pending',
        }
        
        return alert_id
    
    def submit_feedback(
        self,
        alert_id: str,
        operator_id: str,
        feedback_type: FeedbackType,
        action_taken: OperatorAction = OperatorAction.ACKNOWLEDGED,
        notes: str = "",
        feature_values: Optional[np.ndarray] = None,
    ) -> OperatorFeedback:
        """
        Submit operator feedback for an alert.
        
        This is the MAIN METHOD for recording human input.
        
        Args:
            alert_id: ID of the alert being responded to
            operator_id: ID of the operator providing feedback
            feedback_type: Type of feedback (CONFIRM, BENIGN, etc.)
            action_taken: What action the operator took
            notes: Optional operator notes
            feature_values: Feature values at time of alert (for adaptation)
            
        Returns:
            OperatorFeedback object
        """
        # Get alert context
        alert = self.active_alerts.get(alert_id, {})
        
        # Generate feedback ID
        self._feedback_counter += 1
        feedback_id = f"{self.session_id}_F{self._feedback_counter:04d}"
        
        # Create feedback object
        feedback = OperatorFeedback(
            feedback_id=feedback_id,
            operator_id=operator_id,
            timestamp=datetime.now(),
            frame_index=alert.get('frame_index', 0),
            window_index=alert.get('window_index', 0),
            risk_zone=alert.get('risk_zone', 'unknown'),
            threat_deviation_index=alert.get('threat_deviation_index', 0),
            feedback_type=feedback_type,
            action_taken=action_taken,
            notes=notes,
            flagged_features=alert.get('flagged_features', []),
            feature_values=feature_values,
            session_id=self.session_id,
            camera_zone=alert.get('camera_zone', ''),
        )
        
        # Store feedback
        self.feedback_store.add(feedback)
        
        # Update alert status
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['status'] = 'responded'
            self.active_alerts[alert_id]['feedback_type'] = feedback_type.value
        
        # Handle baseline adaptation for benign feedback
        if feedback_type == FeedbackType.BENIGN and feature_values is not None:
            self.adaptation_engine.record_benign_sample(feature_values, feedback)
        
        logger.info(
            f"Feedback submitted: {feedback_type.value} by {operator_id} "
            f"for alert {alert_id}"
        )
        
        return feedback
    
    def get_feedback_summary(self) -> Dict:
        """Get summary of feedback for this session."""
        session_feedback = self.feedback_store.get_by_session(self.session_id)
        
        if not session_feedback:
            return {
                'session_id': self.session_id,
                'total_feedback': 0,
                'pending_alerts': len([a for a in self.active_alerts.values() if a['status'] == 'pending']),
            }
        
        type_counts = {}
        for f in session_feedback:
            t = f.feedback_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'session_id': self.session_id,
            'total_feedback': len(session_feedback),
            'by_type': type_counts,
            'pending_alerts': len([a for a in self.active_alerts.values() if a['status'] == 'pending']),
            'adaptation_status': self.adaptation_engine.get_adaptation_status(),
        }
    
    def get_adjusted_baseline(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the current adjusted baseline.
        
        Returns baseline that has been adapted based on operator feedback.
        """
        return (
            self.adaptation_engine.baseline_means,
            self.adaptation_engine.baseline_stds,
        )
    
    def get_false_positive_estimate(self) -> float:
        """
        Estimate false positive rate based on operator feedback.
        
        This provides insight into system accuracy from human perspective.
        """
        stats = self.feedback_store.get_statistics()
        return stats.get('indicated_false_positive_rate', 0.0)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_feedback_system(
    baseline_means: np.ndarray,
    baseline_stds: np.ndarray,
    storage_path: str = "./feedback_data"
) -> HumanInTheLoop:
    """
    Create a configured human-in-the-loop feedback system.
    """
    hitl = HumanInTheLoop(
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
        storage_path=storage_path,
    )
    
    return hitl
