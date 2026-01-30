"""
NOISE FLOOR - Security & Audit Module
=======================================
Defense-grade security, integrity, and audit trail system.

This module provides:
1. Data Integrity Checks - Hash verification for baseline data
2. Audit Trail - Immutable log of all system actions
3. Adversarial Detection - Detect input manipulation
4. Graceful Degradation - Handle system failures

DESIGN PHILOSOPHY:
------------------
"In defense systems, trust but verify. Log everything."
"""

import numpy as np
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import deque
from enum import Enum
import logging
import os
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# AUDIT TRAIL
# =============================================================================

class AuditEventType(Enum):
    """Types of auditable events."""
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    
    # Baseline events
    BASELINE_CREATED = "baseline_created"
    BASELINE_FROZEN = "baseline_frozen"
    BASELINE_UPDATE_REQUESTED = "baseline_update_requested"
    BASELINE_UPDATE_APPROVED = "baseline_update_approved"
    BASELINE_UPDATE_REJECTED = "baseline_update_rejected"
    
    # Detection events
    ZONE_TRANSITION = "zone_transition"
    ALERT_GENERATED = "alert_generated"
    ALERT_ESCALATED = "alert_escalated"
    
    # Operator events
    OPERATOR_LOGIN = "operator_login"
    OPERATOR_LOGOUT = "operator_logout"
    OPERATOR_FEEDBACK = "operator_feedback"
    INCIDENT_ACKNOWLEDGED = "incident_acknowledged"
    INCIDENT_RESOLVED = "incident_resolved"
    
    # Security events
    INTEGRITY_CHECK_PASSED = "integrity_check_passed"
    INTEGRITY_CHECK_FAILED = "integrity_check_failed"
    ADVERSARIAL_DETECTED = "adversarial_detected"
    TAMPERING_SUSPECTED = "tampering_suspected"
    
    # Data events
    DATA_EXPORT = "data_export"
    REPORT_GENERATED = "report_generated"


@dataclass
class AuditEntry:
    """Single audit log entry."""
    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    actor: str                         # Who performed the action
    target: str                        # What was affected
    details: Dict[str, Any]
    session_id: str
    ip_address: str = "localhost"
    
    # Integrity
    previous_hash: str = ""
    entry_hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute hash for this entry."""
        data = {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor,
            'target': self.target,
            'details': json.dumps(self.details, sort_keys=True),
            'session_id': self.session_id,
            'previous_hash': self.previous_hash,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor,
            'target': self.target,
            'details': self.details,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'previous_hash': self.previous_hash,
            'entry_hash': self.entry_hash,
        }


class AuditTrail:
    """
    Immutable audit trail for all system actions.
    
    Uses hash chaining to ensure integrity.
    """
    
    def __init__(
        self,
        storage_path: str = "./audit_logs",
        max_memory_entries: int = 10000,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.entries: List[AuditEntry] = []
        self.max_memory_entries = max_memory_entries
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entry_counter = 0
        
        # Genesis hash
        self.genesis_hash = hashlib.sha256(b"NOISE_FLOOR_GENESIS").hexdigest()
        self.last_hash = self.genesis_hash
        
        logger.info(f"AuditTrail initialized, session: {self.session_id}")
    
    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        target: str,
        details: Dict[str, Any] = None,
    ) -> AuditEntry:
        """Log an audit event."""
        self.entry_counter += 1
        entry_id = f"{self.session_id}_{self.entry_counter:06d}"
        
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            event_type=event_type,
            actor=actor,
            target=target,
            details=details or {},
            session_id=self.session_id,
            previous_hash=self.last_hash,
        )
        
        # Compute and set hash
        entry.entry_hash = entry.compute_hash()
        self.last_hash = entry.entry_hash
        
        self.entries.append(entry)
        
        # Persist if needed
        if len(self.entries) > self.max_memory_entries:
            self._persist_entries()
        
        return entry
    
    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify the integrity of the audit chain."""
        if not self.entries:
            return True, None
        
        expected_prev = self.genesis_hash
        
        for entry in self.entries:
            # Verify previous hash linkage
            if entry.previous_hash != expected_prev:
                return False, f"Chain broken at {entry.entry_id}: previous hash mismatch"
            
            # Verify entry hash
            computed = entry.compute_hash()
            if computed != entry.entry_hash:
                return False, f"Entry {entry.entry_id} hash mismatch"
            
            expected_prev = entry.entry_hash
        
        return True, None
    
    def get_entries_by_type(
        self,
        event_type: AuditEventType,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get entries of specific type."""
        filtered = [e for e in self.entries if e.event_type == event_type]
        return filtered[-limit:]
    
    def get_entries_by_actor(self, actor: str, limit: int = 100) -> List[AuditEntry]:
        """Get entries by actor."""
        filtered = [e for e in self.entries if e.actor == actor]
        return filtered[-limit:]
    
    def get_security_events(self, limit: int = 100) -> List[AuditEntry]:
        """Get security-related events."""
        security_types = [
            AuditEventType.INTEGRITY_CHECK_FAILED,
            AuditEventType.ADVERSARIAL_DETECTED,
            AuditEventType.TAMPERING_SUSPECTED,
        ]
        filtered = [e for e in self.entries if e.event_type in security_types]
        return filtered[-limit:]
    
    def _persist_entries(self) -> None:
        """Persist entries to disk."""
        # Keep recent entries in memory
        to_persist = self.entries[:len(self.entries) // 2]
        self.entries = self.entries[len(self.entries) // 2:]
        
        # Save to file
        filename = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_path / filename
        
        with open(filepath, 'w') as f:
            json.dump([e.to_dict() for e in to_persist], f, indent=2)
        
        logger.info(f"Persisted {len(to_persist)} audit entries to {filepath}")
    
    def export_full_log(self, filepath: str = None) -> str:
        """Export complete audit log."""
        if filepath is None:
            filepath = self.storage_path / f"full_audit_{self.session_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'genesis_hash': self.genesis_hash,
                'entries': [e.to_dict() for e in self.entries],
                'chain_valid': self.verify_chain()[0],
            }, f, indent=2)
        
        return str(filepath)


# =============================================================================
# DATA INTEGRITY
# =============================================================================

@dataclass
class IntegrityReport:
    """Report on data integrity check."""
    checked_at: datetime
    target: str
    expected_hash: str
    actual_hash: str
    is_valid: bool
    details: str = ""


class DataIntegrityManager:
    """
    Manages data integrity verification.
    
    Uses HMAC for baseline data protection.
    """
    
    def __init__(
        self,
        secret_key: str = None,
        storage_path: str = "./integrity",
    ):
        # Generate or use provided secret key
        if secret_key:
            self.secret_key = secret_key.encode()
        else:
            self.secret_key = self._generate_key()
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.hash_registry: Dict[str, str] = {}
        self._load_registry()
        
        logger.info("DataIntegrityManager initialized")
    
    def _generate_key(self) -> bytes:
        """Generate a secret key."""
        return os.urandom(32)
    
    def _load_registry(self) -> None:
        """Load hash registry from disk."""
        registry_file = self.storage_path / "hash_registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.hash_registry = json.load(f)
    
    def _save_registry(self) -> None:
        """Save hash registry to disk."""
        registry_file = self.storage_path / "hash_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.hash_registry, f, indent=2)
    
    def compute_hash(self, data: Union[np.ndarray, Dict, str, bytes]) -> str:
        """Compute HMAC hash for data."""
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode()
        elif isinstance(data, str):
            data_bytes = data.encode()
        else:
            data_bytes = data
        
        return hmac.new(self.secret_key, data_bytes, hashlib.sha256).hexdigest()
    
    def register_baseline(
        self,
        name: str,
        data: Union[np.ndarray, Dict],
    ) -> str:
        """Register baseline data with integrity hash."""
        data_hash = self.compute_hash(data)
        
        self.hash_registry[name] = {
            'hash': data_hash,
            'registered_at': datetime.now().isoformat(),
            'type': 'ndarray' if isinstance(data, np.ndarray) else 'dict',
        }
        
        self._save_registry()
        logger.info(f"Registered baseline '{name}' with hash {data_hash[:16]}...")
        return data_hash
    
    def verify_baseline(
        self,
        name: str,
        data: Union[np.ndarray, Dict],
    ) -> IntegrityReport:
        """Verify baseline data integrity."""
        if name not in self.hash_registry:
            return IntegrityReport(
                checked_at=datetime.now(),
                target=name,
                expected_hash="NOT_REGISTERED",
                actual_hash="N/A",
                is_valid=False,
                details="Baseline not registered in integrity registry",
            )
        
        expected_hash = self.hash_registry[name]['hash']
        actual_hash = self.compute_hash(data)
        is_valid = hmac.compare_digest(expected_hash, actual_hash)
        
        return IntegrityReport(
            checked_at=datetime.now(),
            target=name,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            is_valid=is_valid,
            details="" if is_valid else "HASH MISMATCH - Possible tampering detected",
        )
    
    def verify_all(self) -> Dict[str, bool]:
        """Get verification status of all registered baselines."""
        return {name: True for name in self.hash_registry}  # Placeholder


# =============================================================================
# ADVERSARIAL DETECTION
# =============================================================================

@dataclass
class AdversarialReport:
    """Report on potential adversarial input."""
    detected_at: datetime
    input_type: str
    confidence: float
    attack_type: str
    details: str
    recommended_action: str


class AdversarialDetector:
    """
    Detects potential adversarial manipulation of inputs.
    
    Checks for:
    - Unusual input patterns
    - Gradient-based perturbations
    - Distribution shifts
    - Statistical anomalies
    """
    
    def __init__(
        self,
        sensitivity: float = 0.8,
        history_window: int = 100,
    ):
        self.sensitivity = sensitivity
        self.history_window = history_window
        
        # Statistical tracking
        self.input_history = deque(maxlen=history_window)
        self.gradient_history = deque(maxlen=history_window)
        
        # Learned statistics
        self.mean_input = None
        self.std_input = None
        self.learned = False
    
    def learn_distribution(self, normal_inputs: np.ndarray) -> None:
        """Learn normal input distribution."""
        self.mean_input = np.mean(normal_inputs, axis=0)
        self.std_input = np.std(normal_inputs, axis=0) + 1e-6
        self.learned = True
        logger.info("AdversarialDetector learned normal input distribution")
    
    def check_input(self, features: np.ndarray) -> AdversarialReport:
        """
        Check input for adversarial characteristics.
        """
        attack_indicators = []
        confidence = 0.0
        
        # Store in history
        self.input_history.append(features.copy())
        
        if len(self.input_history) >= 2:
            # Check for suspiciously small perturbations (gradient attack signature)
            prev = self.input_history[-2]
            diff = np.abs(features - prev)
            
            # Very small, uniform changes might indicate gradient attack
            if np.std(diff) < 0.001 and np.mean(diff) > 0:
                attack_indicators.append("uniform_perturbation")
                confidence += 0.3
        
        if self.learned:
            # Check for out-of-distribution inputs
            z_scores = np.abs((features - self.mean_input) / self.std_input)
            max_z = np.max(z_scores)
            
            if max_z > 10:
                attack_indicators.append("extreme_outlier")
                confidence += 0.4
            elif max_z > 5:
                attack_indicators.append("significant_outlier")
                confidence += 0.2
            
            # Check for NaN or Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                attack_indicators.append("invalid_values")
                confidence += 0.5
        
        # Check for repeating patterns (replay attack)
        if len(self.input_history) >= 10:
            recent = list(self.input_history)[-10:]
            for i, hist_input in enumerate(recent[:-1]):
                if np.allclose(features, hist_input, rtol=1e-5):
                    attack_indicators.append("replay_detected")
                    confidence += 0.4
                    break
        
        # Determine attack type
        if not attack_indicators:
            attack_type = "none"
            details = "Input appears normal"
            action = "Continue normal processing"
        else:
            if "uniform_perturbation" in attack_indicators:
                attack_type = "gradient_attack"
                details = "Possible gradient-based adversarial perturbation detected"
                action = "Flag for review, apply input smoothing"
            elif "replay_detected" in attack_indicators:
                attack_type = "replay_attack"
                details = "Input appears to be a replay of previous data"
                action = "Verify data source, check for tampering"
            elif "invalid_values" in attack_indicators:
                attack_type = "corruption"
                details = "Invalid numerical values detected in input"
                action = "Reject input, check data pipeline"
            else:
                attack_type = "anomalous_input"
                details = f"Detected: {', '.join(attack_indicators)}"
                action = "Enhanced monitoring, possible manual review"
        
        confidence = min(confidence, 1.0) * self.sensitivity
        
        return AdversarialReport(
            detected_at=datetime.now(),
            input_type="behavioral_features",
            confidence=confidence,
            attack_type=attack_type,
            details=details,
            recommended_action=action,
        )


# =============================================================================
# GRACEFUL DEGRADATION
# =============================================================================

class SystemHealth(Enum):
    """System health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    IMPAIRED = "impaired"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class ComponentStatus:
    """Status of a system component."""
    name: str
    healthy: bool
    last_check: datetime
    error_message: str = ""
    recovery_action: str = ""


class GracefulDegradationManager:
    """
    Manages system degradation and recovery.
    
    Ensures the system continues operating even when components fail.
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentStatus] = {}
        self.fallback_modes: Dict[str, str] = {}
        self.error_counts: Dict[str, int] = {}
        self.max_errors = 5
        
        # Define fallback modes
        self._register_fallbacks()
    
    def _register_fallbacks(self) -> None:
        """Register fallback modes for components."""
        self.fallback_modes = {
            'lstm_vae': 'Use simpler statistical anomaly detection',
            'ensemble': 'Fall back to single model (LSTM-VAE)',
            'video_feed': 'Use cached frames or static analysis',
            'database': 'Use in-memory storage with periodic disk writes',
            'network': 'Continue offline, queue events for later sync',
            'ai_classification': 'Use rule-based classification',
            'prediction': 'Disable prediction, use current-state only',
        }
    
    def register_component(self, name: str) -> None:
        """Register a component for monitoring."""
        self.components[name] = ComponentStatus(
            name=name,
            healthy=True,
            last_check=datetime.now(),
        )
        self.error_counts[name] = 0
    
    def report_error(
        self,
        component: str,
        error_message: str,
    ) -> ComponentStatus:
        """Report an error in a component."""
        if component not in self.components:
            self.register_component(component)
        
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        
        status = self.components[component]
        status.last_check = datetime.now()
        status.error_message = error_message
        
        if self.error_counts[component] >= self.max_errors:
            status.healthy = False
            status.recovery_action = self.fallback_modes.get(
                component, 'Manual intervention required'
            )
        
        logger.warning(f"Component {component} error ({self.error_counts[component]}): {error_message}")
        return status
    
    def report_recovery(self, component: str) -> None:
        """Report component recovery."""
        if component in self.components:
            self.components[component].healthy = True
            self.components[component].error_message = ""
            self.components[component].recovery_action = ""
            self.error_counts[component] = 0
            logger.info(f"Component {component} recovered")
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health."""
        if not self.components:
            return SystemHealth.HEALTHY
        
        unhealthy = [c for c in self.components.values() if not c.healthy]
        
        if not unhealthy:
            return SystemHealth.HEALTHY
        elif len(unhealthy) == 1:
            return SystemHealth.DEGRADED
        elif len(unhealthy) <= len(self.components) // 2:
            return SystemHealth.IMPAIRED
        elif len(unhealthy) < len(self.components):
            return SystemHealth.CRITICAL
        else:
            return SystemHealth.FAILED
    
    def get_active_fallbacks(self) -> List[Dict[str, str]]:
        """Get list of active fallback modes."""
        fallbacks = []
        for comp in self.components.values():
            if not comp.healthy:
                fallbacks.append({
                    'component': comp.name,
                    'error': comp.error_message,
                    'fallback': comp.recovery_action,
                })
        return fallbacks
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get complete status report."""
        return {
            'system_health': self.get_system_health().value,
            'components': {
                name: {
                    'healthy': status.healthy,
                    'last_check': status.last_check.isoformat(),
                    'error': status.error_message,
                }
                for name, status in self.components.items()
            },
            'active_fallbacks': self.get_active_fallbacks(),
            'error_counts': dict(self.error_counts),
        }


# =============================================================================
# INTEGRATED SECURITY MANAGER
# =============================================================================

class SecurityManager:
    """
    Integrated security management for the NOISE FLOOR system.
    """
    
    def __init__(
        self,
        storage_path: str = "./security",
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.audit = AuditTrail(storage_path=str(self.storage_path / "audit"))
        self.integrity = DataIntegrityManager(storage_path=str(self.storage_path / "integrity"))
        self.adversarial = AdversarialDetector()
        self.degradation = GracefulDegradationManager()
        
        # Register core components
        for comp in ['lstm_vae', 'ensemble', 'video_feed', 'database', 'ai_classification']:
            self.degradation.register_component(comp)
        
        # Log system start
        self.audit.log(
            event_type=AuditEventType.SYSTEM_START,
            actor="system",
            target="noise_floor",
            details={'version': '2.0.0'},
        )
        
        logger.info("SecurityManager initialized")
    
    def check_input_security(self, features: np.ndarray) -> Tuple[bool, Optional[AdversarialReport]]:
        """
        Check input security.
        
        Returns (is_safe, report_if_unsafe)
        """
        report = self.adversarial.check_input(features)
        
        if report.confidence > 0.5:
            self.audit.log(
                event_type=AuditEventType.ADVERSARIAL_DETECTED,
                actor="system",
                target="input_features",
                details={
                    'attack_type': report.attack_type,
                    'confidence': report.confidence,
                    'details': report.details,
                },
            )
            return False, report
        
        return True, None
    
    def verify_baseline_integrity(
        self,
        baseline_name: str,
        baseline_data: Union[np.ndarray, Dict],
    ) -> bool:
        """Verify baseline integrity."""
        report = self.integrity.verify_baseline(baseline_name, baseline_data)
        
        event_type = (
            AuditEventType.INTEGRITY_CHECK_PASSED
            if report.is_valid
            else AuditEventType.INTEGRITY_CHECK_FAILED
        )
        
        self.audit.log(
            event_type=event_type,
            actor="system",
            target=baseline_name,
            details={
                'is_valid': report.is_valid,
                'expected_hash': report.expected_hash[:16] + '...',
                'actual_hash': report.actual_hash[:16] + '...',
            },
        )
        
        return report.is_valid
    
    def log_operator_action(
        self,
        operator_id: str,
        action_type: AuditEventType,
        target: str,
        details: Dict = None,
    ) -> None:
        """Log operator action."""
        self.audit.log(
            event_type=action_type,
            actor=operator_id,
            target=target,
            details=details or {},
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        chain_valid, chain_error = self.audit.verify_chain()
        
        return {
            'system_health': self.degradation.get_system_health().value,
            'audit_chain_valid': chain_valid,
            'audit_chain_error': chain_error,
            'audit_entries': len(self.audit.entries),
            'registered_baselines': list(self.integrity.hash_registry.keys()),
            'active_fallbacks': self.degradation.get_active_fallbacks(),
            'security_events': len(self.audit.get_security_events()),
        }
