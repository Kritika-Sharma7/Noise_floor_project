"""
NOISE FLOOR - Incident Logger & Analytics Module
==================================================
Comprehensive incident logging, analytics, and reporting system.

This module provides:
1. Incident Logging - Timestamped log of all alerts
2. Analytics Dashboard - Historical trend analysis
3. Report Generation - Daily/Weekly summary reports
4. Export Functionality - CSV/PDF export

DESIGN PHILOSOPHY:
------------------
"Every incident is data. Every data point enables improvement."
"""

import numpy as np
import json
import csv
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class IncidentStatus(Enum):
    """Incident status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


@dataclass
class Incident:
    """Single incident record."""
    # Core identification
    incident_id: str
    timestamp: datetime
    frame_number: int
    session_id: str
    
    # Detection metrics
    tdi: float
    zone: str
    trend: str
    confidence: float
    
    # Classification
    anomaly_category: str
    severity: IncidentSeverity
    
    # Attribution
    top_features: List[Dict[str, float]]
    explanation: str
    
    # Status tracking
    status: IncidentStatus = IncidentStatus.OPEN
    operator_notes: str = ""
    resolved_at: Optional[datetime] = None
    resolution_type: str = ""
    
    # Audit trail
    created_by: str = "system"
    last_modified: Optional[datetime] = None
    last_modified_by: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.last_modified:
            data['last_modified'] = self.last_modified.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Incident':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['severity'] = IncidentSeverity(data['severity'])
        data['status'] = IncidentStatus(data['status'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        if data.get('last_modified'):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


@dataclass
class IncidentStats:
    """Statistics about incidents."""
    total_incidents: int
    by_severity: Dict[str, int]
    by_zone: Dict[str, int]
    by_category: Dict[str, int]
    by_status: Dict[str, int]
    avg_tdi: float
    max_tdi: float
    avg_resolution_time: Optional[float]  # minutes
    false_positive_rate: float
    incidents_per_hour: float


@dataclass 
class DailyReport:
    """Daily intelligence summary report."""
    date: datetime
    session_count: int
    total_incidents: int
    incidents_by_severity: Dict[str, int]
    incidents_by_category: Dict[str, int]
    peak_tdi: float
    avg_tdi: float
    time_in_zones: Dict[str, float]  # percentage
    detection_performance: Dict[str, float]
    key_events: List[Dict]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class IncidentLogger:
    """
    Comprehensive incident logging system.
    
    Logs all security-relevant events with full audit trail.
    """
    
    def __init__(
        self,
        storage_path: str = "./incident_logs",
        max_memory_incidents: int = 1000,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_incidents = max_memory_incidents
        self.incidents: List[Incident] = []
        self.incident_index: Dict[str, int] = {}  # id -> index
        
        # Session tracking
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Alert thresholds for incident creation
        self.min_tdi_for_incident = 25.0  # Only log incidents above this TDI
        
        logger.info(f"IncidentLogger initialized, storage: {self.storage_path}")
    
    def _generate_incident_id(self, timestamp: datetime, frame: int) -> str:
        """Generate unique incident ID."""
        data = f"{timestamp.isoformat()}-{frame}-{self.current_session_id}"
        return hashlib.md5(data.encode()).hexdigest()[:12].upper()
    
    def log_incident(
        self,
        frame_number: int,
        tdi: float,
        zone: str,
        trend: str,
        confidence: float,
        anomaly_category: str,
        severity: IncidentSeverity,
        top_features: List[Dict[str, float]],
        explanation: str,
    ) -> Incident:
        """
        Log a new incident.
        
        Returns the created Incident object.
        """
        timestamp = datetime.now()
        incident_id = self._generate_incident_id(timestamp, frame_number)
        
        incident = Incident(
            incident_id=incident_id,
            timestamp=timestamp,
            frame_number=frame_number,
            session_id=self.current_session_id,
            tdi=tdi,
            zone=zone,
            trend=trend,
            confidence=confidence,
            anomaly_category=anomaly_category,
            severity=severity,
            top_features=top_features,
            explanation=explanation,
        )
        
        self.incidents.append(incident)
        self.incident_index[incident_id] = len(self.incidents) - 1
        
        # Trim memory if needed
        if len(self.incidents) > self.max_memory_incidents:
            self._archive_old_incidents()
        
        logger.info(f"Incident logged: {incident_id} - {zone} (TDI={tdi:.1f})")
        return incident
    
    def update_incident(
        self,
        incident_id: str,
        status: Optional[IncidentStatus] = None,
        operator_notes: Optional[str] = None,
        resolution_type: Optional[str] = None,
        modified_by: str = "operator",
    ) -> Optional[Incident]:
        """Update an existing incident."""
        if incident_id not in self.incident_index:
            logger.warning(f"Incident not found: {incident_id}")
            return None
        
        idx = self.incident_index[incident_id]
        incident = self.incidents[idx]
        
        if status:
            incident.status = status
            if status in [IncidentStatus.RESOLVED, IncidentStatus.FALSE_POSITIVE]:
                incident.resolved_at = datetime.now()
        
        if operator_notes:
            incident.operator_notes = operator_notes
        
        if resolution_type:
            incident.resolution_type = resolution_type
        
        incident.last_modified = datetime.now()
        incident.last_modified_by = modified_by
        
        logger.info(f"Incident updated: {incident_id}")
        return incident
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        if incident_id in self.incident_index:
            return self.incidents[self.incident_index[incident_id]]
        return None
    
    def get_recent_incidents(
        self,
        count: int = 50,
        min_severity: IncidentSeverity = IncidentSeverity.LOW,
    ) -> List[Incident]:
        """Get recent incidents."""
        filtered = [i for i in self.incidents if i.severity.value >= min_severity.value]
        return filtered[-count:]
    
    def get_incidents_by_timerange(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Incident]:
        """Get incidents within time range."""
        return [
            i for i in self.incidents
            if start <= i.timestamp <= end
        ]
    
    def get_statistics(
        self,
        incidents: Optional[List[Incident]] = None,
    ) -> IncidentStats:
        """Calculate statistics for incidents."""
        if incidents is None:
            incidents = self.incidents
        
        if not incidents:
            return IncidentStats(
                total_incidents=0,
                by_severity={},
                by_zone={},
                by_category={},
                by_status={},
                avg_tdi=0.0,
                max_tdi=0.0,
                avg_resolution_time=None,
                false_positive_rate=0.0,
                incidents_per_hour=0.0,
            )
        
        # Count by categories
        by_severity = defaultdict(int)
        by_zone = defaultdict(int)
        by_category = defaultdict(int)
        by_status = defaultdict(int)
        
        tdi_values = []
        resolution_times = []
        
        for inc in incidents:
            by_severity[inc.severity.name] += 1
            by_zone[inc.zone] += 1
            by_category[inc.anomaly_category] += 1
            by_status[inc.status.value] += 1
            tdi_values.append(inc.tdi)
            
            if inc.resolved_at and inc.timestamp:
                resolution_time = (inc.resolved_at - inc.timestamp).total_seconds() / 60
                resolution_times.append(resolution_time)
        
        # Calculate time span
        if len(incidents) > 1:
            time_span = (incidents[-1].timestamp - incidents[0].timestamp).total_seconds() / 3600
            incidents_per_hour = len(incidents) / max(time_span, 0.1)
        else:
            incidents_per_hour = 0.0
        
        # False positive rate
        fp_count = by_status.get('false_positive', 0)
        resolved_count = fp_count + by_status.get('resolved', 0)
        fp_rate = fp_count / resolved_count if resolved_count > 0 else 0.0
        
        return IncidentStats(
            total_incidents=len(incidents),
            by_severity=dict(by_severity),
            by_zone=dict(by_zone),
            by_category=dict(by_category),
            by_status=dict(by_status),
            avg_tdi=float(np.mean(tdi_values)) if tdi_values else 0.0,
            max_tdi=float(np.max(tdi_values)) if tdi_values else 0.0,
            avg_resolution_time=float(np.mean(resolution_times)) if resolution_times else None,
            false_positive_rate=fp_rate,
            incidents_per_hour=incidents_per_hour,
        )
    
    def _archive_old_incidents(self) -> None:
        """Archive old incidents to disk."""
        # Keep most recent half
        to_archive = self.incidents[:len(self.incidents) // 2]
        self.incidents = self.incidents[len(self.incidents) // 2:]
        
        # Rebuild index
        self.incident_index = {inc.incident_id: i for i, inc in enumerate(self.incidents)}
        
        # Save to file
        archive_file = self.storage_path / f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(archive_file, 'w') as f:
            json.dump([inc.to_dict() for inc in to_archive], f, indent=2)
        
        logger.info(f"Archived {len(to_archive)} incidents to {archive_file}")
    
    def export_to_csv(self, filepath: str = None) -> str:
        """Export incidents to CSV."""
        if filepath is None:
            filepath = self.storage_path / f"incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = Path(filepath)
        
        headers = [
            'incident_id', 'timestamp', 'frame', 'session', 'tdi', 'zone',
            'trend', 'confidence', 'category', 'severity', 'status',
            'operator_notes', 'explanation'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for inc in self.incidents:
                writer.writerow([
                    inc.incident_id,
                    inc.timestamp.isoformat(),
                    inc.frame_number,
                    inc.session_id,
                    f"{inc.tdi:.2f}",
                    inc.zone,
                    inc.trend,
                    f"{inc.confidence:.3f}",
                    inc.anomaly_category,
                    inc.severity.name,
                    inc.status.value,
                    inc.operator_notes,
                    inc.explanation[:100],  # Truncate long explanations
                ])
        
        logger.info(f"Exported {len(self.incidents)} incidents to {filepath}")
        return str(filepath)
    
    def export_to_json(self, filepath: str = None) -> str:
        """Export incidents to JSON."""
        if filepath is None:
            filepath = self.storage_path / f"incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            json.dump([inc.to_dict() for inc in self.incidents], f, indent=2)
        
        logger.info(f"Exported {len(self.incidents)} incidents to {filepath}")
        return str(filepath)


class AnalyticsDashboard:
    """
    Analytics engine for historical analysis.
    """
    
    def __init__(self, incident_logger: IncidentLogger):
        self.logger = incident_logger
        
        # Time-series data for visualization
        self.tdi_timeseries: List[Tuple[datetime, float]] = []
        self.zone_timeseries: List[Tuple[datetime, str]] = []
        
    def record_frame(
        self,
        timestamp: datetime,
        tdi: float,
        zone: str,
    ) -> None:
        """Record frame data for analytics."""
        self.tdi_timeseries.append((timestamp, tdi))
        self.zone_timeseries.append((timestamp, zone))
        
        # Keep last 24 hours of data in memory
        cutoff = datetime.now() - timedelta(hours=24)
        self.tdi_timeseries = [(t, v) for t, v in self.tdi_timeseries if t > cutoff]
        self.zone_timeseries = [(t, v) for t, v in self.zone_timeseries if t > cutoff]
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict]:
        """Get hourly statistics."""
        now = datetime.now()
        stats = []
        
        for h in range(hours):
            start = now - timedelta(hours=h+1)
            end = now - timedelta(hours=h)
            
            # Filter TDI values
            hour_tdi = [v for t, v in self.tdi_timeseries if start <= t < end]
            
            if hour_tdi:
                stats.append({
                    'hour': end.strftime('%H:%M'),
                    'avg_tdi': np.mean(hour_tdi),
                    'max_tdi': np.max(hour_tdi),
                    'min_tdi': np.min(hour_tdi),
                    'samples': len(hour_tdi),
                })
            else:
                stats.append({
                    'hour': end.strftime('%H:%M'),
                    'avg_tdi': 0,
                    'max_tdi': 0,
                    'min_tdi': 0,
                    'samples': 0,
                })
        
        return list(reversed(stats))
    
    def get_zone_distribution(self, hours: int = 24) -> Dict[str, float]:
        """Get zone time distribution."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_zones = [z for t, z in self.zone_timeseries if t > cutoff]
        
        if not recent_zones:
            return {'NORMAL': 100, 'WATCH': 0, 'WARNING': 0, 'CRITICAL': 0}
        
        total = len(recent_zones)
        distribution = defaultdict(int)
        for zone in recent_zones:
            distribution[zone] += 1
        
        return {zone: count / total * 100 for zone, count in distribution.items()}
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze overall trends."""
        if len(self.tdi_timeseries) < 10:
            return {'trend': 'insufficient_data', 'slope': 0, 'volatility': 0}
        
        values = [v for _, v in self.tdi_timeseries[-100:]]
        
        # Calculate trend slope
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Calculate volatility
        volatility = np.std(values)
        
        # Determine trend
        if slope > 0.1:
            trend = 'deteriorating'
        elif slope < -0.1:
            trend = 'improving'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'volatility': float(volatility),
            'current_avg': float(np.mean(values[-10:])),
            'overall_avg': float(np.mean(values)),
        }
    
    def generate_comparison(
        self,
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime,
    ) -> Dict[str, Any]:
        """Compare two time periods."""
        # Get incidents for each period
        period1_incidents = self.logger.get_incidents_by_timerange(period1_start, period1_end)
        period2_incidents = self.logger.get_incidents_by_timerange(period2_start, period2_end)
        
        stats1 = self.logger.get_statistics(period1_incidents)
        stats2 = self.logger.get_statistics(period2_incidents)
        
        return {
            'period1': {
                'range': f"{period1_start.strftime('%Y-%m-%d')} to {period1_end.strftime('%Y-%m-%d')}",
                'stats': asdict(stats1) if hasattr(stats1, '__dataclass_fields__') else stats1,
            },
            'period2': {
                'range': f"{period2_start.strftime('%Y-%m-%d')} to {period2_end.strftime('%Y-%m-%d')}",
                'stats': asdict(stats2) if hasattr(stats2, '__dataclass_fields__') else stats2,
            },
            'change': {
                'incidents': stats2.total_incidents - stats1.total_incidents,
                'avg_tdi_change': stats2.avg_tdi - stats1.avg_tdi,
            }
        }


class ReportGenerator:
    """
    Generate intelligence reports.
    """
    
    def __init__(
        self,
        incident_logger: IncidentLogger,
        analytics: AnalyticsDashboard,
    ):
        self.logger = incident_logger
        self.analytics = analytics
    
    def generate_daily_report(self, date: datetime = None) -> DailyReport:
        """Generate daily summary report."""
        if date is None:
            date = datetime.now()
        
        # Get day boundaries
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        # Get incidents
        incidents = self.logger.get_incidents_by_timerange(start, end)
        stats = self.logger.get_statistics(incidents)
        
        # Get zone distribution
        zone_dist = self.analytics.get_zone_distribution(hours=24)
        
        # Key events (high severity incidents)
        key_events = [
            {
                'time': inc.timestamp.strftime('%H:%M'),
                'category': inc.anomaly_category,
                'tdi': inc.tdi,
                'severity': inc.severity.name,
            }
            for inc in incidents
            if inc.severity.value >= IncidentSeverity.HIGH.value
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats, zone_dist)
        
        return DailyReport(
            date=date,
            session_count=len(set(inc.session_id for inc in incidents)),
            total_incidents=stats.total_incidents,
            incidents_by_severity=stats.by_severity,
            incidents_by_category=stats.by_category,
            peak_tdi=stats.max_tdi,
            avg_tdi=stats.avg_tdi,
            time_in_zones=zone_dist,
            detection_performance={
                'false_positive_rate': stats.false_positive_rate,
                'incidents_per_hour': stats.incidents_per_hour,
            },
            key_events=key_events,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        stats: IncidentStats,
        zone_dist: Dict[str, float],
    ) -> List[str]:
        """Generate recommendations based on data."""
        recommendations = []
        
        # High incident rate
        if stats.incidents_per_hour > 5:
            recommendations.append(
                "High incident rate detected. Consider reviewing zone threshold settings."
            )
        
        # High false positive rate
        if stats.false_positive_rate > 0.3:
            recommendations.append(
                "False positive rate above 30%. Recommend baseline recalibration with operator input."
            )
        
        # Too much time in elevated zones
        watch_warning = zone_dist.get('WATCH', 0) + zone_dist.get('WARNING', 0)
        if watch_warning > 30:
            recommendations.append(
                "Extended time in elevated zones. Review for potential baseline drift or persistent threat."
            )
        
        # Specific category recommendations
        if stats.by_category.get('intrusion', 0) > 0:
            recommendations.append(
                "Intrusion events detected. Verify perimeter security measures."
            )
        
        if stats.by_category.get('coordinated', 0) > 0:
            recommendations.append(
                "Coordinated activity detected. Recommend command briefing and enhanced monitoring."
            )
        
        if not recommendations:
            recommendations.append("System operating within normal parameters. No action required.")
        
        return recommendations
    
    def format_report_text(self, report: DailyReport) -> str:
        """Format report as text."""
        lines = [
            "=" * 60,
            "NOISE FLOOR - Daily Intelligence Summary",
            "=" * 60,
            f"Date: {report.date.strftime('%Y-%m-%d')}",
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "OVERVIEW",
            "-" * 40,
            f"Sessions: {report.session_count}",
            f"Total Incidents: {report.total_incidents}",
            f"Peak TDI: {report.peak_tdi:.1f}",
            f"Average TDI: {report.avg_tdi:.1f}",
            "",
            "ZONE DISTRIBUTION",
            "-" * 40,
        ]
        
        for zone, pct in report.time_in_zones.items():
            lines.append(f"  {zone}: {pct:.1f}%")
        
        lines.extend([
            "",
            "INCIDENTS BY SEVERITY",
            "-" * 40,
        ])
        
        for sev, count in report.incidents_by_severity.items():
            lines.append(f"  {sev}: {count}")
        
        if report.key_events:
            lines.extend([
                "",
                "KEY EVENTS",
                "-" * 40,
            ])
            for event in report.key_events[:5]:
                lines.append(f"  [{event['time']}] {event['category']} - TDI: {event['tdi']:.1f}")
        
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        for rec in report.recommendations:
            lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def export_report(self, report: DailyReport, filepath: str = None) -> str:
        """Export report to file."""
        if filepath is None:
            filepath = f"./reports/daily_report_{report.date.strftime('%Y%m%d')}.txt"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        text = self.format_report_text(report)
        with open(filepath, 'w') as f:
            f.write(text)
        
        logger.info(f"Report exported to {filepath}")
        return filepath
