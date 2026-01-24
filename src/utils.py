"""
NOISE FLOOR - Utility Functions
================================
Helper functions for data loading, visualization, and system utilities.
"""

import os
import sys
import cv2
import numpy as np
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import logging
import json
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress indication.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = downloaded / total_size * 100
                        print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
        
        print()
        return True
    
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract tar.gz or zip archive.
    """
    try:
        if str(archive_path).endswith('.tar.gz') or str(archive_path).endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            logger.error(f"Unknown archive format: {archive_path}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def load_ucsd_dataset(
    data_dir: Path,
    subset: str = 'UCSDped1',
    split: str = 'Train'
) -> List[Path]:
    """
    Load UCSD Pedestrian dataset frame directories.
    
    Dataset structure:
    UCSD_Anomaly_Dataset.v1p2/
        UCSDped1/
            Train/
                Train001/
                Train002/
                ...
            Test/
                Test001/
                ...
        UCSDped2/
            ...
    """
    base_path = data_dir / 'UCSD_Anomaly_Dataset.v1p2' / subset / split
    
    if not base_path.exists():
        logger.warning(f"Dataset path not found: {base_path}")
        return []
    
    # Get all sequence directories
    sequences = sorted([d for d in base_path.iterdir() if d.is_dir()])
    logger.info(f"Found {len(sequences)} sequences in {base_path}")
    
    return sequences


def load_frames_from_directory(
    frames_dir: Path,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields frames from a directory of images.
    """
    # Common image extensions
    extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
    
    frame_files = []
    for ext in extensions:
        frame_files.extend(frames_dir.glob(ext))
    
    frame_files = sorted(frame_files)
    
    if not frame_files:
        logger.warning(f"No frames found in {frames_dir}")
        return
    
    for i, frame_file in enumerate(frame_files):
        if max_frames and i >= max_frames:
            break
        
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
        
        if resize:
            frame = cv2.resize(frame, resize)
        
        yield frame


def video_to_frames(
    video_path: str,
    output_dir: Optional[Path] = None,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields frames from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_idx >= max_frames:
            break
        
        if resize:
            frame = cv2.resize(frame, resize)
        
        if output_dir:
            output_path = output_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(output_path), frame)
        
        yield frame
        frame_idx += 1
    
    cap.release()


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def create_drift_visualization(
    frame: np.ndarray,
    drift_score: float,
    zone: str,
    zone_color: str,
    thresholds: dict
) -> np.ndarray:
    """
    Overlay drift information on a video frame.
    """
    # Create copy
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Convert zone color from hex to BGR
    color_hex = zone_color.lstrip('#')
    color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    
    # Draw zone indicator bar at top
    bar_height = 30
    cv2.rectangle(vis_frame, (0, 0), (w, bar_height), color_bgr, -1)
    
    # Add zone text
    cv2.putText(
        vis_frame,
        f"ZONE: {zone.upper()}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Add drift score
    cv2.putText(
        vis_frame,
        f"Drift: {drift_score:.2f}",
        (w - 120, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Draw drift meter
    meter_x = 10
    meter_y = h - 40
    meter_width = w - 20
    meter_height = 20
    
    # Background
    cv2.rectangle(
        vis_frame,
        (meter_x, meter_y),
        (meter_x + meter_width, meter_y + meter_height),
        (50, 50, 50),
        -1
    )
    
    # Threshold markers
    for threshold_name, threshold_val in thresholds.items():
        if threshold_val <= 5:  # Only show reasonable thresholds
            x = meter_x + int(threshold_val / 5 * meter_width)
            cv2.line(vis_frame, (x, meter_y), (x, meter_y + meter_height), (200, 200, 200), 1)
    
    # Current drift level
    fill_width = min(int(drift_score / 5 * meter_width), meter_width)
    cv2.rectangle(
        vis_frame,
        (meter_x, meter_y),
        (meter_x + fill_width, meter_y + meter_height),
        color_bgr,
        -1
    )
    
    return vis_frame


def plot_drift_timeline(
    drift_scores: np.ndarray,
    zones: List[str],
    zone_colors: dict,
    thresholds: dict,
    save_path: Optional[str] = None
):
    """
    Create a matplotlib plot of drift scores over time.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot drift scores
    frames = np.arange(len(drift_scores))
    ax.plot(frames, drift_scores, 'b-', linewidth=1, alpha=0.7, label='Drift Score')
    
    # Color background by zone
    current_zone = zones[0] if zones else 'normal'
    zone_start = 0
    
    for i, zone in enumerate(zones + [None]):
        if zone != current_zone:
            color = zone_colors.get(current_zone, '#CCCCCC')
            ax.axvspan(zone_start, i, alpha=0.2, color=color)
            current_zone = zone
            zone_start = i
    
    # Add threshold lines
    for name, value in thresholds.items():
        color = zone_colors.get(name, 'gray')
        ax.axhline(y=value, color=color, linestyle='--', alpha=0.5, label=f'{name.title()} threshold')
    
    # Labels and legend
    ax.set_xlabel('Frame')
    ax.set_ylabel('Drift Score')
    ax.set_title('NOISE FLOOR - Behavioral Drift Detection')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_results(results: dict, path: Path):
    """Save results dictionary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    logger.info(f"Results saved to {path}")


def load_results(path: Path) -> dict:
    """Load results dictionary from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def save_model_state(state: dict, path: Path):
    """Save model state using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    
    logger.info(f"Model state saved to {path}")


def load_model_state(path: Path) -> dict:
    """Load model state from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# DEMO DATA GENERATION
# =============================================================================

def create_demo_video(
    output_path: str,
    duration_seconds: int = 30,
    fps: int = 30,
    size: Tuple[int, int] = (320, 240),
    drift_start_ratio: float = 0.5
):
    """
    Create a synthetic demo video with simulated drift.
    
    The video shows:
    - Normal period: Regular random motion patterns
    - Drift period: Gradually increasing motion intensity
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    total_frames = duration_seconds * fps
    drift_start = int(total_frames * drift_start_ratio)
    
    # Create moving particles
    n_particles = 20
    particles = np.random.rand(n_particles, 2) * np.array([size[0], size[1]])
    velocities = np.random.randn(n_particles, 2) * 2
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Add some noise
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Calculate drift factor
        if frame_idx >= drift_start:
            drift_progress = (frame_idx - drift_start) / (total_frames - drift_start)
            drift_factor = 1 + drift_progress * 3  # Gradually increase motion
        else:
            drift_factor = 1.0
        
        # Update and draw particles
        velocities_scaled = velocities * drift_factor
        particles += velocities_scaled
        
        # Bounce off walls
        for i in range(n_particles):
            if particles[i, 0] < 0 or particles[i, 0] >= size[0]:
                velocities[i, 0] *= -1
            if particles[i, 1] < 0 or particles[i, 1] >= size[1]:
                velocities[i, 1] *= -1
            particles[i] = np.clip(particles[i], [0, 0], [size[0]-1, size[1]-1])
        
        # Draw particles
        for i in range(n_particles):
            center = (int(particles[i, 0]), int(particles[i, 1]))
            radius = int(5 + drift_factor * 3)
            color = (0, 255, 0) if frame_idx < drift_start else (
                int(255 * min(1, drift_factor - 1)),
                int(255 * max(0, 2 - drift_factor)),
                0
            )
            cv2.circle(frame, center, radius, color, -1)
        
        # Add frame counter and drift indicator
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        if frame_idx >= drift_start:
            cv2.putText(
                frame,
                f"DRIFT: {drift_factor:.1f}x",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        
        out.write(frame)
    
    out.release()
    logger.info(f"Demo video created: {output_path}")
    logger.info(f"Total frames: {total_frames}, Drift starts at frame {drift_start}")


if __name__ == "__main__":
    # Test utilities
    print("Testing Utility Functions")
    print("=" * 50)
    
    # Create demo video
    demo_path = "output/demo_video.mp4"
    Path("output").mkdir(exist_ok=True)
    create_demo_video(demo_path, duration_seconds=10)
    
    print(f"\nDemo video created at: {demo_path}")
    print("\nUtility module ready!")
