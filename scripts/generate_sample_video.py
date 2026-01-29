"""
Generate Sample Surveillance Video for NOISE FLOOR Demo
=========================================================
Creates realistic-looking surveillance footage with:
- Normal pedestrian movement patterns
- Anomalous behavior injection (running, loitering, unusual paths)

This allows demonstration of real video processing without
requiring external dataset downloads.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def generate_sample_surveillance_video(
    output_path: str,
    duration_seconds: int = 30,
    fps: int = 15,
    width: int = 238,
    height: int = 158,
    anomaly_start_ratio: float = 0.5
):
    """
    Generate a sample surveillance video with normal and anomalous behavior.
    
    Args:
        output_path: Path to save the video
        duration_seconds: Total video duration
        fps: Frames per second
        width: Frame width (UCSD Ped1 is 238x158)
        height: Frame height
        anomaly_start_ratio: When anomaly starts (0.5 = halfway)
    """
    
    total_frames = duration_seconds * fps
    anomaly_start_frame = int(total_frames * anomaly_start_ratio)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating {duration_seconds}s surveillance video...")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Anomaly starts at frame: {anomaly_start_frame}")
    
    # Simulated pedestrians
    class Pedestrian:
        def __init__(self, is_anomalous=False):
            self.x = np.random.randint(0, width)
            self.y = np.random.randint(0, height)
            self.vx = np.random.uniform(-2, 2)
            self.vy = np.random.uniform(-1, 1)
            self.size = np.random.randint(8, 15)
            self.is_anomalous = is_anomalous
            self.color = (100 + np.random.randint(0, 100),
                         100 + np.random.randint(0, 100),
                         100 + np.random.randint(0, 100))
            
        def update(self, frame_num, is_anomaly_period):
            if is_anomaly_period and self.is_anomalous:
                # Anomalous behavior: faster, erratic movement
                self.vx += np.random.uniform(-1, 1)
                self.vy += np.random.uniform(-0.5, 0.5)
                # Speed up
                self.vx = np.clip(self.vx * 1.2, -8, 8)
                self.vy = np.clip(self.vy * 1.2, -5, 5)
            else:
                # Normal walking behavior
                self.vx += np.random.uniform(-0.2, 0.2)
                self.vy += np.random.uniform(-0.1, 0.1)
                self.vx = np.clip(self.vx, -2, 2)
                self.vy = np.clip(self.vy, -1, 1)
            
            self.x += self.vx
            self.y += self.vy
            
            # Wrap around
            self.x = self.x % width
            self.y = self.y % height
            
        def draw(self, frame):
            cv2.circle(frame, (int(self.x), int(self.y)), 
                      self.size, self.color, -1)
            # Head
            cv2.circle(frame, (int(self.x), int(self.y - self.size)), 
                      self.size // 2, self.color, -1)
    
    # Create pedestrians
    normal_peds = [Pedestrian(is_anomalous=False) for _ in range(8)]
    anomalous_peds = [Pedestrian(is_anomalous=True) for _ in range(3)]
    
    # Background - gray concrete/pavement look
    background = np.ones((height, width, 3), dtype=np.uint8) * 128
    # Add some texture
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    background = cv2.add(background, noise)
    
    for frame_num in range(total_frames):
        # Create frame
        frame = background.copy()
        
        is_anomaly_period = frame_num >= anomaly_start_frame
        
        # Update and draw pedestrians
        for ped in normal_peds:
            ped.update(frame_num, is_anomaly_period)
            ped.draw(frame)
        
        if is_anomaly_period:
            for ped in anomalous_peds:
                ped.update(frame_num, is_anomaly_period)
                ped.draw(frame)
        
        # Add timestamp overlay
        timestamp = f"FRAME: {frame_num:04d}"
        cv2.putText(frame, timestamp, (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Status indicator
        status = "ANOMALY" if is_anomaly_period else "NORMAL"
        color = (0, 0, 255) if is_anomaly_period else (0, 255, 0)
        cv2.putText(frame, status, (width - 60, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        out.write(frame)
        
        if frame_num % 100 == 0:
            print(f"  Generated frame {frame_num}/{total_frames}")
    
    out.release()
    print(f"\n✓ Video saved to: {output_path}")
    print(f"  Normal frames: 0-{anomaly_start_frame-1}")
    print(f"  Anomaly frames: {anomaly_start_frame}-{total_frames-1}")
    
    return output_path


if __name__ == "__main__":
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "sample_video"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = str(output_dir / "surveillance_sample.avi")
    
    generate_sample_surveillance_video(
        output_path=output_path,
        duration_seconds=30,
        fps=15,
        anomaly_start_ratio=0.5
    )
    
    print("\n" + "="*60)
    print("To run NOISE FLOOR with this video:")
    print(f"  python main.py --real-demo --video \"{output_path}\"")
    print("="*60)
