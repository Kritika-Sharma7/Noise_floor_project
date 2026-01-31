"""Quick script to check video properties"""
import cv2

video_path = r"D:\Certifications\Hackathons\SnowHack\NoiseFloor\Noise_floor_project\data\videos\drone_test.mp4"
cap = cv2.VideoCapture(video_path)

print(f"Video: {video_path}")
print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
print(f"Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Duration: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))} seconds")

cap.release()
