import cv2
import numpy as np
from pathlib import Path



def variance_of_laplacian(image):
    # Measure image blur
    return cv2.Laplacian(image, cv2.CV_64F).var()


def extract_quality_frames(
    video_path, output_dir, frame_interval=10, blur_threshold=100
):
    cap = cv2.VideoCapture(video_path)
    Path(output_dir).mkdir(exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Check image quality
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = variance_of_laplacian(gray)

            if blur_score > blur_threshold:
                # Save frame if it passes quality check
                output_path = f"{output_dir}/frame_{saved_count:04d}.jpg"
                cv2.imwrite(output_path, frame)
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} quality frames")


# Usage
video_path = "Videos/Eiffel_Tower.m4v"
output_dir = "frames_for_colmap"
extract_quality_frames(video_path, output_dir, frame_interval=60, blur_threshold=100)
