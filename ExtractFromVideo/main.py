import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


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

    pbar = tqdm()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        pbar.update()

        if frame_count % frame_interval == 0:
            # Check image quality
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = variance_of_laplacian(gray)

            if blur_score > blur_threshold:
                # Save frame if it passes quality check
                output_path = f"{output_dir}/frame_{saved_count:04d}.jpg"
                cv2.imwrite(output_path, frame)
                saved_count += 1



    cap.release()
    print(f"Extracted {saved_count} quality frames")


# Usage
video_path = "Videos/IRAN Takht-e Jamshid - Perspolis |   تخت جمشید.mp4"
output_dir = "input"
extract_quality_frames(video_path, output_dir, frame_interval=10, blur_threshold=100)
