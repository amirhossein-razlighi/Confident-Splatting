import cv2
import os


def extract_frames(video_path, output_dir, frame_interval=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_dir, f"frame_{extracted_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames from the video.")


if __name__ == "__main__":
    video_path = "data/Eiffel_Tower.mkv"
    output_dir = "extracted_frames"
    frame_interval = 30  # Extract one frame every 30 frames
    extract_frames(video_path, output_dir, frame_interval)
