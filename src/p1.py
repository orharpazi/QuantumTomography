import csv
import importlib.util
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "data" / "raw" / "p1"
VIDEO_PATHS = list(VIDEO_DIR.glob("*.mp4"))


def _load_interactive_module():
    module_path = Path(__file__).with_name("interactive.py")
    spec = importlib.util.spec_from_file_location("interactive", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load interactive module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interactive = _load_interactive_module()

clamp_frame_index = interactive.clamp_frame_index
read_frame = interactive.read_frame
preview_frame_selection = interactive.preview_frame_selection
select_roi = interactive.select_roi
resolve_rois = interactive.resolve_rois
resolve_rois_from_preview = interactive.resolve_rois_from_preview


def _resolve_missing_roi(roi, frame_width: int, frame_height: int):
    if roi is None:
        return (0, 0, frame_width, frame_height)
    return roi


def process_video(
    video_path,
    output_filename,
    roi1=(374, 179, 16, 20),
    roi2=(666, 141, 20, 18),
    roi3=(315, 522, 21, 21),
    roi4=(727, 409, 27, 28)
):
    video_path = Path(video_path)
    output_filename = Path(output_filename)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Loaded: {total_frames} frames at {fps} FPS")

        if roi1 is not None and roi2 is not None and roi3 is not None and roi4 is not None:
            print("Skipping frame preview because all ROIs were pre-provided.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, _ = cap.read()
            if not ret:
                print("Could not read the first frame for ROI processing.")
                return None
        else:
            roi1, roi2 = resolve_rois_from_preview(
                cap,
                total_frames,
                fps,
                roi1=roi1,
                roi2=roi2,
            )
            roi3, roi4 = resolve_rois_from_preview(
                cap,
                total_frames,
                fps,
                roi1=roi3,
                roi2=roi4,
            )
            if roi1 is None and roi2 is None and roi3 is None and roi4 is None:
                print("Preview cancelled; exiting.")
                return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi1 = _resolve_missing_roi(roi1, frame_width, frame_height)
        roi2 = _resolve_missing_roi(roi2, frame_width, frame_height)
        roi3 = _resolve_missing_roi(roi3, frame_width, frame_height)
        roi4 = _resolve_missing_roi(roi4, frame_width, frame_height)

        print(f"Using ROI for Webcam 1: {roi1}")
        print(f"Using ROI for Webcam 2: {roi2}")
        print(f"Using ROI for Webcam 3: {roi3}")
        print(f"Using ROI for Webcam 4: {roi4}")

        x1, y1, w1, h1 = roi1
        x2, y2, w2, h2 = roi2
        x3, y3, w3, h3 = roi3
        x4, y4, w4, h4 = roi4

        data_log = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            cam1_crop = frame[int(y1):int(y1 + h1), int(x1):int(x1 + w1)]
            cam2_crop = frame[int(y2):int(y2 + h2), int(x2):int(x2 + w2)]
            cam3_crop = frame[int(y3):int(y3 + h3), int(x3):int(x3 + w3)]
            cam4_crop = frame[int(y4):int(y4 + h4), int(x4):int(x4 + w4)]

            if cam1_crop.size == 0 or cam2_crop.size == 0 or cam3_crop.size == 0 or cam4_crop.size == 0:
                print("Encountered an empty crop from the selected ROI; skipping frame.")
                frame_idx += 1
                continue

            gray1 = cv2.cvtColor(cam1_crop, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(cam2_crop, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(cam3_crop, cv2.COLOR_BGR2GRAY)
            gray4 = cv2.cvtColor(cam4_crop, cv2.COLOR_BGR2GRAY)
            avg1 = np.mean(gray1)
            avg2 = np.mean(gray2)
            avg3 = np.mean(gray3)
            avg4 = np.mean(gray4)
            data_log.append([timestamp, avg1, avg2, avg3, avg4])

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...", end="\r")

        print("\nProcessing complete.")

        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with output_filename.open(mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Webcam_1_Average", "Webcam_2_Average", "Webcam_3_Average", "Webcam_4_Average"])
            writer.writerows(data_log)

        print(f"Data saved to {output_filename}")
        return output_filename
    finally:
        cap.release()


def main():
    for video_path in VIDEO_PATHS:
        process_video(video_path, output_filename=ROOT_DIR / "data" / "p1" / f"{video_path.stem}_intensities.csv")


if __name__ == "__main__":
    main()
