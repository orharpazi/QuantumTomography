from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2

ROI = Tuple[int, int, int, int]


def clamp_frame_index(frame_idx: int, total_frames: int) -> int:
    total_frames = max(1, int(total_frames))
    return max(0, min(int(frame_idx), total_frames - 1))


def read_frame(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    return cap.read()


def preview_frame_selection(cap, total_frames: int, fps: float):
    current_idx = 0
    playing = False
    frames_per_0_2s = max(1, int(0.2 * fps))
    frame_interval_ms = max(1, int(1000 / max(fps, 1)))
    print("Preview controls: a/d skip ±0.2s, space toggles play, enter confirms, q quits.")

    try:
        while True:
            current_idx = clamp_frame_index(current_idx, total_frames)
            ret, frame = read_frame(cap, current_idx)
            if not ret:
                break

            cv2.imshow("Frame Preview", frame)
            delay_ms = 0 if not playing else frame_interval_ms
            key = cv2.waitKeyEx(delay_ms) & 0xFF

            if key == 255 or key == -1:
                if playing:
                    current_idx += 1
                continue

            if key in (ord("q"), ord("Q"), 27):
                return None

            if key == ord(" "):
                playing = not playing
                continue

            if key == ord("a"):
                current_idx -= frames_per_0_2s
                continue

            if key == ord("d"):
                current_idx += frames_per_0_2s
                continue

            if key in (13, 10):
                return current_idx
    finally:
        cv2.destroyAllWindows()

    return None


def select_roi(frame, label: str):
    print(f"{label}: drag a rectangle and press SPACE/ENTER, or press c to cancel.")
    roi = cv2.selectROI(label, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(label)

    if roi is None:
        return None

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def resolve_rois(frame, roi1: Optional[ROI] = None, roi2: Optional[ROI] = None, select_roi_fn=None):
    if select_roi_fn is None:
        select_roi_fn = select_roi

    resolved_roi1 = roi1 if roi1 is not None else select_roi_fn(frame, "Select Webcam 1")
    resolved_roi2 = roi2 if roi2 is not None else select_roi_fn(frame, "Select Webcam 2")

    if resolved_roi1 is None:
        print("No valid ROI selected for Webcam 1; using the full frame.")
        resolved_roi1 = (0, 0, frame.shape[1], frame.shape[0])

    if resolved_roi2 is None:
        print("No valid ROI selected for Webcam 2; using the full frame.")
        resolved_roi2 = (0, 0, frame.shape[1], frame.shape[0])

    return resolved_roi1, resolved_roi2


def resolve_rois_from_preview(
    cap,
    total_frames: int,
    fps: float,
    roi1: Optional[ROI] = None,
    roi2: Optional[ROI] = None,
    select_roi_fn=None,
    preview_frame_selection_fn=None,
    read_frame_fn=None,
):
    if select_roi_fn is None:
        select_roi_fn = select_roi
    if preview_frame_selection_fn is None:
        preview_frame_selection_fn = preview_frame_selection
    if read_frame_fn is None:
        read_frame_fn = read_frame

    def pick_roi(label: str, current_roi: Optional[ROI]):
        if current_roi is not None:
            return current_roi

        preview_frame_idx = preview_frame_selection_fn(cap, total_frames, fps)
        if preview_frame_idx is None:
            return None

        ret, frame = read_frame_fn(cap, preview_frame_idx)
        if not ret:
            return None

        return select_roi_fn(frame, label)

    resolved_roi1 = pick_roi("Select Webcam 1", roi1)
    if resolved_roi1 is None and roi1 is None:
        return None, roi2

    if cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    resolved_roi2 = pick_roi("Select Webcam 2", roi2)
    if resolved_roi2 is None and roi2 is None:
        return resolved_roi1, None

    return resolved_roi1, resolved_roi2
