import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "p3",
    Path(__file__).resolve().parents[1] / "src" / "p3.py",
)
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)


def test_clamp_frame_index_stays_in_bounds():
    assert p3.clamp_frame_index(-5, 10) == 0
    assert p3.clamp_frame_index(0, 10) == 0
    assert p3.clamp_frame_index(7, 10) == 7
    assert p3.clamp_frame_index(15, 10) == 9


def test_resolve_rois_uses_preprovided_values_without_prompting():
    prompts = []

    def fake_select_roi(frame, label):
        prompts.append(label)
        return (0, 0, 10, 10)

    roi1, roi2 = p3.resolve_rois(
        frame=None,
        roi1=(1, 2, 3, 4),
        roi2=(5, 6, 7, 8),
        select_roi_fn=fake_select_roi,
    )

    assert roi1 == (1, 2, 3, 4)
    assert roi2 == (5, 6, 7, 8)
    assert prompts == []


def test_resolve_rois_from_preview_uses_separate_frames_for_each_roi():
    preview_calls = []
    read_calls = []
    labels = []

    def fake_preview_frame_selection(cap, total_frames, fps):
        preview_calls.append((total_frames, fps))
        return len(preview_calls)

    def fake_read_frame(cap, frame_idx):
        read_calls.append(frame_idx)
        return True, {"frame_idx": frame_idx}

    def fake_select_roi(frame, label):
        labels.append(label)
        return (frame["frame_idx"], 0, 1, 1)

    roi1, roi2 = p3.resolve_rois_from_preview(
        cap=None,
        total_frames=10,
        fps=5,
        roi1=None,
        roi2=None,
        select_roi_fn=fake_select_roi,
        preview_frame_selection_fn=fake_preview_frame_selection,
        read_frame_fn=fake_read_frame,
    )

    assert roi1 == (1, 0, 1, 1)
    assert roi2 == (2, 0, 1, 1)
    assert labels == ["Select Webcam 1", "Select Webcam 2"]
    assert read_calls == [1, 2]


def test_process_video_creates_parent_directory_for_output(tmp_path, monkeypatch):
    class FakeCapture:
        def __init__(self):
            self._frame_index = 0

        def isOpened(self):
            return True

        def release(self):
            return None

        def get(self, prop):
            values = {
                p3.cv2.CAP_PROP_FPS: 10,
                p3.cv2.CAP_PROP_FRAME_COUNT: 1,
                p3.cv2.CAP_PROP_FRAME_WIDTH: 10,
                p3.cv2.CAP_PROP_FRAME_HEIGHT: 10,
            }
            return values[prop]

        def set(self, prop, value):
            return True

        def read(self):
            if self._frame_index == 0:
                self._frame_index += 1
                return True, np.zeros((10, 10, 3), dtype=np.uint8)
            return False, None

    monkeypatch.setattr(p3.cv2, "VideoCapture", lambda _: FakeCapture())

    output_path = tmp_path / "nested" / "results" / "intensities.csv"
    result = p3.process_video("dummy.mp4", output_path, roi1=(0, 0, 1, 1), roi2=(0, 0, 1, 1))

    assert result == output_path
    assert output_path.exists()
    assert output_path.parent.exists()
