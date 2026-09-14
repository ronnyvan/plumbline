"""Video ingest (§5.1, D-15)."""
import cv2
import numpy as np
import pytest

from core.ingest.video import IngestRejected, ingest, iter_frames


def _video(path, fps, seconds, size=(320, 180), still_head_s=0.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV build cannot write mp4v"
    for i in range(int(round(fps * seconds))):
        t = i / fps
        frame = np.full((size[1], size[0], 3), 40, np.uint8)
        x = 40 if t < still_head_s else int(40 + 150 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)))
        cv2.rectangle(frame, (x, 50), (x + 60, 130), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_sixty_fps_is_resampled_to_thirty_by_timestamp(tmp_path):
    r = ingest(_video(tmp_path / "a.mp4", 60, 4))
    assert r.native_fps == pytest.approx(60)
    assert r.n_frames == 120
    assert r.source_index[:5] == [0, 2, 4, 6, 8]
    assert r.timestamps_s[1] == pytest.approx(1 / 30, abs=1e-3)


def test_low_frame_rate_is_upsampled_without_inventing_frames(tmp_path):
    r = ingest(_video(tmp_path / "b.mp4", 24, 3))
    assert r.n_frames == 90
    assert max(r.source_index) < r.source_frames
    assert len(set(r.source_index)) == r.source_frames


def test_longest_side_is_capped(tmp_path):
    r = ingest(_video(tmp_path / "c.mp4", 30, 2.5, size=(1920, 1080)))
    assert (r.width, r.height) == (1080, 608)
    _, frame = next(iter_frames(tmp_path / "c.mp4", r, (0, 0)))
    assert frame.shape[:2] == (608, 1080)


def test_activity_window_skips_static_lead_in(tmp_path):
    r = ingest(_video(tmp_path / "d.mp4", 30, 6, still_head_s=3.0))
    assert 50 <= r.active[0] <= 70          # motion at 90, minus 30 frames of padding


def test_iter_frames_yields_exactly_the_window(tmp_path):
    path = _video(tmp_path / "e.mp4", 30, 3)
    r = ingest(path)
    assert [k for k, _ in iter_frames(path, r, (10, 14))] == [10, 11, 12, 13, 14]


@pytest.mark.parametrize("fps,seconds,reason", [(30, 1, "too_short"), (2, 95, "too_long")])
def test_duration_limits(tmp_path, fps, seconds, reason):
    with pytest.raises(IngestRejected) as err:
        ingest(_video(tmp_path / "f.mp4", fps, seconds, size=(64, 64)))
    assert err.value.reason == reason


def test_unreadable_file_is_rejected(tmp_path):
    bad = tmp_path / "g.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(IngestRejected):
        ingest(bad)
