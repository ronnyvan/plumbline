"""Real RTMPose backend. Runs only with `make test-pose` (downloads ~150 MB of weights, needs a GPU)."""
import numpy as np
import pytest

pytestmark = pytest.mark.pose


@pytest.fixture(scope="module")
def backend():
    from core.pose.rtmpose import RTMPoseBackend
    return RTMPoseBackend(device="cuda")


def test_runs_on_cuda_and_records_model_hashes(backend):
    assert backend.meta["provider"] == ["CUDAExecutionProvider"]
    assert len(backend.meta["pose_sha256"]) == 64
    assert backend.stage_spec().config["provider"] == ["CUDAExecutionProvider"]


def test_empty_scene_produces_no_skeleton(backend):
    # rtmlib, given zero boxes, would estimate a pose over the whole image.
    frames = ((i, np.zeros((720, 1280, 3), np.uint8)) for i in range(5))
    pose = backend.infer(frames, n_frames=5, width=1280, height=720, fps=30.0)
    assert len(pose.frame) == 0
