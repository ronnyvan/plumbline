"""Halpe-26 keypoint schema (D-13).

RTMPose-m trained on body7 with the Halpe-26 head is the production backbone. It is
still RTMPose-m (D-04), but unlike COCO-17 it carries heels, toes, a neck point and a
hip-centre point. Feet are required for the foot-placement cues (LQ-03, LG-01); neck
and hip centre give a torso axis that does not depend on which shoulder is occluded.
"""

NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)
N_KEYPOINTS = len(NAMES)
INDEX = {name: i for i, name in enumerate(NAMES)}

NECK = INDEX["neck"]
HIP_CENTRE = INDEX["hip"]

# Interior joint angles as (proximal, joint, distal) triplets, per side.
TRIPLETS = {
    "elbow": {"left": ("left_shoulder", "left_elbow", "left_wrist"),
              "right": ("right_shoulder", "right_elbow", "right_wrist")},
    "knee": {"left": ("left_hip", "left_knee", "left_ankle"),
             "right": ("right_hip", "right_knee", "right_ankle")},
    "hip": {"left": ("left_shoulder", "left_hip", "left_knee"),
            "right": ("right_shoulder", "right_hip", "right_knee")},
    "shoulder": {"left": ("left_hip", "left_shoulder", "left_elbow"),
                 "right": ("right_hip", "right_shoulder", "right_elbow")},
}
