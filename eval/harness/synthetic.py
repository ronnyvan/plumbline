"""Synthetic lifts with exact ground truth (spec §11.7).

A 2D forward-kinematics skeleton performs a set from a lateral station (S-LAT) and emits
the same `PoseFrames` the real backend produces, so every downstream stage — tracking,
features, segmentation, metrics — runs unmodified on it.

What this is for: proving the signal path is correct. What it is not: evidence about
real video. Any number computed on synthetic clips is labelled `corpus: synthetic` and is
never admissible for a tier gate (§2.3) or a success criterion (§1.5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from core.keypoints import INDEX, N_KEYPOINTS
from core.pose.types import PoseFrames, SubjectTrack

GENERATOR_VERSION = "1"


@dataclass(frozen=True)
class Template:
    exercise_id: str
    drive: str                  # "elbow" | "knee"
    start_deg: float            # interior angle at the start position
    turn_deg: float             # interior angle at the turnaround
    torso_deg: float            # lean back (+) or forward (-) from vertical
    upper_arm_deg: float        # absolute, degrees from image-up toward facing direction
    upper_arm_turn_deg: float | None
    elbow_deg: float
    thigh_deg: float
    knee_deg: float


TEMPLATES = {
    "lat_pulldown": Template("lat_pulldown", "elbow", 165, 75, 15, 15, 175, 165, 90, 90),
    "preacher_curl": Template("preacher_curl", "elbow", 160, 50, -15, 135, None, 160, 90, 90),
    "leg_extension": Template("leg_extension", "knee", 90, 170, 10, 175, None, 160, 90, 90),
    # Starts at lockout: the first phase is eccentric (D-16).
    "leg_press_quad_bias": Template("leg_press_quad_bias", "knee", 165, 85, 45, 150, None, 150, 45, 165),
    "tricep_pushdown": Template("tricep_pushdown", "elbow", 70, 170, -10, 175, None, 70, 180, 175),
}


@dataclass(frozen=True)
class SynthParams:
    exercise_id: str = "lat_pulldown"
    n_reps: int = 8
    fps: float = 30.0
    concentric_s: float = 1.2
    eccentric_s: float = 2.0
    turn_pause_s: float = 0.3
    start_pause_s: float = 0.5
    tempo_jitter: float = 0.05
    lead_in_s: float = 1.5
    lead_out_s: float = 1.5
    rom_scale: float = 1.0               # fraction of the template's full range used
    rom_decay_per_rep: float = 0.0       # fraction of ROM lost per rep
    torso_sway_deg: float = 2.0          # natural motion on every rep
    swing_deg: float = 0.0               # extra swing on `swing_reps`
    swing_reps: tuple[int, ...] = ()     # zero-based rep indices
    noise_px: float = 1.5
    near_score: float = 0.90
    far_score: float = 0.60
    miss_rate: float = 0.0               # detector misses the subject on this fraction of frames
    partial_first: bool = False          # clip begins mid-rep
    bystander: bool = False
    image_size: tuple[int, int] = (1080, 1920)   # portrait phone, (width, height)
    torso_px: float = 320.0
    hip_xy: tuple[float, float] = (500.0, 1050.0)
    seed: int = 0


@dataclass(frozen=True)
class TruthRep:
    start: int
    turn_arrive: int
    turn_leave: int
    end: int
    first_phase: str
    rom_deg: float
    concentric_s: float
    eccentric_s: float
    torso_swing_deg: float


@dataclass(frozen=True)
class GroundTruth:
    exercise_id: str
    fps: float
    n_frames: int
    reps: tuple[TruthRep, ...]
    partial_reps: int
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticClip:
    pose: PoseFrames
    track: SubjectTrack
    truth: GroundTruth


def _u(deg: np.ndarray) -> np.ndarray:
    r = np.radians(deg)
    return np.stack([np.sin(r), -np.cos(r)], axis=-1)


def _ease(u: np.ndarray) -> np.ndarray:
    return (1.0 - np.cos(np.pi * np.clip(u, 0.0, 1.0))) / 2.0


def _skeleton(tpl: Template, drive: np.ndarray, progress: np.ndarray, torso: np.ndarray,
              L: float, hip: np.ndarray) -> np.ndarray:
    """Near-side Halpe-26 keypoints, shape (T, 26, 2)."""
    t = len(drive)
    kp = np.zeros((t, N_KEYPOINTS, 2))
    t_dir = -torso
    H = np.broadcast_to(hip, (t, 2))
    N = H + L * _u(t_dir)
    head = N + 0.30 * L * _u(t_dir)
    S = N + 0.12 * L * _u(t_dir + 180)

    upper = np.full(t, tpl.upper_arm_deg, float)
    if tpl.upper_arm_turn_deg is not None:
        upper = tpl.upper_arm_deg + (tpl.upper_arm_turn_deg - tpl.upper_arm_deg) * progress
    elbow = drive if tpl.drive == "elbow" else np.full(t, tpl.elbow_deg, float)
    E = S + 0.55 * L * _u(upper)
    W = E + 0.50 * L * _u(upper - (180 - elbow))

    thigh = np.full(t, tpl.thigh_deg, float)
    knee = drive if tpl.drive == "knee" else np.full(t, tpl.knee_deg, float)
    K = H + 0.80 * L * _u(thigh)
    shank = thigh + (180 - knee)
    A = K + 0.80 * L * _u(shank)

    kp[:, INDEX["neck"]], kp[:, INDEX["hip"]], kp[:, INDEX["head"]] = N, H, head
    kp[:, INDEX["nose"]] = head + 0.08 * L * _u(t_dir + 90)
    for side in ("left", "right"):
        kp[:, INDEX[f"{side}_eye"]] = head + 0.06 * L * _u(t_dir + 60)
        kp[:, INDEX[f"{side}_ear"]] = head + 0.03 * L * _u(t_dir - 90)
        kp[:, INDEX[f"{side}_shoulder"]] = S
        kp[:, INDEX[f"{side}_elbow"]] = E
        kp[:, INDEX[f"{side}_wrist"]] = W
        kp[:, INDEX[f"{side}_hip"]] = H
        kp[:, INDEX[f"{side}_knee"]] = K
        kp[:, INDEX[f"{side}_ankle"]] = A
        kp[:, INDEX[f"{side}_heel"]] = A + 0.07 * L * _u(shank + 90)
        kp[:, INDEX[f"{side}_big_toe"]] = A + 0.25 * L * _u(shank - 90)
        kp[:, INDEX[f"{side}_small_toe"]] = A + 0.22 * L * _u(shank - 80)
    return kp


def _bbox(kp: np.ndarray, sc: np.ndarray) -> np.ndarray:
    ok = sc > 0.3
    pts = kp[ok] if ok.any() else kp
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my], np.float32)


def generate(p: SynthParams) -> SyntheticClip:
    tpl = TEMPLATES[p.exercise_id]
    rng = np.random.default_rng(p.seed)
    fps = p.fps

    # Which phase moves from start to turnaround, in raw-angle terms.
    from core.registry import default_registry
    concentric_dir = default_registry().exercises[p.exercise_id].rep_phase.concentric
    raw_first = "increasing" if tpl.turn_deg > tpl.start_deg else "decreasing"
    first_phase = "concentric" if raw_first == concentric_dir else "eccentric"
    first_s, second_s = (p.concentric_s, p.eccentric_s) if first_phase == "concentric" \
        else (p.eccentric_s, p.concentric_s)

    total_reps = p.n_reps + (1 if p.partial_first else 0)
    events = []   # (t_start, t_turn_arrive, t_turn_leave, t_end, rom_scale, swing)
    t = p.lead_in_s
    for k in range(total_reps):
        j = 1.0 + rng.uniform(-p.tempo_jitter, p.tempo_jitter, size=3)
        a = t
        b = a + first_s * j[0]
        c = b + p.turn_pause_s * j[1]
        d = c + second_s * j[2]
        rep_no = k - (1 if p.partial_first else 0)
        scale = p.rom_scale * (1.0 - p.rom_decay_per_rep * max(rep_no, 0))
        swing = p.torso_sway_deg + (p.swing_deg if rep_no in p.swing_reps else 0.0)
        events.append((a, b, c, d, scale, swing))
        t = d + p.start_pause_s
    duration = t - p.start_pause_s + p.lead_out_s
    crop = events[0][1] if p.partial_first else 0.0

    n = int(round((duration - crop) * fps))
    times = crop + np.arange(n) / fps
    progress = np.zeros(n)
    scale = np.ones(n)
    swing = np.zeros(n)
    for a, b, c, d, s, sw in events:
        up = (times >= a) & (times < b)
        hold = (times >= b) & (times < c)
        down = (times >= c) & (times < d)
        progress[up] = _ease((times[up] - a) / (b - a))
        progress[hold] = 1.0
        progress[down] = 1.0 - _ease((times[down] - c) / (d - c))
        active = (times >= a) & (times < d)
        scale[active], swing[active] = s, sw

    drive = tpl.start_deg + (tpl.turn_deg - tpl.start_deg) * scale * progress
    torso = tpl.torso_deg + swing * progress
    hip = np.array(p.hip_xy)
    near = _skeleton(tpl, drive, progress, torso, p.torso_px, hip)
    far = near + np.array([4.0, -3.0]) * (p.torso_px / 320.0)

    kp = near.copy()
    sc = np.full((n, N_KEYPOINTS), p.near_score)
    for name in ("right_shoulder", "right_elbow", "right_wrist", "right_hip",
                 "right_knee", "right_ankle", "right_eye", "right_ear",
                 "right_heel", "right_big_toe", "right_small_toe"):
        i = INDEX[name]
        kp[:, i] = far[:, i]
        sc[:, i] = p.far_score
    kp = kp + rng.normal(0.0, p.noise_px, kp.shape)
    sc = np.clip(sc + rng.normal(0.0, 0.03, sc.shape), 0.0, 1.0)

    present = rng.random(n) >= p.miss_rate
    track_kp = np.where(present[:, None, None], kp, np.nan).astype(np.float32)
    track_sc = np.where(present[:, None], sc, 0.0).astype(np.float32)
    w, h = p.image_size
    track = SubjectTrack(fps, w, h, track_kp, track_sc, present)

    frames, boxes, kps, scs = [], [], [], []
    for i in np.flatnonzero(present):
        frames.append(i)
        boxes.append(_bbox(kp[i], sc[i]))
        kps.append(kp[i])
        scs.append(sc[i])
    if p.bystander:
        by_start, by_len = int(0.2 * n), int(0.4 * n)
        tpl_by = TEMPLATES["tricep_pushdown"]
        for i in range(by_start, min(n, by_start + by_len)):
            x = 50 + (w - 100) * (i - by_start) / by_len
            bk = _skeleton(tpl_by, np.array([175.0]), np.array([0.0]), np.array([0.0]),
                           p.torso_px * 0.45, np.array([x, 700.0]))[0]
            bk = bk + rng.normal(0, p.noise_px, bk.shape)
            bs = np.full(N_KEYPOINTS, 0.8)
            frames.append(i)
            boxes.append(_bbox(bk, bs))
            kps.append(bk)
            scs.append(bs)
    order = np.argsort(np.array(frames), kind="stable")
    pose = PoseFrames(
        fps=fps, width=w, height=h, n_frames=n,
        frame=np.array(frames, np.int32)[order],
        bbox=np.array(boxes, np.float32).reshape(-1, 4)[order],
        kp=np.array(kps, np.float32).reshape(-1, N_KEYPOINTS, 2)[order],
        kp_score=np.array(scs, np.float32).reshape(-1, N_KEYPOINTS)[order],
        meta={"generator": "synthetic", "generator_version": GENERATOR_VERSION},
    )

    def fr(time_s: float) -> int:
        return int(round((time_s - crop) * fps))

    reps = []
    for k, (a, b, c, d, s, sw) in enumerate(events):
        if p.partial_first and k == 0:
            continue
        conc = (b - a) if first_phase == "concentric" else (d - c)
        ecc = (d - c) if first_phase == "concentric" else (b - a)
        reps.append(TruthRep(fr(a), fr(b), fr(c), fr(d), first_phase,
                             abs(tpl.turn_deg - tpl.start_deg) * s, conc, ecc, sw))
    truth = GroundTruth(p.exercise_id, fps, n, tuple(reps), 1 if p.partial_first else 0,
                        {**asdict(p), "generator_version": GENERATOR_VERSION})
    return SyntheticClip(pose, track, truth)


def with_(p: SynthParams, **changes) -> SynthParams:
    return replace(p, **changes)
