"""Analyse one real clip from the command line.

    python -m core.cli analyze clip.mp4 --exercise lat_pulldown --station S-LAT [--out report.json]

Consent is the caller's responsibility here: this command is for your own footage. Clips of
anyone else go through data/clips/ with a labels file that records consent (§9.2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.ingest.video import IngestRejected
from core.pipeline import analyze_video
from core.pose.rtmpose import PoseBackendError, RTMPoseBackend
from core.registry import default_registry
from core.staging import ArtifactStore

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    registry = default_registry()
    ap = argparse.ArgumentParser(prog="core.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse one clip and write a report")
    a.add_argument("video", type=Path)
    a.add_argument("--exercise", required=True, choices=sorted(registry.exercises))
    a.add_argument("--station", required=True, choices=sorted(registry.stations),
                   help="where the phone was; cannot be verified until M5")
    a.add_argument("--out", type=Path)
    a.add_argument("--cpu", action="store_true", help="run pose on CPU (slower, different numbers)")
    args = ap.parse_args(argv)

    try:
        backend = RTMPoseBackend(device="cpu" if args.cpu else "cuda")
        report = analyze_video(args.video, args.exercise, station_requested=args.station,
                               store=ArtifactStore(ROOT / "artifacts"), backend=backend, registry=registry)
    except IngestRejected as e:
        print(f"Clip rejected: {e.reason} {e.detail}")
        return 2
    except PoseBackendError as e:
        print(str(e))
        return 3

    out = args.out or args.video.with_suffix(".report.json")
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    s = report["set_summary"] or {}
    print(f"{report['exercise']['name']}: {report['status']}")
    if report["submission_abstention"]:
        print(f"  {report['submission_abstention']['copy']}")
    for key in ("reps", "partial_reps", "tut", "rom_cv", "fatigue_slope_rom_rel"):
        m = s.get(key)
        if m:
            value = m["value"] if m["available"] else f"unavailable ({m['reason']})"
            print(f"  {key:22s} {value} {m['units'] if m['available'] else ''}")
    for ab in report["abstained_cues"]:
        print(f"  not assessed {ab['cue_id']}: {ab['reason']}")
    for note in report["exercise"]["notes"]:
        print(f"  note: {note}")
    ing = report["diagnostics"].get("ingest", {})
    print(f"  pose on {ing.get('provider')} in {ing.get('pose_inference_s')} s; report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
