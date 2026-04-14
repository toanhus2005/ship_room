from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict

import cv2
from pydantic import ValidationError

from src.config import ToanConfig, build_shared_config, ensure_parent
from src.module1.video_input import iter_sampled_frames
from src.module2.person_tracker import PersonTracker

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


def load_config(config_path: Path) -> ToanConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return ToanConfig.model_validate(data)


def resolve_config(
    config_path: str | None,
    video_source: str | None,
    timezone: str | None,
    force_track_buffer: int | None = None,
    process_width: int | None = None,
) -> ToanConfig:
    cfg = load_config(Path(config_path)) if config_path else build_shared_config()
    if video_source:
        cfg.video.source = video_source
    if timezone:
        cfg.video.timezone = timezone
    if force_track_buffer is not None and int(force_track_buffer) > 0:
        cfg.tracking.track_buffer = int(force_track_buffer)
    if process_width is not None:
        cfg.stream.process_width = int(process_width)
    return cfg


def run_pipeline(config: ToanConfig) -> None:
    tracker = PersonTracker(
        detection_cfg=config.detection,
        tracking_cfg=config.tracking,
        zone_cfg=config.zone,
    )

    ensure_parent(config.output.events_jsonl)

    # Compute frame resize scale to match preview processing
    process_width = getattr(config.stream, 'process_width', 960)
    resize_scale = 1.0

    summary: Dict[int, Dict[str, int]] = {}
    write_buffer: list[str] = []
    write_batch_size = 20
    min_track_confidence = max(
        0.0,
        float(
            getattr(
                config.tracking,
                "repo_track_min_confidence",
                getattr(config.tracking, "repo_deepsort_min_confidence", 0.0),
            )
        ),
    )
    total_events = 0
    interrupted = False
    start_utc = datetime.now(UTC)
    event_file = config.output.events_jsonl.open("w", encoding="utf-8", buffering=1024 * 1024)

    def _update_summary(track_id: int, frame_index: int, in_package_zone: bool) -> None:
        state = summary.get(track_id)
        if state is None:
            summary[track_id] = {
                "first_frame": frame_index,
                "last_frame": frame_index,
                "zone_hits": 1 if in_package_zone else 0,
            }
            return
        state["last_frame"] = frame_index
        if in_package_zone:
            state["zone_hits"] += 1

    def _flush_buffer(force: bool = False) -> None:
        if not write_buffer:
            return
        if (not force) and len(write_buffer) < write_batch_size:
            return
        event_file.write("\n".join(write_buffer) + "\n")
        write_buffer.clear()
        # Keep file visible for near real-time readers.
        if force:
            event_file.flush()

    try:
        packets = iter_sampled_frames(
            video_cfg=config.video,
            stream_cfg=config.stream,
            start_utc=start_utc,
        )
        iterator = tqdm(packets, desc="Processing Video", unit="frame") if tqdm else packets

        for packet in iterator:
            # Resize frame before processing to match preview detection behavior
            frame_to_process = packet.frame
            if process_width > 0:
                h, w = packet.frame.shape[:2]
                if w > process_width:
                    resize_scale = process_width / float(w)
                    new_h = int(round(h * resize_scale))
                    frame_to_process = cv2.resize(packet.frame, (process_width, new_h), interpolation=cv2.INTER_LINEAR)
            
            events = tracker.process_frame(packet.frame_index, frame_to_process)

            for e in events:
                # Drop unstable tracks that cannot be matched back to a confident detection.
                if float(e.confidence) < min_track_confidence:
                    continue

                row = asdict(e)
                # packet timestamps are derived from video position/frame index, avoiding wall-clock drift.
                elapsed_seconds = (packet.timestamp_utc - start_utc).total_seconds()
                row["timestamp_local"] = packet.timestamp_local.isoformat()
                row["timestamp_utc"] = packet.timestamp_utc.isoformat()
                row["elapsed_seconds"] = round(float(elapsed_seconds), 3)
                write_buffer.append(json.dumps(row, ensure_ascii=False))
                _update_summary(e.track_id, e.frame_index, e.in_package_zone)
                total_events += 1

            _flush_buffer()
    except KeyboardInterrupt:
        interrupted = True
        print("[INFO] Pipeline interrupted by user.")
    finally:
        _flush_buffer(force=True)
        event_file.close()

        print("=== Tracking summary ===")
        for track_id in sorted(summary.keys()):
            s = summary[track_id]
            print(
                f"track_id={track_id} first={s['first_frame']} last={s['last_frame']} zone_hits={s['zone_hits']}"
            )
        print(f"Total events: {total_events}")
        print(f"Saved events: {config.output.events_jsonl}")
        if interrupted:
            print("[INFO] Saved partial results after interruption.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toan pipeline: video ingest + person tracking")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional path to JSON config. Leave empty to use shared defaults in src/config.py",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Optional video source override (file path / RTSP / camera index)",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="",
        help="Optional timezone override for timestamps",
    )
    parser.add_argument(
        "--force-track-buffer",
        type=int,
        default=0,
        help="Optional override for tracking.track_buffer (use 30 to match live preview behavior)",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=0,
        help="Optional frame width resize before detection (default 960 to match live preview, 0 = original size)",
    )
    args = parser.parse_args()

    try:
        cfg = resolve_config(
            config_path=args.config or None,
            video_source=args.video or None,
            timezone=args.timezone or None,
            force_track_buffer=args.force_track_buffer or None,
            process_width=args.process_width or None,
        )
        run_pipeline(cfg)
    except FileNotFoundError as e:
        raise SystemExit(f"Config not found: {e}")
    except ValidationError as e:
        raise SystemExit(f"Invalid config: {e}")


if __name__ == "__main__":
    main()
