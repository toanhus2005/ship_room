from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import List

from pydantic import ValidationError

from src.config import ToanConfig, ensure_parent
from src.module1.video_input import iter_sampled_frames
from src.module2.person_tracker import PersonTrackEvent, PersonTracker, summarize_trajectories


def load_config(config_path: Path) -> ToanConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return ToanConfig.model_validate(data)


def run_pipeline(config: ToanConfig) -> None:
    tracker = PersonTracker(
        detection_cfg=config.detection,
        tracking_cfg=config.tracking,
        zone_cfg=config.zone,
    )

    ensure_parent(config.output.events_jsonl)

    all_events: List[PersonTrackEvent] = []
    start_utc = datetime.now(UTC)

    with config.output.events_jsonl.open("w", encoding="utf-8") as f:
        for packet in iter_sampled_frames(
            video_cfg=config.video,
            stream_cfg=config.stream,
            start_utc=start_utc,
        ):
            events = tracker.process_frame(packet.frame_index, packet.frame)

            for e in events:
                row = asdict(e)
                elapsed_seconds = (packet.timestamp_utc - start_utc).total_seconds()
                row["timestamp_local"] = packet.timestamp_local.isoformat()
                row["timestamp_utc"] = packet.timestamp_utc.isoformat()
                row["elapsed_seconds"] = round(float(elapsed_seconds), 3)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            all_events.extend(events)

    summary = summarize_trajectories(all_events)
    print("=== Tracking summary ===")
    for track_id, s in summary.items():
        print(
            f"track_id={track_id} first={s['first_frame']} last={s['last_frame']} zone_hits={s['zone_hits']}"
        )
    print(f"Saved events: {config.output.events_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toan pipeline: video ingest + person tracking")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toan_config.example.json",
        help="Path to JSON config",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(Path(args.config))
        run_pipeline(cfg)
    except FileNotFoundError as e:
        raise SystemExit(f"Config not found: {e}")
    except ValidationError as e:
        raise SystemExit(f"Invalid config: {e}")


if __name__ == "__main__":
    main()
