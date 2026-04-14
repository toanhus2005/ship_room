from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PersonAppearanceEvent:
    start_frame: int
    end_frame: int
    start_second: float
    end_second: float
    duration_second: float
    max_confidence: float
    detections: int


@dataclass
class PersonTrackAppearanceEvent:
    track_id: int
    segment_index: int
    first_frame: int
    last_frame: int
    first_second: float
    last_second: float
    duration_second: float
    first_timestamp_utc: str
    last_timestamp_utc: str
    first_timestamp_local: str
    last_timestamp_local: str
    detections: int
    zone_hits: int
    max_confidence: float


def _round3(value: float) -> float:
    return round(float(value), 3)


class PersonEventTour:
    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        batch_size: int = 8,
        device: str = "auto",
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou = iou
        self.batch_size = max(1, int(batch_size))
        self.device = self._resolve_device(device)
        self.use_half = bool(self.device != "cpu")

    @staticmethod
    def _resolve_device(device: str) -> str:
        req = str(device).strip().lower()
        if req in {"", "auto"}:
            return "0" if torch.cuda.is_available() else "cpu"
        return req

    def _detect_person(self, frame: np.ndarray) -> Tuple[bool, float]:
        return self._detect_person_batch([frame])[0]

    def _detect_person_batch(self, frames: List[np.ndarray]) -> List[Tuple[bool, float]]:
        if not frames:
            return []

        results = self.model.predict(
            source=frames,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            half=self.use_half,
            verbose=False,
        )

        out: List[Tuple[bool, float]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.conf is None or len(boxes) == 0:
                out.append((False, 0.0))
                continue
            max_conf = float(boxes.conf.max().item())
            out.append((True, max_conf))
        return out

    def scan(
        self,
        video_path: str,
        scan_fps: float = 2.0,
        absence_tolerance_sec: float = 2.0,
        min_event_sec: float = 0.8,
    ) -> List[PersonAppearanceEvent]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0:
                native_fps = 25.0

            stride = max(1, int(round(native_fps / max(scan_fps, 0.1))))
            absent_frames_limit = max(1, int(math.ceil(absence_tolerance_sec * scan_fps)))

            events: List[PersonAppearanceEvent] = []

            open_start_frame: Optional[int] = None
            open_start_sec: Optional[float] = None
            last_seen_frame: Optional[int] = None
            last_seen_sec: Optional[float] = None
            open_max_conf = 0.0
            open_detections = 0
            absent_samples = 0

            batch_frames: List[np.ndarray] = []
            batch_meta: List[Tuple[int, float]] = []

            frame_index = 0

            def finalize_open_event() -> None:
                nonlocal open_start_frame
                nonlocal open_start_sec
                nonlocal last_seen_frame
                nonlocal last_seen_sec
                nonlocal open_max_conf
                nonlocal open_detections
                nonlocal absent_samples

                if (
                    open_start_frame is None
                    or open_start_sec is None
                    or last_seen_frame is None
                    or last_seen_sec is None
                ):
                    return

                duration = last_seen_sec - open_start_sec
                if duration >= min_event_sec:
                    events.append(
                        PersonAppearanceEvent(
                            start_frame=open_start_frame,
                            end_frame=int(last_seen_frame),
                            start_second=float(open_start_sec),
                            end_second=float(last_seen_sec),
                            duration_second=float(duration),
                            max_confidence=float(open_max_conf),
                            detections=int(open_detections),
                        )
                    )

                open_start_frame = None
                open_start_sec = None
                last_seen_frame = None
                last_seen_sec = None
                open_max_conf = 0.0
                open_detections = 0
                absent_samples = 0

            def process_batch() -> None:
                nonlocal open_start_frame
                nonlocal open_start_sec
                nonlocal last_seen_frame
                nonlocal last_seen_sec
                nonlocal open_max_conf
                nonlocal open_detections
                nonlocal absent_samples

                if not batch_frames:
                    return

                batch_results = self._detect_person_batch(batch_frames)
                for (sample_frame_index, t_sec), (has_person, best_conf) in zip(batch_meta, batch_results):
                    if has_person:
                        absent_samples = 0
                        if open_start_frame is None:
                            open_start_frame = sample_frame_index
                            open_start_sec = t_sec
                            open_max_conf = best_conf
                            open_detections = 1
                        else:
                            open_max_conf = max(open_max_conf, best_conf)
                            open_detections += 1

                        last_seen_frame = sample_frame_index
                        last_seen_sec = t_sec
                    elif open_start_frame is not None:
                        absent_samples += 1
                        if absent_samples >= absent_frames_limit:
                            finalize_open_event()

                batch_frames.clear()
                batch_meta.clear()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % stride == 0:
                    t_sec = frame_index / native_fps
                    batch_frames.append(frame)
                    batch_meta.append((frame_index, t_sec))
                    if len(batch_frames) >= self.batch_size:
                        process_batch()

                frame_index += 1

            process_batch()
            finalize_open_event()
            return events
        finally:
            cap.release()


def build_track_appearance_events(
    tracks_jsonl: Path,
    gap_seconds_limit: float = 2.0,
    min_event_sec: float = 2.0,
    min_detections: int = 3,
) -> List[PersonTrackAppearanceEvent]:
    if not tracks_jsonl.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracks_jsonl}")

    rows_by_track: Dict[int, List[Dict[str, Any]]] = {}

    with tracks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue

            try:
                data = json.loads(row)
            except json.JSONDecodeError:
                continue

            track_id = int(data.get("track_id", -1))
            if track_id < 0:
                continue

            frame_index = int(data.get("frame_index", 0))
            elapsed_seconds = float(data.get("elapsed_seconds", 0.0))
            timestamp_utc = str(data.get("timestamp_utc", ""))
            timestamp_local = str(data.get("timestamp_local", ""))
            confidence = float(data.get("confidence", 0.0))
            in_zone = bool(data.get("in_package_zone", False))

            if elapsed_seconds < 0:
                continue

            rows_by_track.setdefault(track_id, []).append(
                {
                    "frame_index": frame_index,
                    "elapsed_seconds": elapsed_seconds,
                    "timestamp_utc": timestamp_utc,
                    "timestamp_local": timestamp_local,
                    "confidence": confidence,
                    "in_zone": in_zone,
                }
            )

    events: List[PersonTrackAppearanceEvent] = []
    for track_id in sorted(rows_by_track.keys()):
        rows = sorted(
            rows_by_track[track_id],
            key=lambda r: (float(r["elapsed_seconds"]), int(r["frame_index"])),
        )

        raw_segments: List[List[Dict[str, Any]]] = []
        current_segment: List[Dict[str, Any]] = []

        for row in rows:
            if not current_segment:
                current_segment.append(row)
                continue

            gap = float(row["elapsed_seconds"]) - float(current_segment[-1]["elapsed_seconds"])
            if gap > gap_seconds_limit:
                raw_segments.append(current_segment)
                current_segment = [row]
            else:
                current_segment.append(row)

        if current_segment:
            raw_segments.append(current_segment)

        kept_segments: List[List[Dict[str, Any]]] = []
        for segment in raw_segments:
            first = segment[0]
            last = segment[-1]
            duration_second = max(0.0, float(last["elapsed_seconds"]) - float(first["elapsed_seconds"]))
            if duration_second < min_event_sec:
                continue
            if len(segment) < min_detections:
                continue
            kept_segments.append(segment)

        for segment_index, segment in enumerate(kept_segments, start=1):
            first = segment[0]
            last = segment[-1]
            duration_second = max(0.0, float(last["elapsed_seconds"]) - float(first["elapsed_seconds"]))
            zone_hits = sum(1 for r in segment if bool(r["in_zone"]))
            max_confidence = max(float(r["confidence"]) for r in segment)

            events.append(
                PersonTrackAppearanceEvent(
                    track_id=track_id,
                    segment_index=segment_index,
                    first_frame=int(first["frame_index"]),
                    last_frame=int(last["frame_index"]),
                    first_second=_round3(float(first["elapsed_seconds"])),
                    last_second=_round3(float(last["elapsed_seconds"])),
                    duration_second=_round3(duration_second),
                    first_timestamp_utc=str(first["timestamp_utc"]),
                    last_timestamp_utc=str(last["timestamp_utc"]),
                    first_timestamp_local=str(first["timestamp_local"]),
                    last_timestamp_local=str(last["timestamp_local"]),
                    detections=int(len(segment)),
                    zone_hits=int(zone_hits),
                    max_confidence=_round3(max_confidence),
                )
            )

    return sorted(events, key=lambda e: (e.first_second, e.track_id, e.segment_index))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast tour to moments where a person appears")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Person confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for YOLO inference")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, 0, 0,1...",
    )
    parser.add_argument("--scan-fps", type=float, default=2.0, help="Sampling fps for scan")
    parser.add_argument(
        "--absence-tolerance",
        type=float,
        default=2.0,
        help="Seconds without person before closing an event",
    )
    parser.add_argument(
        "--min-event",
        type=float,
        default=0.8,
        help="Minimum duration (sec) to keep an event",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/events/person_appearance_tour.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--tracks-jsonl",
        type=str,
        default="",
        help="Optional tracking JSONL path to export per-track appearance events",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=2.0,
        help="Max gap (sec) inside one track segment before splitting appearance events",
    )
    parser.add_argument(
        "--min-track-event-seconds",
        type=float,
        default=2.0,
        help="Minimum duration (sec) to keep a track appearance segment",
    )
    parser.add_argument(
        "--min-track-detections",
        type=int,
        default=3,
        help="Minimum detection count to keep a track appearance segment",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracks_jsonl_path = Path(args.tracks_jsonl) if args.tracks_jsonl else None
    if tracks_jsonl_path and tracks_jsonl_path.exists():
        track_events = build_track_appearance_events(
            tracks_jsonl=tracks_jsonl_path,
            gap_seconds_limit=max(0.1, float(args.gap_seconds)),
            min_event_sec=max(0.0, float(args.min_track_event_seconds)),
            min_detections=max(1, int(args.min_track_detections)),
        )
        unique_track_ids = sorted({e.track_id for e in track_events})
        total_zone_hits = int(sum(e.zone_hits for e in track_events))
        payload = {
            "video": args.video,
            "source_tracks_jsonl": str(tracks_jsonl_path),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "criteria": {
                "gap_seconds": _round3(max(0.1, float(args.gap_seconds))),
                "min_track_event_seconds": _round3(max(0.0, float(args.min_track_event_seconds))),
                "min_track_detections": int(max(1, int(args.min_track_detections))),
            },
            "track_events": [asdict(e) for e in track_events],
            "track_count": len(track_events),
            "unique_track_ids": unique_track_ids,
            "unique_track_count": len(unique_track_ids),
            "total_zone_hits": total_zone_hits,
            "first_appearance_second": track_events[0].first_second if track_events else None,
            "first_appearance_frame": track_events[0].first_frame if track_events else None,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if not track_events:
            print("No per-track appearance event found.")
            print(f"Saved: {out_path}")
            return

        print(f"Found {len(track_events)} track event(s).")
        print(
            "First appearance: "
            f"track_id={track_events[0].track_id}, frame={track_events[0].first_frame}, "
            f"second={track_events[0].first_second:.2f}"
        )
        for e in track_events:
            print(
                f"[ID {e.track_id}#{e.segment_index}] {e.first_second:.2f}s -> {e.last_second:.2f}s "
                f"(dur={e.duration_second:.2f}s, det={e.detections}, zone_hits={e.zone_hits})"
            )
        print(f"Saved: {out_path}")
        return

    tour = PersonEventTour(
        model_name=args.model,
        confidence=args.conf,
        iou=args.iou,
        batch_size=args.batch_size,
        device=args.device,
    )
    events = tour.scan(
        video_path=args.video,
        scan_fps=args.scan_fps,
        absence_tolerance_sec=args.absence_tolerance,
        min_event_sec=args.min_event,
    )

    payload = {
        "video": args.video,
        "events": [asdict(e) for e in events],
        "first_appearance_second": events[0].start_second if events else None,
        "first_appearance_frame": events[0].start_frame if events else None,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not events:
        print("No person appearance found.")
        return

    print(f"Found {len(events)} event(s).")
    print(
        f"First appearance: frame={events[0].start_frame}, second={events[0].start_second:.2f}"
    )
    for i, e in enumerate(events, start=1):
        print(
            f"[{i}] {e.start_second:.2f}s -> {e.end_second:.2f}s "
            f"(dur={e.duration_second:.2f}s, max_conf={e.max_confidence:.2f})"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
