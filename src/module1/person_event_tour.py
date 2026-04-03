from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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


class PersonEventTour:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence

    def _detect_person(self, frame: np.ndarray) -> Tuple[bool, float]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return False, 0.0

        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        person_conf = conf[cls == 0]  # COCO class 0 = person
        if person_conf.size == 0:
            return False, 0.0

        return True, float(np.max(person_conf))

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

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            native_fps = 25.0

        stride = max(1, int(round(native_fps / max(scan_fps, 0.1))))

        events: List[PersonAppearanceEvent] = []

        open_start_frame: Optional[int] = None
        open_start_sec: Optional[float] = None
        last_seen_frame: Optional[int] = None
        last_seen_sec: Optional[float] = None
        open_max_conf = 0.0
        open_detections = 0

        frame_index = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break

            t_sec = frame_index / native_fps
            has_person, best_conf = self._detect_person(frame)

            if has_person:
                if open_start_frame is None:
                    open_start_frame = frame_index
                    open_start_sec = t_sec
                    open_max_conf = best_conf
                    open_detections = 1
                else:
                    open_max_conf = max(open_max_conf, best_conf)
                    open_detections += 1

                last_seen_frame = frame_index
                last_seen_sec = t_sec
            elif open_start_frame is not None and last_seen_sec is not None:
                if (t_sec - last_seen_sec) >= absence_tolerance_sec:
                    duration = last_seen_sec - float(open_start_sec)
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

            frame_index += stride

        if open_start_frame is not None and last_seen_sec is not None and open_start_sec is not None:
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

        cap.release()
        return events


def build_track_appearance_events(tracks_jsonl: Path) -> List[PersonTrackAppearanceEvent]:
    if not tracks_jsonl.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracks_jsonl}")

    grouped: Dict[int, Dict[str, object]] = {}

    with tracks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue

            data = json.loads(row)
            track_id = int(data.get("track_id", -1))
            if track_id < 0:
                continue

            frame_index = int(data.get("frame_index", 0))
            elapsed_seconds = float(data.get("elapsed_seconds", 0.0))
            timestamp_utc = str(data.get("timestamp_utc", ""))
            timestamp_local = str(data.get("timestamp_local", ""))
            confidence = float(data.get("confidence", 0.0))
            in_zone = bool(data.get("in_package_zone", False))

            state = grouped.get(track_id)
            if state is None:
                grouped[track_id] = {
                    "track_id": track_id,
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "first_second": elapsed_seconds,
                    "last_second": elapsed_seconds,
                    "first_timestamp_utc": timestamp_utc,
                    "last_timestamp_utc": timestamp_utc,
                    "first_timestamp_local": timestamp_local,
                    "last_timestamp_local": timestamp_local,
                    "detections": 1,
                    "zone_hits": 1 if in_zone else 0,
                    "max_confidence": confidence,
                }
                continue

            state["detections"] = int(state["detections"]) + 1
            if in_zone:
                state["zone_hits"] = int(state["zone_hits"]) + 1
            state["max_confidence"] = max(float(state["max_confidence"]), confidence)

            if frame_index < int(state["first_frame"]):
                state["first_frame"] = frame_index
                state["first_second"] = elapsed_seconds
                state["first_timestamp_utc"] = timestamp_utc
                state["first_timestamp_local"] = timestamp_local

            if frame_index > int(state["last_frame"]):
                state["last_frame"] = frame_index
                state["last_second"] = elapsed_seconds
                state["last_timestamp_utc"] = timestamp_utc
                state["last_timestamp_local"] = timestamp_local

    events: List[PersonTrackAppearanceEvent] = []
    for track_id in sorted(grouped.keys()):
        s = grouped[track_id]
        first_second = float(s["first_second"])
        last_second = float(s["last_second"])
        events.append(
            PersonTrackAppearanceEvent(
                track_id=int(s["track_id"]),
                first_frame=int(s["first_frame"]),
                last_frame=int(s["last_frame"]),
                first_second=first_second,
                last_second=last_second,
                duration_second=max(0.0, last_second - first_second),
                first_timestamp_utc=str(s["first_timestamp_utc"]),
                last_timestamp_utc=str(s["last_timestamp_utc"]),
                first_timestamp_local=str(s["first_timestamp_local"]),
                last_timestamp_local=str(s["last_timestamp_local"]),
                detections=int(s["detections"]),
                zone_hits=int(s["zone_hits"]),
                max_confidence=float(s["max_confidence"]),
            )
        )

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast tour to moments where a person appears")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Person confidence threshold")
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
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracks_jsonl_path = Path(args.tracks_jsonl) if args.tracks_jsonl else None
    if tracks_jsonl_path and tracks_jsonl_path.exists():
        track_events = build_track_appearance_events(tracks_jsonl=tracks_jsonl_path)
        payload = {
            "video": args.video,
            "source_tracks_jsonl": str(tracks_jsonl_path),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "track_events": [asdict(e) for e in track_events],
            "track_count": len(track_events),
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
                f"[ID {e.track_id}] {e.first_second:.2f}s -> {e.last_second:.2f}s "
                f"(dur={e.duration_second:.2f}s, det={e.detections}, zone_hits={e.zone_hits})"
            )
        print(f"Saved: {out_path}")
        return

    tour = PersonEventTour(model_name=args.model, confidence=args.conf)
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
