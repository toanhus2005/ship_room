from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.config import DetectionConfig, TrackingConfig, ZoneConfig


@dataclass
class PersonTrackEvent:
    frame_index: int
    track_id: int
    xyxy: Tuple[float, float, float, float]
    confidence: float
    in_package_zone: bool


class PersonTracker:
    def __init__(
        self,
        detection_cfg: DetectionConfig,
        tracking_cfg: TrackingConfig,
        zone_cfg: ZoneConfig,
    ) -> None:
        self.model = YOLO(detection_cfg.model_name)
        self.confidence_threshold = detection_cfg.confidence_threshold
        self.iou_threshold = detection_cfg.iou_threshold

        # ByteTrack params follow supervision API.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence_threshold,
            lost_track_buffer=tracking_cfg.track_buffer,
            minimum_matching_threshold=tracking_cfg.match_threshold,
        )

        polygon = np.array(zone_cfg.package_zone_polygon, dtype=np.int32)
        self.zone = sv.PolygonZone(polygon=polygon)

    def process_frame(self, frame_index: int, frame: np.ndarray) -> List[PersonTrackEvent]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],
            verbose=False,
        )[0]

        det = sv.Detections.from_ultralytics(result)
        if det.class_id is None or len(det) == 0:
            return []

        # COCO class 0 = person
        person_mask = det.class_id == 0
        det = det[person_mask]

        if len(det) == 0:
            return []

        tracked = self.tracker.update_with_detections(det)

        inside_zone_mask = self.zone.trigger(detections=tracked)

        events: List[PersonTrackEvent] = []
        for i, (xyxy, conf) in enumerate(zip(tracked.xyxy, tracked.confidence)):
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            events.append(
                PersonTrackEvent(
                    frame_index=frame_index,
                    track_id=tid,
                    xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=float(conf),
                    in_package_zone=bool(inside_zone_mask[i]),
                )
            )

        return events


def summarize_trajectories(events: List[PersonTrackEvent]) -> Dict[int, Dict[str, int]]:
    summary: Dict[int, Dict[str, int]] = {}
    for e in events:
        if e.track_id not in summary:
            summary[e.track_id] = {
                "first_frame": e.frame_index,
                "last_frame": e.frame_index,
                "zone_hits": 1 if e.in_package_zone else 0,
            }
        else:
            summary[e.track_id]["last_frame"] = e.frame_index
            if e.in_package_zone:
                summary[e.track_id]["zone_hits"] += 1

    return summary
