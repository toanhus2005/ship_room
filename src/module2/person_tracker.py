from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
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
        self.min_box_area_ratio = detection_cfg.min_box_area_ratio
        self.border_exclusion_px = detection_cfg.border_exclusion_px
        self.inference_scale = max(0.25, min(1.0, detection_cfg.inference_scale))
        self.inference_imgsz = detection_cfg.inference_imgsz
        self.inference_half = detection_cfg.inference_half
        self.max_detections = detection_cfg.max_detections

        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence_threshold,
            lost_track_buffer=tracking_cfg.track_buffer,
            minimum_matching_threshold=tracking_cfg.match_threshold,
        )

        polygon = np.array(zone_cfg.package_zone_polygon, dtype=np.int32)
        self.zone = sv.PolygonZone(polygon=polygon)

    def process_frame(self, frame_index: int, frame: np.ndarray) -> List[PersonTrackEvent]:
        det = self._predict_person_detections(frame)
        if det.class_id is None or len(det) == 0:
            return []

        person_mask = det.class_id == 0
        det = det[person_mask]
        det = self._filter_person_detections(det, frame)

        if len(det) == 0:
            return []

        tracked = self.tracker.update_with_detections(det)
        inside_zone_mask = self.zone.trigger(detections=tracked)

        events: List[PersonTrackEvent] = []
        for i, (xyxy, conf) in enumerate(zip(tracked.xyxy, tracked.confidence)):
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            events.append(
                PersonTrackEvent(
                    frame_index=frame_index,
                    track_id=tid,
                    xyxy=box,
                    confidence=float(conf),
                    in_package_zone=bool(inside_zone_mask[i]),
                )
            )
        return events

    def has_person(self, frame: np.ndarray) -> bool:
        det = self._predict_person_detections(frame)
        if len(det) == 0 or det.confidence is None:
            return False
        return bool(np.any(det.confidence >= self.confidence_threshold))

    def _predict_person_detections(self, frame: np.ndarray) -> sv.Detections:
        source_frame = frame
        scale_x = 1.0
        scale_y = 1.0

        if self.inference_scale < 1.0:
            h, w = frame.shape[:2]
            new_w = max(64, int(round(w * self.inference_scale)))
            new_h = max(64, int(round(h * self.inference_scale)))
            source_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_x = w / float(new_w)
            scale_y = h / float(new_h)

        result = self.model.predict(
            source=source_frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],
            imgsz=self.inference_imgsz,
            half=self.inference_half,
            max_det=self.max_detections,
            verbose=False,
        )[0]

        det = sv.Detections.from_ultralytics(result)
        if len(det) == 0:
            return det

        if scale_x != 1.0 or scale_y != 1.0:
            det.xyxy[:, [0, 2]] *= scale_x
            det.xyxy[:, [1, 3]] *= scale_y

        return det

    def _filter_person_detections(self, det: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if len(det) == 0:
            return det

        frame_h, frame_w = frame.shape[:2]
        min_area = float(frame_w * frame_h) * self.min_box_area_ratio

        xyxy = det.xyxy
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        areas = widths * heights

        area_mask = areas >= min_area
        if self.border_exclusion_px > 0:
            b = float(self.border_exclusion_px)
            border_mask = (
                (xyxy[:, 0] > b)
                & (xyxy[:, 1] > b)
                & (xyxy[:, 2] < (frame_w - b))
                & (xyxy[:, 3] < (frame_h - b))
            )
            keep_mask = area_mask & border_mask
        else:
            keep_mask = area_mask

        return det[keep_mask]


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
