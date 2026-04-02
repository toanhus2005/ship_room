from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

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


@dataclass
class _StableTrackState:
    last_frame: int
    last_xyxy: Tuple[float, float, float, float]


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

        # ByteTrack params follow supervision API.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence_threshold,
            lost_track_buffer=tracking_cfg.track_buffer,
            minimum_matching_threshold=tracking_cfg.match_threshold,
        )

        self.id_reassign_iou_threshold = tracking_cfg.id_reassign_iou_threshold
        self.id_reassign_center_distance = tracking_cfg.id_reassign_center_distance
        self.id_reassign_max_frame_gap = tracking_cfg.id_reassign_max_frame_gap
        self.id_reassign_min_area_ratio = tracking_cfg.id_reassign_min_area_ratio
        self.id_reassign_max_area_ratio = tracking_cfg.id_reassign_max_area_ratio
        self.id_reassign_max_aspect_ratio_delta = tracking_cfg.id_reassign_max_aspect_ratio_delta

        self._next_stable_id = 1
        self._raw_to_stable: Dict[int, int] = {}
        self._raw_last_seen_frame: Dict[int, int] = {}
        self._stable_states: Dict[int, _StableTrackState] = {}

        polygon = np.array(zone_cfg.package_zone_polygon, dtype=np.int32)
        self.zone = sv.PolygonZone(polygon=polygon)

    def process_frame(self, frame_index: int, frame: np.ndarray) -> List[PersonTrackEvent]:
        self._cleanup_stale_maps(frame_index)

        det = self._predict_person_detections(frame)
        if det.class_id is None or len(det) == 0:
            return []

        # COCO class 0 = person
        person_mask = det.class_id == 0
        det = det[person_mask]
        det = self._filter_person_detections(det, frame)

        if len(det) == 0:
            return []

        tracked = self.tracker.update_with_detections(det)

        inside_zone_mask = self.zone.trigger(detections=tracked)

        events: List[PersonTrackEvent] = []
        stable_ids_in_frame: Set[int] = set()

        for i, (xyxy, conf) in enumerate(zip(tracked.xyxy, tracked.confidence)):
            raw_tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            tid = self._assign_stable_id(
                frame_index=frame_index,
                raw_tid=raw_tid,
                xyxy=box,
                stable_ids_in_frame=stable_ids_in_frame,
            )
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

    def _assign_stable_id(
        self,
        frame_index: int,
        raw_tid: int,
        xyxy: Tuple[float, float, float, float],
        stable_ids_in_frame: Set[int],
    ) -> int:
        if raw_tid >= 0:
            stable_tid = self._raw_to_stable.get(raw_tid)
            if stable_tid is None:
                stable_tid = self._reassociate_stable_id(frame_index, xyxy, stable_ids_in_frame)
                self._raw_to_stable[raw_tid] = stable_tid
            self._raw_last_seen_frame[raw_tid] = frame_index
        else:
            stable_tid = self._reassociate_stable_id(frame_index, xyxy, stable_ids_in_frame)

        self._stable_states[stable_tid] = _StableTrackState(last_frame=frame_index, last_xyxy=xyxy)
        stable_ids_in_frame.add(stable_tid)
        return stable_tid

    def _reassociate_stable_id(
        self,
        frame_index: int,
        xyxy: Tuple[float, float, float, float],
        stable_ids_in_frame: Set[int],
    ) -> int:
        best_match_id = -1
        best_score = -1.0

        for stable_id, state in self._stable_states.items():
            if stable_id in stable_ids_in_frame:
                continue

            gap = frame_index - state.last_frame
            if gap <= 0 or gap > self.id_reassign_max_frame_gap:
                continue

            iou = self._bbox_iou(xyxy, state.last_xyxy)
            if iou < self.id_reassign_iou_threshold and not self._is_center_close(
                xyxy, state.last_xyxy
            ):
                continue

            if not self._is_size_shape_compatible(xyxy, state.last_xyxy):
                continue

            center_dist = self._normalized_center_distance(xyxy, state.last_xyxy)
            score = (0.75 * iou) + (0.25 * (1.0 - min(center_dist, 1.0)))
            if score > best_score:
                best_score = score
                best_match_id = stable_id

        if best_match_id != -1:
            return best_match_id

        new_id = self._next_stable_id
        self._next_stable_id += 1
        return new_id

    def _is_size_shape_compatible(
        self,
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> bool:
        area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        area_ratio = area_a / area_b
        if area_ratio < self.id_reassign_min_area_ratio or area_ratio > self.id_reassign_max_area_ratio:
            return False

        aspect_a = (box_a[2] - box_a[0]) / max(1.0, (box_a[3] - box_a[1]))
        aspect_b = (box_b[2] - box_b[0]) / max(1.0, (box_b[3] - box_b[1]))
        if abs(aspect_a - aspect_b) > self.id_reassign_max_aspect_ratio_delta:
            return False

        return True

    def _is_center_close(
        self,
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> bool:
        return self._normalized_center_distance(box_a, box_b) <= self.id_reassign_center_distance

    @staticmethod
    def _normalized_center_distance(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        ax = (box_a[0] + box_a[2]) * 0.5
        ay = (box_a[1] + box_a[3]) * 0.5
        bx = (box_b[0] + box_b[2]) * 0.5
        by = (box_b[1] + box_b[3]) * 0.5

        dx = ax - bx
        dy = ay - by
        center_dist = float(np.hypot(dx, dy))

        bw = max(1.0, box_b[2] - box_b[0])
        bh = max(1.0, box_b[3] - box_b[1])
        norm = float(np.hypot(bw, bh))
        return center_dist / norm

    @staticmethod
    def _bbox_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        inter_w = max(0.0, x_right - x_left)
        inter_h = max(0.0, y_bottom - y_top)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _cleanup_stale_maps(self, frame_index: int) -> None:
        stale_raw_ids = [
            raw_id
            for raw_id, last_seen in self._raw_last_seen_frame.items()
            if frame_index - last_seen > self.id_reassign_max_frame_gap
        ]
        for raw_id in stale_raw_ids:
            self._raw_last_seen_frame.pop(raw_id, None)
            self._raw_to_stable.pop(raw_id, None)

        stale_stable_ids = [
            stable_id
            for stable_id, state in self._stable_states.items()
            if frame_index - state.last_frame > self.id_reassign_max_frame_gap
        ]
        for stable_id in stale_stable_ids:
            self._stable_states.pop(stable_id, None)


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
