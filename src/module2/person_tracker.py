from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv
from deep_sort.deep_sort import DeepSort
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
        self.repo_device = detection_cfg.repo_device
        self.deepsort_min_confidence = tracking_cfg.repo_deepsort_min_confidence
        self.single_person_mode = bool(tracking_cfg.single_person_mode)
        self._stable_track_id = 1
        self._stable_box: Tuple[float, float, float, float] | None = None

        self.tracker = self._build_tracker(tracking_cfg)

        polygon = np.array(zone_cfg.package_zone_polygon, dtype=np.int32)
        self.zone = sv.PolygonZone(polygon=polygon)

    def process_frame(self, frame_index: int, frame: np.ndarray) -> List[PersonTrackEvent]:
        det = self._predict_person_detections(frame)
        if det.class_id is None or len(det) == 0:
            return []

        person_mask = det.class_id == 0
        det = det[person_mask]
        if len(det) == 0:
            return []

        track_boxes, track_ids, track_confs = self._update_tracks(det, frame)

        if not track_boxes:
            return []

        if self.single_person_mode:
            chosen_index = self._choose_single_person_track(track_boxes, track_confs)
            chosen_box = track_boxes[chosen_index]
            chosen_conf = track_confs[chosen_index]
            self._stable_box = chosen_box
            return [
                PersonTrackEvent(
                    frame_index=frame_index,
                    track_id=self._stable_track_id,
                    xyxy=(float(chosen_box[0]), float(chosen_box[1]), float(chosen_box[2]), float(chosen_box[3])),
                    confidence=float(chosen_conf),
                    in_package_zone=bool(
                        self.zone.trigger(
                            detections=sv.Detections(
                                xyxy=np.array([chosen_box], dtype=np.float32),
                                confidence=np.array([chosen_conf], dtype=np.float32),
                                class_id=np.array([0], dtype=np.int32),
                            )
                        )[0]
                    ),
                )
            ]

        tracked = sv.Detections(
            xyxy=np.array(track_boxes, dtype=np.float32),
            confidence=np.array(track_confs, dtype=np.float32),
            class_id=np.zeros(len(track_boxes), dtype=np.int32),
        )
        inside_zone_mask = self.zone.trigger(detections=tracked)

        events: List[PersonTrackEvent] = []
        for i, (xyxy, conf) in enumerate(zip(track_boxes, track_confs)):
            box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            events.append(
                PersonTrackEvent(
                    frame_index=frame_index,
                    track_id=track_ids[i],
                    xyxy=box,
                    confidence=float(conf),
                    in_package_zone=bool(inside_zone_mask[i]),
                )
            )
        return events

    def _choose_single_person_track(
        self,
        track_boxes: List[Tuple[float, float, float, float]],
        track_confs: List[float],
    ) -> int:
        if not track_boxes:
            return 0

        if self._stable_box is None:
            return int(max(range(len(track_boxes)), key=lambda i: track_confs[i]))

        best_index = 0
        best_score = -1.0
        prev_box = self._stable_box
        px1, py1, px2, py2 = prev_box
        prev_area = max(1.0, (px2 - px1) * (py2 - py1))

        for index, box in enumerate(track_boxes):
            x1, y1, x2, y2 = box
            inter_x1 = max(px1, x1)
            inter_y1 = max(py1, y1)
            inter_x2 = min(px2, x2)
            inter_y2 = min(py2, y2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            union = max(1.0, prev_area + box_area - inter_area)
            iou = inter_area / union
            score = (iou * 2.0) + float(track_confs[index])
            if score > best_score:
                best_score = score
                best_index = index

        return best_index

    def _build_tracker(self, tracking_cfg: TrackingConfig):
        use_cuda = bool(tracking_cfg.repo_deepsort_use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0)
        return DeepSort(
            model_path=tracking_cfg.repo_deepsort_weights,
            max_dist=tracking_cfg.deepsort_max_cosine_distance,
            min_confidence=tracking_cfg.repo_deepsort_min_confidence,
            max_iou_distance=tracking_cfg.deepsort_max_iou_distance,
            max_age=tracking_cfg.track_buffer,
            n_init=tracking_cfg.deepsort_n_init,
            nn_budget=tracking_cfg.deepsort_nn_budget,
            use_cuda=use_cuda,
        )

    def _update_tracks(
        self,
        det: sv.Detections,
        frame: np.ndarray,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[int], List[float]]:
        bboxes_xywh: List[List[float]] = []
        for xyxy in det.xyxy:
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + (w * 0.5)
            cy = y1 + (h * 0.5)
            bboxes_xywh.append([cx, cy, w, h])

        if not bboxes_xywh:
            return [], [], []

        confs = np.asarray(det.confidence, dtype=np.float32)
        tracks = self.tracker.update(np.asarray(bboxes_xywh, dtype=np.float32), confs, frame)

        if tracks is None or len(tracks) == 0:
            return [], [], []

        track_boxes: List[Tuple[float, float, float, float]] = []
        track_ids: List[int] = []
        for row in tracks:
            x1, y1, x2, y2, track_id = row.tolist()
            track_boxes.append((float(x1), float(y1), float(x2), float(y2)))
            track_ids.append(int(track_id))

        track_confs = self._estimate_track_confidences(track_boxes, det)
        return track_boxes, track_ids, track_confs

    def _estimate_track_confidences(
        self,
        track_boxes: List[Tuple[float, float, float, float]],
        det: sv.Detections,
    ) -> List[float]:
        if len(det) == 0:
            return [0.0] * len(track_boxes)

        det_xyxy = np.asarray(det.xyxy, dtype=np.float32)
        det_conf = np.asarray(det.confidence, dtype=np.float32)
        confs: List[float] = []

        for box in track_boxes:
            tx1, ty1, tx2, ty2 = box
            t_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
            best_iou = 0.0
            best_conf = 0.0
            for i, d in enumerate(det_xyxy):
                dx1, dy1, dx2, dy2 = d.tolist()
                ix1 = max(tx1, dx1)
                iy1 = max(ty1, dy1)
                ix2 = min(tx2, dx2)
                iy2 = min(ty2, dy2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                d_area = max(1.0, (dx2 - dx1) * (dy2 - dy1))
                union = max(1.0, t_area + d_area - inter)
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_conf = float(det_conf[i])
            confs.append(best_conf)

        return confs

    def has_person(self, frame: np.ndarray) -> bool:
        det = self._predict_person_detections(frame)
        if len(det) == 0 or det.confidence is None:
            return False
        return bool(np.any(det.confidence >= self.confidence_threshold))

    def _predict_person_detections(self, frame: np.ndarray) -> sv.Detections:
        model_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = self.model(
                model_input,
                device=self.repo_device,
                classes=0,
                conf=self.confidence_threshold,
                verbose=False,
            )[0]
        except Exception:
            result = self.model(
                model_input,
                device="cpu",
                classes=0,
                conf=self.confidence_threshold,
                verbose=False,
            )[0]
        return sv.Detections.from_ultralytics(result)

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
