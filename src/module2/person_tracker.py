from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
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
    PERSON_CLASS_ID = 0  # COCO class 0 = person
    TRACK_CONF_MIN_IOU = 0.3

    def __init__(
        self,
        detection_cfg: DetectionConfig,
        tracking_cfg: TrackingConfig,
        zone_cfg: ZoneConfig,
    ) -> None:
        if torch.cuda.is_available():
            # Favor throughput for fixed-size realtime inference workloads.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = YOLO(detection_cfg.model_name)
        self.confidence_threshold = detection_cfg.confidence_threshold
        self.repo_device = detection_cfg.repo_device
        self._force_gpu = str(self.repo_device).lower() not in {"cpu", "mps"}
        self._inference_device = self.repo_device
        self._cpu_fallback_active = False
        self._use_half = bool(self._force_gpu and torch.cuda.is_available())
        self._inference_imgsz = 640
        self.deepsort_min_confidence = tracking_cfg.repo_deepsort_min_confidence
        self.single_person_mode = bool(tracking_cfg.single_person_mode)
        self._stable_track_id = 1
        self._stable_source_track_id: int | None = None
        self._stable_box: Tuple[float, float, float, float] | None = None

        print(f"[INFO] YOLO inference device requested: {self.repo_device}")
        self.tracker = self._build_tracker(tracking_cfg)

        polygon = np.array(zone_cfg.package_zone_polygon, dtype=np.int32)
        try:
            self.zone = sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,),
            )
        except TypeError:
            # Backward compatible with older supervision versions.
            self.zone = sv.PolygonZone(polygon=polygon)

    def process_frame(self, frame_index: int, frame: np.ndarray) -> List[PersonTrackEvent]:
        det = self._predict_person_detections(frame)
        if len(det) == 0:
            return []

        track_boxes, track_ids, track_confs = self._update_tracks(det, frame)

        if not track_boxes:
            return []

        if self.single_person_mode:
            chosen_index = self._choose_single_person_track(track_boxes, track_ids, track_confs)
            chosen_box = track_boxes[chosen_index]
            chosen_conf = track_confs[chosen_index]
            self._stable_box = chosen_box
            self._stable_source_track_id = int(track_ids[chosen_index])
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
        track_ids: List[int],
        track_confs: List[float],
    ) -> int:
        if not track_boxes:
            return 0

        # Keep the same DeepSORT id when available to reduce ID switching.
        if self._stable_source_track_id is not None:
            for i, tid in enumerate(track_ids):
                if int(tid) == self._stable_source_track_id:
                    return i

        if self._stable_box is None:
            return int(max(range(len(track_boxes)), key=lambda i: track_confs[i]))

        px1, py1, px2, py2 = self._stable_box
        prev_cx = (px1 + px2) * 0.5
        prev_cy = (py1 + py2) * 0.5

        best_index = 0
        best_score = -1e18
        for index, box in enumerate(track_boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            dist_sq = ((cx - prev_cx) ** 2) + ((cy - prev_cy) ** 2)
            # Prefer spatial continuity, then confidence as tie-breaker.
            score = -dist_sq + (float(track_confs[index]) * 100.0)
            if score > best_score:
                best_score = score
                best_index = index

        return int(best_index)

    def _build_tracker(self, tracking_cfg: TrackingConfig):
        use_cuda = bool(tracking_cfg.repo_deepsort_use_cuda and torch.cuda.is_available())
        print(f"[INFO] DeepSORT CUDA enabled: {use_cuda}")
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
        if len(det) == 0 or not track_boxes:
            return [0.0] * len(track_boxes)

        track_xyxy = np.asarray(track_boxes, dtype=np.float32)
        det_xyxy = np.asarray(det.xyxy, dtype=np.float32)
        det_conf = np.asarray(det.confidence, dtype=np.float32)

        # Vectorized IoU matrix: (n_tracks, n_detections)
        t = track_xyxy[:, None, :]
        d = det_xyxy[None, :, :]

        ix1 = np.maximum(t[..., 0], d[..., 0])
        iy1 = np.maximum(t[..., 1], d[..., 1])
        ix2 = np.minimum(t[..., 2], d[..., 2])
        iy2 = np.minimum(t[..., 3], d[..., 3])

        iw = np.clip(ix2 - ix1, a_min=0.0, a_max=None)
        ih = np.clip(iy2 - iy1, a_min=0.0, a_max=None)
        inter = iw * ih

        t_area = np.clip((track_xyxy[:, 2] - track_xyxy[:, 0]) * (track_xyxy[:, 3] - track_xyxy[:, 1]), a_min=1.0, a_max=None)
        d_area = np.clip((det_xyxy[:, 2] - det_xyxy[:, 0]) * (det_xyxy[:, 3] - det_xyxy[:, 1]), a_min=1.0, a_max=None)
        union = np.clip(t_area[:, None] + d_area[None, :] - inter, a_min=1.0, a_max=None)
        iou_matrix = inter / union

        best_det_idx = np.argmax(iou_matrix, axis=1)
        best_iou = iou_matrix[np.arange(iou_matrix.shape[0]), best_det_idx]

        confs = np.zeros((len(track_boxes),), dtype=np.float32)
        valid = best_iou >= self.TRACK_CONF_MIN_IOU
        confs[valid] = det_conf[best_det_idx[valid]]
        return [float(v) for v in confs]

    def has_person(self, frame: np.ndarray) -> bool:
        det = self._predict_person_detections(frame)
        if len(det) == 0 or det.confidence is None:
            return False
        return bool(np.any(det.confidence >= self.confidence_threshold))

    def _predict_person_detections(self, frame: np.ndarray) -> sv.Detections:
        if self._cpu_fallback_active:
            result = self.model.predict(
                source=frame,
                device="cpu",
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence_threshold,
                imgsz=self._inference_imgsz,
                verbose=False,
            )[0]
            return sv.Detections.from_ultralytics(result)

        try:
            result = self.model.predict(
                source=frame,
                device=self._inference_device,
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence_threshold,
                half=self._use_half,
                imgsz=self._inference_imgsz,
                verbose=False,
            )[0]
        except Exception as exc:
            if self._force_gpu:
                raise RuntimeError(
                    f"GPU inference failed on device '{self.repo_device}'. "
                    "Please verify CUDA-capable Torch is installed and GPU is available."
                ) from exc
            print("[WARN] Falling back to CPU inference for subsequent frames.")
            self._cpu_fallback_active = True
            self._inference_device = "cpu"
            self._use_half = False
            result = self.model.predict(
                source=frame,
                device="cpu",
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence_threshold,
                imgsz=self._inference_imgsz,
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
