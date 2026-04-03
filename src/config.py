from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel, Field


class VideoSourceConfig(BaseModel):
    source: str = Field(
        ..., description="RTSP URL, camera index (as string), or path to video file"
    )
    timezone: str = "Asia/Ho_Chi_Minh"


class StreamConfig(BaseModel):
    sample_fps: float = 5.0
    output_frames_dir: Path = Path("artifacts/frames")
    save_sampled_frames: bool = False


class DetectionConfig(BaseModel):
    model_name: str = "yolov8s.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.5
    min_box_area_ratio: float = 0.003
    min_box_height_ratio: float = 0.12
    min_box_height_width_ratio: float = 1.15
    border_exclusion_px: int = 0
    inference_scale: float = 0.85
    inference_imgsz: int = 640
    inference_half: bool = False
    max_detections: int = 25


class TrackingConfig(BaseModel):
    track_buffer: int = 80
    match_threshold: float = 0.7
    deepsort_n_init: int = 3
    deepsort_max_iou_distance: float = 0.7
    deepsort_max_cosine_distance: float = 0.25
    deepsort_nn_budget: int = 100
    deepsort_embedder: str = "mobilenet"


class ZoneConfig(BaseModel):
    # Polygon points in image coordinates: [[x1, y1], [x2, y2], ...]
    package_zone_polygon: List[Tuple[int, int]] = [
        (200, 150),
        (1100, 150),
        (1100, 650),
        (200, 650),
    ]


class OutputConfig(BaseModel):
    events_jsonl: Path = Path("artifacts/events/person_tracks.jsonl")


class ToanConfig(BaseModel):
    video: VideoSourceConfig
    stream: StreamConfig = StreamConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    zone: ZoneConfig = ZoneConfig()
    output: OutputConfig = OutputConfig()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
