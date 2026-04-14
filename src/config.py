from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel, Field


class VideoSourceConfig(BaseModel):
    source: str = Field(
        "data/IMG_5812.MOV",
        description="RTSP URL, camera index (as string), or path to video file",
    )
    timezone: str = "Asia/Ho_Chi_Minh"


class StreamConfig(BaseModel):
    sample_fps: float = 5.0
    process_width: int = 960  # Match preview default (0 = original size)
    output_frames_dir: Path = Path("artifacts/frames")
    save_sampled_frames: bool = False


class DetectionConfig(BaseModel):
    model_name: str = "yolov8m.pt"
    confidence_threshold: float = 0.3
    repo_device: str = "0"


class TrackingConfig(BaseModel):
    track_buffer: int = 150
    bytetrack_high_thresh: float = 0.5
    bytetrack_low_thresh: float = 0.1
    bytetrack_new_track_thresh: float = 0.5
    bytetrack_match_thresh: float = 0.8
    bytetrack_fuse_score: bool = True
    repo_track_min_confidence: float = 0.3
    single_person_mode: bool = False


class ZoneConfig(BaseModel):
    # Polygon points in image coordinates: [[x1, y1], [x2, y2], ...]
    package_zone_polygon: List[Tuple[int, int]] = [
        (1040, 440),
        (1460, 440),
        (1460, 780),
        (1040, 780),
    ]


class OutputConfig(BaseModel):
    events_jsonl: Path = Path("artifacts/events/person_tracks.jsonl")


class ToanConfig(BaseModel):
    video: VideoSourceConfig = VideoSourceConfig()
    stream: StreamConfig = StreamConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    zone: ZoneConfig = ZoneConfig()
    output: OutputConfig = OutputConfig()


def build_shared_config(video_source: str | None = None, timezone: str | None = None) -> ToanConfig:
    cfg = ToanConfig()
    if video_source:
        cfg.video.source = video_source
    if timezone:
        cfg.video.timezone = timezone
    return cfg


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
