from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from dateutil import tz

from src.config import StreamConfig, VideoSourceConfig


@dataclass
class FramePacket:
    frame_index: int
    frame: np.ndarray
    timestamp_utc: datetime
    timestamp_local: datetime


def _open_video_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def iter_sampled_frames(
    video_cfg: VideoSourceConfig,
    stream_cfg: StreamConfig,
    start_utc: Optional[datetime] = None,
) -> Generator[FramePacket, None, None]:
    cap = _open_video_source(video_cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_cfg.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / max(stream_cfg.sample_fps, 0.1))))

    if start_utc is None:
        start_utc = datetime.now(UTC)
    elif start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=UTC)

    local_tz = tz.gettz(video_cfg.timezone)
    if local_tz is None:
        local_tz = tz.UTC
    frames_dir = Path(stream_cfg.output_frames_dir)
    if stream_cfg.save_sampled_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % step == 0:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            elapsed_seconds = (pos_msec / 1000.0) if pos_msec and pos_msec > 0 else (frame_index / fps)
            ts_utc = start_utc + timedelta(seconds=elapsed_seconds)
            ts_local = ts_utc.astimezone(local_tz)

            if stream_cfg.save_sampled_frames:
                out_path = frames_dir / f"frame_{frame_index:08d}.jpg"
                cv2.imwrite(str(out_path), frame)

            yield FramePacket(
                frame_index=frame_index,
                frame=frame,
                timestamp_utc=ts_utc,
                timestamp_local=ts_local,
            )

        frame_index += 1

    cap.release()
