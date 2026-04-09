from __future__ import annotations

import argparse
from threading import Lock
import time
from typing import Generator

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from src.config import DetectionConfig, TrackingConfig, ZoneConfig
from src.module2.person_tracker import PersonTracker


def _default_zone_cli_values() -> list[int]:
    values: list[int] = []
    for x, y in ZoneConfig().package_zone_polygon:
        values.extend([int(x), int(y)])
    return values


def build_app(
    video_path: str,
    model_name: str,
    conf: float,
    iou: float,
    zone_points: list[tuple[int, int]],
    process_width: int,
    jpeg_quality: int,
    status_poll_ms: int,
) -> Flask:
    app = Flask(__name__)

    cap_meta = cv2.VideoCapture(video_path)
    if not cap_meta.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    native_fps = cap_meta.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0
    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    src_width = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    src_height = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap_meta.release()

    resize_scale = 1.0
    if process_width > 0 and src_width > 0 and process_width < src_width:
        resize_scale = process_width / float(src_width)

    tracker_zone_points = [
        (
            int(round(x * resize_scale)),
            int(round(y * resize_scale)),
        )
        for x, y in zone_points
    ]
    jpg_quality = int(max(35, min(95, jpeg_quality)))
    poll_interval_ms = int(max(150, status_poll_ms))

    def make_tracker() -> PersonTracker:
        return PersonTracker(
            detection_cfg=DetectionConfig(
                model_name=model_name,
                confidence_threshold=conf,
            ),
            tracking_cfg=TrackingConfig(track_buffer=30),
            zone_cfg=ZoneConfig(package_zone_polygon=tracker_zone_points),
        )

    total_duration = (total_frames / native_fps) if native_fps > 0 else 0.0

    state_lock = Lock()
    state = {
        "current_frame": 0,
        "skip_requested": False,
        "last_skip_frame": None,
        "paused": False,
        "seek_frame": None,
    }

    def has_person(frame, tracker: PersonTracker) -> bool:
        return tracker.has_person(frame)

    def find_next_person_frame(
        cap: cv2.VideoCapture,
        tracker: PersonTracker,
        start_frame: int,
        end_frame: int,
        step: int,
    ) -> int | None:
        cand = max(0, start_frame)
        while cand < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand)
            ok, frm = cap.read()
            if not ok:
                return None
            if has_person(frm, tracker):
                return cand
            cand += step
        return None

    def frame_generator() -> Generator[bytes, None, None]:
        tracker = make_tracker()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        ema_fps = 0.0
        scan_stride = max(1, int(native_fps))
        last_payload = None

        try:
            while True:
                with state_lock:
                    should_skip = bool(state["skip_requested"])
                    state["skip_requested"] = False
                    paused = bool(state["paused"])
                    seek_target = state["seek_frame"]
                    state["seek_frame"] = None

                force_render_once = False

                if seek_target is not None:
                    target_frame = int(max(0, min(total_frames - 1, int(seek_target)))) if total_frames > 0 else 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    frame_idx = target_frame
                    tracker = make_tracker()
                    force_render_once = True

                if should_skip:
                    target = find_next_person_frame(
                        cap=cap,
                        tracker=tracker,
                        start_frame=frame_idx + 1,
                        end_frame=total_frames,
                        step=scan_stride,
                    )
                    if target is None and total_frames > 0:
                        target = find_next_person_frame(
                            cap=cap,
                            tracker=tracker,
                            start_frame=0,
                            end_frame=frame_idx,
                            step=scan_stride,
                        )
                    if target is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        frame_idx = int(target)
                        tracker = make_tracker()
                        force_render_once = True
                        with state_lock:
                            state["last_skip_frame"] = int(target)

                if paused and not force_render_once:
                    if last_payload is not None:
                        with state_lock:
                            state["current_frame"] = int(frame_idx)
                        yield last_payload
                    time.sleep(0.04)
                    continue

                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    tracker = make_tracker()
                    continue

                if resize_scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (
                            int(round(frame.shape[1] * resize_scale)),
                            int(round(frame.shape[0] * resize_scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

                t0 = time.perf_counter()
                events = tracker.process_frame(frame_idx, frame)
                dt = max(1e-6, time.perf_counter() - t0)
                inst = 1.0 / dt
                ema_fps = inst if ema_fps == 0 else (0.9 * ema_fps + 0.1 * inst)

                if ema_fps < 20:
                    fps_color = (0, 0, 255)
                elif ema_fps < 30:
                    fps_color = (0, 255, 255)
                else:
                    fps_color = (0, 255, 0)

                view = frame

                # Draw package zone border (yellow) for visual reference.
                zone_polygon = np.array(tracker_zone_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(view, [zone_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

                for e in events:
                    x1, y1, x2, y2 = map(int, e.xyxy)
                    color = (0, 0, 255) if e.in_package_zone else (0, 255, 0)
                    cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        view,
                        f"ID {e.track_id} {e.confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                fps_text = f"{int(round(ema_fps))}"
                text_x, text_y = 10, 36
                cv2.putText(
                    view,
                    fps_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    view,
                    fps_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    fps_color,
                    2,
                    cv2.LINE_AA,
                )

                with state_lock:
                    state["current_frame"] = int(frame_idx)

                ok_jpg, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                if not ok_jpg:
                    frame_idx += 1
                    continue

                payload = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
                last_payload = payload
                yield payload

                frame_idx += 1
        finally:
            cap.release()

    @app.get("/")
    def index() -> str:
        return (
            "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Live detect preview</title>"
            "<style>"
            "body{margin:0;background:#111;color:#fff;font-family:Arial,sans-serif;}"
            ".wrap{max-width:1280px;margin:16px auto;padding:0 12px;}"
            ".title{font-size:18px;font-weight:700;margin:0 0 10px;}"
            ".player{position:relative;background:#000;border:1px solid #2b2b2b;border-radius:10px;overflow:hidden;}"
            "#stream{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#000;}"
            ".controls{position:absolute;left:0;right:0;bottom:0;padding:10px;background:linear-gradient(180deg,rgba(0,0,0,0.02) 0%,rgba(0,0,0,0.72) 30%,rgba(0,0,0,0.88) 100%);backdrop-filter:blur(2px);opacity:0.12;transition:opacity .2s ease;}"
            ".player:hover .controls,.controls:focus-within{opacity:1;}"
            ".row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}"
            "button{padding:6px 10px;background:#2b2b2bcc;color:#eee;border:1px solid #777;border-radius:7px;cursor:pointer;}"
            "button:hover{background:#383838;}"
            "#timeline{width:100%;margin:10px 0 4px;accent-color:#ff3d3d;background:transparent;}"
            ".meta{display:flex;justify-content:space-between;font-size:12px;color:#cfcfcf;gap:10px;}"
            "@media (max-width:640px){button{padding:6px 8px;font-size:12px;}}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<h2 class='title'>Live detect preview</h2>"
            "<div class='player'>"
            "<img id='stream' src='/stream'/>"
            "<div class='controls'>"
            "<div class='row'>"
            "<button id='pauseBtn' onclick='togglePause()'>Pause</button>"
            "<button onclick='seekRelative(-10)'>-10s</button>"
            "<button onclick='seekRelative(10)'>+10s</button>"
            "<button onclick='skipNext()'>Skip next person</button>"
            "</div>"
            "<input id='timeline' type='range' min='0' max='100' step='0.1' value='0'/>"
            "<div class='meta'>"
            "<span id='timeLabel'>0:00 / 0:00</span>"
            "<span id='status'>Ready</span>"
            "</div>"
            "</div></div></div>"
            "<script>"
            "let isDragging=false;"
            "let paused=false;"
            "const timeline=document.getElementById('timeline');"
            "function fmt(sec){"
            " sec=Math.max(0,Math.floor(sec||0));"
            " const m=Math.floor(sec/60);"
            " const s=String(sec%60).padStart(2,'0');"
            " return m+':'+s;"
            "}"
            "timeline.addEventListener('mousedown',()=>{isDragging=true;});"
            "timeline.addEventListener('touchstart',()=>{isDragging=true;},{passive:true});"
            "timeline.addEventListener('mouseup',()=>{isDragging=false;seekTo(+timeline.value);});"
            "timeline.addEventListener('touchend',()=>{isDragging=false;seekTo(+timeline.value);},{passive:true});"
            "timeline.addEventListener('change',()=>seekTo(+timeline.value));"
            "async function togglePause(){"
            " const r=await fetch('/toggle-pause',{method:'POST'});"
            " const d=await r.json();"
            " paused=!!d.paused;"
            " document.getElementById('pauseBtn').textContent=paused?'Play':'Pause';"
            " document.getElementById('status').textContent=paused?'Paused':'Playing';"
            "}"
            "async function seekRelative(delta){"
            " const r=await fetch('/seek-relative',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seconds:delta})});"
            " const d=await r.json();"
            " if(d && d.current_second!==undefined){"
            "  timeline.value=d.current_second;"
            " }"
            "}"
            "async function seekTo(second){"
            " const r=await fetch('/seek-to',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({second})});"
            " const d=await r.json();"
            " if(d && d.current_second!==undefined){"
            "  timeline.value=d.current_second;"
            " }"
            "}"
            "async function skipNext(){"
            " const r=await fetch('/skip-next',{method:'POST'});"
            " const d=await r.json();"
            " document.getElementById('status').textContent=d.message;"
            "}"
            "setInterval(async()=>{"
            " const r=await fetch('/status');"
            " const d=await r.json();"
            " if(!d){return;}"
            " if(!isDragging && d.current_second!==undefined){timeline.value=d.current_second;}"
            " if(d.duration_second!==undefined){timeline.max=Math.max(0,d.duration_second);}"
            " document.getElementById('timeLabel').textContent=fmt(d.current_second)+' / '+fmt(d.duration_second);"
            " paused=!!d.paused;"
            " document.getElementById('pauseBtn').textContent=paused?'Play':'Pause';"
            " if(d.current_second!==undefined){"
            "  document.getElementById('status').textContent=(paused?'Paused':'Playing')+' | frame '+d.current_frame+' | '+d.current_second.toFixed(1)+'s';"
            " }"
            f"}},{poll_interval_ms});"
            "</script>"
            "</body></html>"
        )

    @app.get("/stream")
    def stream() -> Response:
        return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/skip-next")
    def skip_next() -> Response:
        with state_lock:
            state["skip_requested"] = True
            current_frame = int(state["current_frame"])
        current_second = current_frame / native_fps
        return jsonify(
            {
                "ok": True,
                "message": f"Đang skip... từ {current_second:.1f}s",
                "current_frame": current_frame,
                "current_second": current_second,
            }
        )

    @app.get("/status")
    def status() -> Response:
        with state_lock:
            current_frame = int(state["current_frame"])
            last_skip_frame = state["last_skip_frame"]
            paused = bool(state["paused"])
        return jsonify(
            {
                "current_frame": current_frame,
                "current_second": current_frame / native_fps,
                "duration_second": total_duration,
                "last_skip_frame": last_skip_frame,
                "last_skip_second": (last_skip_frame / native_fps) if last_skip_frame is not None else None,
                "paused": paused,
            }
        )

    @app.post("/toggle-pause")
    def toggle_pause() -> Response:
        with state_lock:
            state["paused"] = not bool(state["paused"])
            paused = bool(state["paused"])
            current_frame = int(state["current_frame"])
        return jsonify(
            {
                "ok": True,
                "paused": paused,
                "current_frame": current_frame,
                "current_second": current_frame / native_fps,
            }
        )

    @app.post("/seek-relative")
    def seek_relative() -> Response:
        body = request.get_json(silent=True) or {}
        delta_second = float(body.get("seconds", 0.0) or 0.0)

        with state_lock:
            current_frame = int(state["current_frame"])
            target_second = (current_frame / native_fps) + delta_second
            target_second = max(0.0, min(total_duration, target_second))
            target_frame = int(round(target_second * native_fps))
            if total_frames > 0:
                target_frame = max(0, min(total_frames - 1, target_frame))
            state["seek_frame"] = target_frame
        return jsonify(
            {
                "ok": True,
                "current_frame": target_frame,
                "current_second": target_frame / native_fps,
            }
        )

    @app.post("/seek-to")
    def seek_to() -> Response:
        body = request.get_json(silent=True) or {}
        second = float(body.get("second", 0.0) or 0.0)
        second = max(0.0, min(total_duration, second))
        target_frame = int(round(second * native_fps))
        if total_frames > 0:
            target_frame = max(0, min(total_frames - 1, target_frame))

        with state_lock:
            state["seek_frame"] = target_frame
        return jsonify(
            {
                "ok": True,
                "current_frame": target_frame,
                "current_second": target_frame / native_fps,
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Live browser preview: person detect + tracking")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8787, help="Port")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument(
        "--process-width",
        type=int,
        default=960,
        help="Resize frame width before detection/tracking for higher FPS (0 = original size)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=60,
        help="Preview JPEG quality (lower is faster, range 35-95)",
    )
    parser.add_argument(
        "--status-poll-ms",
        type=int,
        default=500,
        help="Browser status polling interval in milliseconds",
    )
    parser.add_argument(
        "--zone",
        type=int,
        nargs="+",
        default=_default_zone_cli_values(),
        help="Package zone polygon as x1 y1 x2 y2 ...",
    )
    args = parser.parse_args()

    if len(args.zone) < 6 or len(args.zone) % 2 != 0:
        raise SystemExit("--zone must contain an even number of values and at least 3 points")

    zone_points = [(args.zone[i], args.zone[i + 1]) for i in range(0, len(args.zone), 2)]

    app = build_app(
        video_path=args.video,
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        zone_points=zone_points,
        process_width=args.process_width,
        jpeg_quality=args.jpeg_quality,
        status_poll_ms=args.status_poll_ms,
    )
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
