# Ship Room Monitoring

He thong giam sat phong/khu dong goi, phat hien va theo doi nguoi theo video/camera.
Stack hien tai:
- Detection: YOLOv8 (mac dinh `yolov8m.pt`)
- Tracking: ByteTrack (thong qua `YOLO.track`)
- Output: JSONL event + JSON timeline
- Preview: Flask + MJPEG web stream

## 1) Muc tieu
- Doc video file, camera index, hoac RTSP.
- Detect nguoi (class `person`) tren tung frame mau.
- Gan track ID theo thoi gian.
- Kiem tra nguoi co vao `package_zone` hay khong.
- Xuat du lieu su kien de phan tich va xem lai nhanh.

## 2) Kien truc du an
- `src/config.py`: schema config bang Pydantic + default value.
- `src/module1/video_input.py`: mo source, sample frame theo `sample_fps`, sinh timestamp UTC/local.
- `src/module2/person_tracker.py`: YOLO detect person + ByteTrack + zone trigger.
- `src/pipeline_toan.py`: pipeline batch offline, ghi event JSONL.
- `src/module1/person_event_tour.py`: tong hop timeline xuat hien (uu tien doc tu tracks JSONL neu co).
- `src/module2/live_preview_web.py`: preview web, pause/seek/skip-next-person.
- `run_project.bat`: chay full flow (preview + pipeline + export timeline).
- `run_preview.bat`: chi chay live preview.

## 3) Yeu cau he thong
- Windows (script `.bat`/`.cmd` da san sang).
- Python 3.10+.
- Khuyen nghi GPU NVIDIA + CUDA de dat FPS/throughput tot hon.

## 4) Cai dat
### Cach A - dung script (de nhat tren Windows)
Chi can chay:
- `run_project.cmd` (chay full)
- `run_preview.cmd` (chi preview)

Script se tu dong:
- Tao `.venv` neu chua co.
- Cai dependency tu `requirements.txt`.
- Kich hoat moi truong va chay module can thiet.

### Cach B - cai dat thu cong
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5) Cach chay
### 5.1 Chay full pipeline (de xuat)
```bat
run_project.cmd
```
Hoac truyen config:
```bat
run_project.cmd configs\toan_config.sample1.json
```

Mac dinh script dung:
- Config fallback: `configs\toan_config.img_5812.json`
- Mo preview tai `http://127.0.0.1:8787`
- Chay pipeline:
  - `python -m src.pipeline_toan --config <CONFIG> --force-track-buffer 30 --process-width 960`
- Export timeline track:
  - `python -m src.module1.person_event_tour --tracks-jsonl artifacts/events/person_tracks.jsonl ...`

### 5.2 Chi chay preview web
```bat
run_preview.cmd
```
Hoac truyen config:
```bat
run_preview.cmd configs\toan_config.sample2.json
```

Preview co cac dieu khien:
- Pause/Play
- Seek -10s / +10s
- Seek tren timeline
- Skip next person

### 5.3 Chay tay bang Python module
Pipeline co config:
```bash
python -m src.pipeline_toan --config configs/toan_config.sample1.json
```

Pipeline override nhanh video/timezone:
```bash
python -m src.pipeline_toan --video data/sample1.mp4 --timezone Asia/Ho_Chi_Minh
```

Preview tay:
```bash
python -m src.module2.live_preview_web --video data/sample1.mp4 --model yolov8m.pt --conf 0.35
```

Export timeline tu JSONL:
```bash
python -m src.module1.person_event_tour --video data/sample1.mp4 --tracks-jsonl artifacts/events/person_tracks.jsonl --gap-seconds 3.0 --out artifacts/events/person_appearance_tour.json
```

## 6) Cau hinh (JSON)
File config dung schema trong `src/config.py`:

```json
{
  "video": {
    "source": "data/sample1.mp4",
    "timezone": "Asia/Ho_Chi_Minh"
  },
  "stream": {
    "sample_fps": 5.0,
    "process_width": 960,
    "output_frames_dir": "artifacts/frames",
    "save_sampled_frames": false
  },
  "detection": {
    "model_name": "yolov8m.pt",
    "confidence_threshold": 0.3,
    "repo_device": "0"
  },
  "tracking": {
    "track_buffer": 150,
    "bytetrack_high_thresh": 0.5,
    "bytetrack_low_thresh": 0.1,
    "bytetrack_new_track_thresh": 0.5,
    "bytetrack_match_thresh": 0.8,
    "bytetrack_fuse_score": true,
    "repo_track_min_confidence": 0.3,
    "single_person_mode": false
  },
  "zone": {
    "package_zone_polygon": [[1040, 440], [1460, 440], [1460, 780], [1040, 780]]
  },
  "output": {
    "events_jsonl": "artifacts/events/person_tracks.jsonl"
  }
}
```

Ghi chu quan trong:
- `video.source`:
  - Duong dan file video, VD `data/sample1.mp4`
  - Camera index dang chuoi, VD `"0"`
  - URL RTSP
- `detection.repo_device`:
  - `"0"` de dung GPU 0
  - `"cpu"` de buoc chay CPU
- `zone.package_zone_polygon`: toa do polygon theo pixel anh.

## 7) Output
### 7.1 Event theo frame/track
File mac dinh: `artifacts/events/person_tracks.jsonl`

Moi dong JSONL gom cac truong chinh:
- `frame_index`
- `track_id`
- `xyxy`
- `confidence`
- `in_package_zone`
- `timestamp_utc`
- `timestamp_local`
- `elapsed_seconds`

### 7.2 Timeline xuat hien
File mac dinh: `artifacts/events/person_appearance_tour.json`

Neu co `--tracks-jsonl`, output la `track_events` (split theo `gap_seconds`).
Neu khong co `--tracks-jsonl`, module se scan video truc tiep bang detector.

## 8) Cac file script nhanh
- `run_project.cmd`/`run_project.bat`: full flow.
- `run_preview.cmd`/`run_preview.bat`: preview-only flow.

`run_preview.bat` se thu dong giai phong port `8787` neu dang bi process khac chiem.

## 9) Thu muc quan trong
- `configs/`: file config mau.
- `data/`: video input.
- `artifacts/events/`: JSONL + timeline JSON sau khi chay.
- `artifacts/preview_frames/`: tai nguyen preview.
- `yolov8m.pt`, `yolov8n.pt`: weight YOLO.
- `deep_sort/`: source DeepSORT cu, duoc giu lai de tham khao (khong con duoc goi trong pipeline hien tai).

## 10) Loi thuong gap va cach xu ly
### Khong mo duoc video
- Kiem tra lai `video.source` trong config.
- Neu dung camera, dam bao de dang chuoi so: `"0"`, `"1"`.

### CUDA khong available
- Script van chay tiep tren CPU.
- De chay CPU ro rang, dat `detection.repo_device` = `"cpu"`.

### Port 8787 da bi chiem
- `run_preview.bat` da co buoc giai phong port.
- Neu van loi, dong process dang dung cong nay roi chay lai.

## 11) Ghi chu phien ban
README nay duoc canh chinh de khop voi code hien tai trong workspace (pipeline, script, tham so CLI, duong dan output).
