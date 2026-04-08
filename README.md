# Ship Room Monitoring

Pipeline detect va track nguoi trong phong kho, su dung YOLOv8 + DeepSORT.

## Kien truc hien tai
- `src/module1/video_input.py`: doc video/camera va sample frame theo `sample_fps`.
- `src/module2/person_tracker.py`: detect person (YOLO) + track ID (DeepSORT) + check vao `package_zone`.
- `deep_sort/`: ma DeepSORT da duoc nhung truc tiep vao project.
- `src/pipeline_toan.py`: chay end-to-end va ghi su kien JSONL co timestamp.
- `src/module1/person_event_tour.py`: tong hop timeline xuat hien theo track.
- `src/module2/live_preview_web.py`: live preview web.

## Tinh nang dang dung
- Detect/track theo stack tu repo tham chieu (YOLOv8m + DeepSORT).
- Luu timestamp UTC/local va elapsed_seconds cho tung event.

## Cai dat
1. Python 3.10+.
2. Cai dependencies:
   - `pip install -r requirements.txt`

## Chay nhanh
- Pipeline:
  - `python -m src.pipeline_toan --video data/sample2.mp4`
  - Hoac van co the dung file json: `python -m src.pipeline_toan --config configs/toan_config.sample1.json`
- Chay full (pipeline + export timeline + live preview):
  - `run_project`
- Mac dinh script se uu tien GPU (`CUDA_VISIBLE_DEVICES=0`). Neu muon CPU, dat `detection.repo_device` thanh `cpu` trong file config.

## Cau hinh chinh
- Shared default config dat trong `src/config.py` (ap dung chung cho moi video)
- Co the override nhanh bang CLI: `--video`, `--timezone`
- File mau (tuy chon): `configs/toan_config.sample1.json`
- Model: `yolov8m.pt`
- DeepSORT weights: `deep_sort/deep/checkpoint/ckpt.t7`

## Output
- Event track: `artifacts/events/person_tracks.jsonl`
- Timeline appearance: `artifacts/events/person_appearance_tour.json`
