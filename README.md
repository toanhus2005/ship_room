# Ship Room Monitoring - Workstream Toàn (Module 1 + 2)

## Scope
Phần này triển khai cho Toàn:
- Module 1: nhận video/camera, tách frame, gắn timestamp.
- Module 2: phát hiện người, tracking ID qua nhiều frame, đánh dấu người có vào vùng gói hàng hay không.

## Cấu trúc chính
- [src/module1/video_input.py](src/module1/video_input.py): đọc video/camera và sinh frame theo thời gian.
- [src/module2/person_tracker.py](src/module2/person_tracker.py): detect + track người, kiểm tra vào `package_zone`.
- [src/pipeline_toan.py](src/pipeline_toan.py): chạy pipeline end-to-end và xuất sự kiện JSONL.
- [configs/toan_config.example.json](configs/toan_config.example.json): cấu hình nguồn video, ROI, ngưỡng detect/track.

## Cài đặt
1. Tạo môi trường Python 3.10+.
2. Cài dependencies:
   - `pip install -r requirements.txt`

## Chạy thử
- Chuẩn bị video tại `data/ship_room_sample.mp4` (hoặc sửa `video.source` trong config).
- Chạy:
  - `python -m src.pipeline_toan --config configs/toan_config.example.json`

## Chạy tự động (Windows)
- Chạy toàn bộ setup + pipeline bằng một lệnh:
  - `run_project.bat`
- Dùng config khác:
  - `run_project.bat configs\\your_config.json`

Script sẽ tự:
- tạo `.venv` nếu chưa có,
- cài dependencies từ `requirements.txt`,
- chạy pipeline với config bạn truyền vào.

## Output
- File sự kiện: `artifacts/events/person_tracks.jsonl`
- Mỗi dòng gồm:
  - `frame_index`, `track_id`, `xyxy`, `confidence`
  - `in_package_zone`
  - `timestamp_utc`, `timestamp_local`

## Gợi ý bàn giao cho Lâm (Module 3)
Lâm có thể dùng trực tiếp `person_tracks.jsonl` để tìm:
- thời điểm người đi vào vùng package,
- khoảng dừng gần package,
- các mốc nghi vấn tiếp cận.

## Báo cáo công việc cá nhân
- File báo cáo theo yêu cầu team: `docs/toan_work_report.md`

## Task checklist (20/3 - 25/3)
- [x] Nhận video từ camera/file
- [x] Tách frame theo tần suất mẫu
- [x] Đồng bộ timestamp
- [x] Detect người theo frame
- [x] Track ID cùng người qua nhiều frame
- [x] Cờ người vào vùng gói hàng
- [ ] Tinh chỉnh ROI theo camera thực tế
- [ ] Đánh giá độ chính xác trên dữ liệu thật
