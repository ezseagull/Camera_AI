# Camera AI: thu ảnh mặt theo track tại cửa ra vào

Mẫu Python cho **video đã ghi**, một camera cố định, nhiều người và chưa có cơ sở
dữ liệu danh tính. Đầu ra là các track tạm thời cùng những ảnh mặt được chọn để
bạn kiểm tra và sử dụng cho bước nhận diện về sau.

## 1. Pipeline và lựa chọn thư viện

1. Hiệu chỉnh méo bằng calibration nếu có.
2. Crop một cửa sổ cố định quanh ROI và vùng đệm; YOLO26s phát hiện người.
3. BoT-SORT cập nhật **mọi frame**, dùng motion và appearance để duy trì track.
4. SCRFD-10G tìm mặt trên crop từng người, mặc định mỗi 2 frame.
5. Khi không thấy mặt, thử xoay crop ±30° theo lịch thưa hơn; gộp detection trùng.
6. Gắn mặt vào người bằng matching một-một, loại trường hợp chủ sở hữu mơ hồ.
7. Chỉ thu mặt có tâm nằm trong ROI, lọc chất lượng và giữ tối đa 5 ảnh/track.

**Tracking người là nhánh chính.** Khi không thấy mặt, track người vẫn được cập
nhật. `Track 17` chỉ có nghĩa là một đoạn theo dõi trong lần chạy này; không phải
tên người, không đảm bảo là người thứ 17 và không dùng để đếm người duy nhất.

| Thành phần | Lựa chọn |
|---|---|
| Person detection | Ultralytics YOLO26s; đổi sang YOLO26n để giảm chi phí |
| Tracking | BoT-SORT với `with_reid: true`, `persist=True` |
| Face detection | InsightFace 2.0, detector SCRFD-10G trong `buffalo_l` |
| Inference mặt | ONNX Runtime, CPU hoặc NVIDIA CUDA |
| ROI, video, calibration | OpenCV |
| Gắn mặt với người | SciPy Hungarian + geometry gates + ambiguity rejection |

`model: auto` trong BoT-SORT dùng feature phù hợp nếu có. Với YOLO26 end-to-end,
phiên bản đã chọn có thể chuyển sang `yolo26n-cls.pt`. Đây là appearance embedding
cho tracking, không phải tra danh tính khuôn mặt; cũng không đồng nghĩa một
person-ReID model chuyên biệt đã được huấn luyện cho camera của bạn.

## 2. Cài đặt

Nên dùng môi trường riêng với Python 3.10–3.12. Hai API chính được khóa ở
`ultralytics==8.4.158` và `insightface==2.0`, đối chiếu ngày 22/09/2026.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Sau khi kích hoạt:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU chạy được nhưng tốc độ phụ thuộc số người và số crop; chưa có benchmark cho
camera của bạn. Khi dùng NVIDIA, cài bản PyTorch phù hợp từ trang cài đặt chính
thức, rồi thay ONNX Runtime CPU bằng GPU **sau** lệnh cài requirements:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Không để hai bản ONNX Runtime cùng tồn tại. Kiểm tra CUDA/cuDNN theo bảng tương
thích chính thức. Nếu cài lại InsightFace, kiểm tra xem CPU runtime có được cài
lại không. `--face-device cuda` sẽ báo lỗi nếu CUDA provider không hoạt động;
`auto` cho phép chạy CPU và in provider đang sử dụng. `--device 0` chọn GPU cho
YOLO, độc lập với lựa chọn provider mặt. Mẫu này dùng CUDA device 0 cho mặt.

Lần chạy inference đầu cần Internet để tải YOLO, classifier fallback nếu cần,
và pack `buffalo_l`. Pack có thể chứa model khác, nhưng code chỉ bật module
`detection`. Có thể dùng detector ONNX cục bộ có 5 landmarks qua `face.onnx_path`.
Không tải/xây database embedding danh tính trong pipeline này.

## 3. Vẽ ROI

Đặt video ở thư mục dự án hoặc dùng đường dẫn tuyệt đối:

```bash
python run.py --source door.mp4 --edit-roi
```

- Kéo các chấm để đổi vị trí/kích thước đa giác.
- Chuột trái thêm đỉnh; chuột phải xóa đỉnh gần nhất.
- `R` xóa các đỉnh để vẽ lại theo thứ tự quanh vùng cửa.
- `Enter` lưu; `Esc` hủy.

Chọn frame dễ nhìn hơn nếu cần:

```bash
python run.py --source door.mp4 --edit-roi --at-second 5
```

Editor lưu `roi.json`. Nếu file này đã có, nó ghi đè các `roi.points` trong YAML.
Muốn điều chỉnh trực tiếp YAML, xóa/đổi tên `roi.json` hoặc đổi `roi.file`.
Tọa độ chuẩn hóa `[x,y]` nằm trong `[0,1]`; ví dụ `[0.25,0.10]` tương ứng khoảng
25% chiều ngang và 10% chiều cao. Đa giác mặc định chỉ là ví dụ vì chưa có video.

**ROI phải bao phủ vị trí khuôn mặt đi qua**, không chỉ sàn hay chân người.

Trong ảnh/video kết quả:

- Đường **xanh lá**: ROI cho phép thu ảnh mặt; xét tâm bbox mặt.
- Khung **xanh cyan**: cửa sổ tracking gồm ROI và vùng đệm.

Người được phát hiện trong toàn bộ khung cyan. Dùng cửa sổ chữ nhật này tránh
cắt mất thân người ngay mép ROI; chỉ ảnh mặt trong đa giác xanh lá được thu.
Giảm/tăng `roi.context_margin_fraction` để thay vùng đệm. Không di chuyển cửa sổ
tracking giữa các frame vì sẽ phá hệ tọa độ của motion model.

Máy không có GUI/Colab: sửa ROI trong YAML hoặc JSON, rồi xem ảnh preview:

```bash
python run.py --source door.mp4 --preview-roi
```

Lệnh này tạo `roi_preview.jpg` và không nạp model. Nếu OpenCV báo không hỗ trợ
cửa sổ, dùng `opencv-python` trên máy có desktop; tránh cài đồng thời nhiều gói
OpenCV cùng cung cấp module `cv2`. Server không có màn hình dùng `--headless`.

## 4. Chạy thu ảnh

```bash
python run.py --source door.mp4
```

NVIDIA:

```bash
python run.py --source door.mp4 --device 0 --face-device cuda
```

CPU/server:

```bash
python run.py --source door.mp4 --device cpu --face-device cpu --headless
```

Chạy thử 300 frame:

```bash
python run.py --source door.mp4 --max-frames 300
```

`Q`, `Esc` hoặc Ctrl+C kết thúc và ghi metadata. Mỗi lần chạy tạo thư mục riêng
trong `outputs`, để các ID từ nhiều lần chạy không bị gộp nhầm.

| Đầu ra trong mỗi lần chạy | Ý nghĩa |
|---|---|
| `annotated.mp4` | ROI, bbox người/mặt, track ID, đường đi; không kèm audio |
| `roi_preview.jpg` | Kiểm tra vùng thu mặt và vùng đệm |
| `faces/track_000001/frame_..._crop.jpg` | Crop từ frame sạch ở độ phân giải đang xử lý |
| `faces/track_000001/frame_..._aligned.jpg` | Alignment 112×112 nếu 5 landmarks cho phép |
| `faces/track_000001/meta.json` | Ảnh đang giữ, score, bbox, landmarks, frame, thời gian |
| `tracks.jsonl` | Track đã kết thúc, kể cả track không thu được mặt |
| `run.json` | Cấu hình, ROI thực tế, phiên bản thư viện, thống kê lọc và thời gian chạy |
| `botsort_runtime.yaml` | Cấu hình tracking thật, gồm buffer tính theo FPS video |

Các ảnh và tọa độ thuộc **full processed frame**: sau undistortion nếu bật;
không phải tọa độ video méo ban đầu. Ảnh được cắt trước khi vẽ overlay. Alignment
chỉ xoay/co giãn/tịnh tiến; không tự tạo góc mặt chính diện.

Score là heuristic gồm độ tin cậy, kích thước, độ nét, độ sáng và đối xứng nhẹ.
Nó không phải xác suất nhận diện đúng hay mô hình face-quality đã được hiệu
chuẩn. `symmetry_hint` không phải phép đo yaw/pitch theo độ. Crop đạt 32 px có thể
được lưu làm bằng chứng, chưa chắc đủ tốt để làm ảnh đăng ký nhận diện.

Top-K được cập nhật khi có ảnh tốt hơn. Các frame gần nhau trong 0.4 giây cạnh
tranh nhau để hạn chế ảnh lặp; đây không phải khử trùng bằng embedding danh tính.

## 5. Xử lý góc cao và méo ống kính

Hai vấn đề khác nhau:

**Góc nhìn cao:** code tìm mặt ở crop người với độ phân giải tương đối lớn, tìm
trên toàn bbox người mặc định, thử xoay để hỗ trợ roll, và tích lũy ảnh qua thời
gian. Không giới hạn cứng khuôn mặt ở 1/3 trên vì cơ thể trong góc cao có thể
nghiêng. Xoay ảnh không sửa được pitch/yaw hay phần mặt bị che. Nếu toàn clip chỉ
thấy đỉnh đầu, không có đủ thông tin để thu ảnh mặt dùng cho nhận diện; cần đổi
vị trí/góc camera hoặc thêm camera gần ngang mặt. Upscale không tạo chi tiết thật.

**Méo quang học:** các cạnh vốn thẳng bị uốn cong có thể là barrel distortion
hoặc fisheye. Không kết luận loại lens chỉ từ mô tả. Bắt đầu bằng
`undistort.mode: none` nếu chưa có calibration; không điền K/D đoán mò.

Để hiệu chỉnh, chụp 20–30 ảnh bàn cờ bằng đúng camera, cùng chế độ ảnh/zoom/focus
và độ phân giải với video, đổi vị trí/góc bàn cờ để phủ cả giữa và rìa ảnh. Giữ
camera cố định, di chuyển bảng. Một chuỗi ảnh gần giống nhau không thay thế được
các góc nhìn đa dạng. Video cửa thông thường không tự cung cấp calibration này.

Ví dụ bàn cờ 9×6 **góc trong**, cạnh mỗi ô 25 mm:

```bash
python calibrate.py --images "calib/*.jpg" --model pinhole --cols 9 --rows 6 --square-size 25 --out camera_calibration.npz
```

Nếu xác định đúng là lens fisheye:

```bash
python calibrate.py --images "calib/*.jpg" --model fisheye --cols 9 --rows 6 --square-size 25 --out camera_calibration.npz
```

Đặt `undistort.mode` đúng loại file vừa tạo, rồi **vẽ lại ROI**. Code kiểm tra
signature và không dùng ROI cũ khi phép biến đổi đổi. K được scale khi đổi độ
phân giải cùng tỉ lệ; đổi crop/FOV/chế độ sensor vẫn cần calibration phù hợp.
Không hiệu chỉnh hai lần nếu camera đã xuất video dewarp.

Quan sát các cạnh thẳng và mặt ở rìa ảnh; RMS thấp không đủ chứng minh calibration
tốt. Homography/bird's-eye-view của mặt phẳng sàn không thay thế calibration lens
và không phù hợp để biến khuôn mặt 3D thành chính diện.

## 6. Những tham số nên chỉnh trên video thật

| Triệu chứng | Điều chỉnh đầu tiên |
|---|---|
| Mất track ngay mép vùng cửa | Mở vùng đệm `context_margin_fraction` khoảng 0.12–0.20 |
| Mất dấu ngắn do che nhau | Thử `lost_seconds: 2–3`; buffer quá dài cũng có thể tăng gán nhầm |
| Bỏ sót người góc cao | Thử `imgsz: 1280`, model lớn hơn hoặc fine-tune ảnh từ camera |
| Có người nhưng ít detection mặt | Thử `face.input_size: [960,960]`, mở vùng ROI và crop người |
| Người nghiêng theo hướng camera | Thử `rotation_angles: [-45,-30,30,45]`; chi phí tăng |
| Có mặt khác trong crop khiến không thử rotation | Thử `rotate_only_if_empty: false` để quét mọi góc theo lịch |
| Crop lẫn nhiều người, gán mặt mơ hồ | Xem bbox/ROI, điều chỉnh prior `expected_head_xy`; không hạ ambiguity tùy tiện |
| Cần ảnh enrollment rõ hơn | Tăng `quality.min_face_px` lên 64 hoặc 80 và kiểm tra bằng mắt |
| Lưu nhiều ảnh mờ | Tăng `min_sharpness` dựa trên score các ảnh thật; kiểm tra shutter/exposure |
| Chậm | YOLO26n, ít góc xoay hơn, hoặc `face.every_n_frames: 3`; tracker vẫn chạy mọi frame |

`rotation_every_n_frames` nên là bội số của `every_n_frames`; nếu không, rotation
chỉ chạy ở frame chia hết cho cả hai. `person.confidence` phải thấp hơn hoặc bằng
`track_low_thresh`; đặt cao sẽ loại detection yếu trước khi tracker được dùng.

**Không có bảo đảm giữ ID tuyệt đối.** ReID bị giới hạn bởi appearance, khoảng
cách và spatial gate. Mất dấu lâu, người ra khỏi khung rồi quay lại, quần áo tương
tự hoặc hai người che sát nhau có thể gây đổi ID/nhầm ID. Mẫu ưu tiên bỏ các gán
mặt mơ hồ, không tự gộp các ID chỉ vì gần nhau. Cần xem lại clip có hai người đi
sát nhau và kiểm tra xem một thư mục track có lẫn mặt hay không trước khi tạo
database nhận diện.

Thời gian mẫu = chỉ số frame / FPS của video, phù hợp video ghi có frame rate cố
định. Với VFR/RTSP cần thêm timestamp PTS/thời gian thực, xử lý reconnect và đồng
bộ buffer; phần đó không nằm trong mẫu cho video đã ghi này.

## 7. Kiểm tra đã thực hiện

Chạy `python -m pip install pytest` rồi `python -m pytest -q`.

17 test pass: tọa độ ROI; rotation ±30°/±90° và ánh xạ ngược landmarks; matching
một-một và loại ambiguity; NMS; alignment; calibration pinhole/fisheye và kiểm
tra ROI cũ; lọc chất lượng; giới hạn top-K; pipeline với backend giả lập xuất MP4,
cập nhật đủ frame và lưu crop sạch đúng ROI.

Đã đối chiếu API với source wheel của hai phiên bản thư viện khóa trong
requirements. **Chưa chạy inference với pretrained weights hoặc benchmark trên
video cửa thực tế của bạn**, vì chưa có video trong yêu cầu. Test giả lập không
đánh giá recall, ID switch hoặc tốc độ GPU. Khi chạy thật, ghi lại số lượt người,
số track tách/nhầm, tỉ lệ lượt có ít nhất một ảnh mặt dùng được và throughput.

## 8. Nguồn và quyền sử dụng model

- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- API tracking: https://docs.ultralytics.com/reference/trackers/track/
- InsightFace: https://pypi.org/project/insightface/
- Model zoo: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
- OpenCV calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- OpenCV fisheye: https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
- PyTorch install: https://pytorch.org/get-started/locally/
- ONNX CUDA: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

Ultralytics có lựa chọn AGPL-3.0/Enterprise. Code InsightFace là MIT, còn các
pretrained model họ cung cấp được ghi dành cho nghiên cứu phi thương mại.
Khi triển khai sản phẩm thương mại, chọn giấy phép/model phù hợp; có thể thay
detector qua `face.onnx_path` bằng model tương thích có quyền sử dụng tương ứng.

Thông tin giấy phép chính thức:
https://www.ultralytics.com/license và model-zoo ở trên.
