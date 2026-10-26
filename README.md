# Cập nhật Sensor Logger 26.000 mẫu

Huấn luyện lại ngày 27/10/2026 từ bản ZIP mới đã tăng nhiễu: accuracy test **49,23%**, macro-F1 **49,31%** trên 1.040 mẫu. Nguồn và giới hạn đánh giá được ghi trong [hướng dẫn huấn luyện](SENSOR_LOGGER_TRAINING.md).

Mô hình hiện tại dùng 6 kênh Sensor Logger; xem [hướng dẫn huấn luyện](SENSOR_LOGGER_TRAINING.md) và `models/training_report.json` để biết phương pháp, kết quả và giới hạn. Giao diện giữ nguyên.

---

## Tài liệu mô hình UWave trước đây

# Gesture AI — Nhận diện 5 cử chỉ bằng model huấn luyện trên dữ liệu thật

Website Flask theo thiết kế ba cột: **Dữ liệu đầu vào → Mô hình AI → Kết quả**. Có đăng ký/đăng nhập, tải CSV/TXT và 5 hình minh họa bàn tay thay đổi theo dự đoán.

## 1. Những gì đã nâng cấp

- **Model:** bỏ Logistic Regression huấn luyện trên *dữ liệu giả*. `train_real_model.py` tải bộ **UWaveGestureLibrary** chứa các bản ghi **gia tốc X/Y/Z thật**. So sánh **SVM RBF** với **Logistic Regression** trên tập validation, chọn mô hình có Accuracy validation cao hơn, sau đó **đánh giá trên tập test độc lập** và lưu file model `.joblib`.
- **5 nhãn:** ID gốc 3 = **Vuốt phải**, 4 = **Vuốt trái**, 5 = **Vuốt lên**, 6 = **Vuốt xuống**, 7 = **Xoay vòng tay**. Quy ước dựa trên hình đường đi của cử chỉ trong bài báo uWave, không phải đo hướng xoay khớp cổ tay riêng biệt.
- **Giao diện:** khi có dự đoán, mã JavaScript đọc `gesture_key` do API trả về và hiển thị đúng `static/gestures/{right,left,up,down,circle}.svg`. Hình vẽ KHÔNG cố định ở “Vuốt phải”, cũng không đọc nhãn từ tên file.
- **API:** `/api/predict` trả về nhãn, `gesture_key`, confidence, xác suất của 5 nhãn và tên model. `/api/health` cho biết model thật đã có trên server chưa.
- **Giữ nguyên:** chức năng đăng ký, đăng nhập, `SECRET_KEY`, CSRF và kết nối Render.

> **Lưu ý quan trọng:** Model được huấn luyện bằng dữ liệu gia tốc thật từ bộ UWave (thiết bị trong nghiên cứu), không phải bộ CSV thu trực tiếp từ iPhone của bạn. Khác biệt thiết bị, hướng đặt điện thoại và tốc độ đo có thể làm giảm độ chính xác. “Xoay vòng tay” là động tác di chuyển tay tạo vòng tròn, không thể xem là thao tác xoay riêng cổ tay. Các hình là minh họa của **nhãn model dự đoán**, không bảo đảm model đoán đúng mọi mẫu.

## 2. Dữ liệu thật

- **Nguồn:** [UWaveGestureLibrary trên trang UEA](https://timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary)
- **File nguồn:** [THUML/Time-Series-Library trên Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/UWaveGestureLibrary)
- **Số liệu dataset tổng:** TRAIN 2.238, TEST 2.241, 8 lớp, 3 trục, 315 điểm mẫu; script chỉ lấy **5 lớp 3,4,5,6,7**, nên số mẫu thực tế ít hơn.
- **Phương pháp kiểm tra:** chọn mô hình bằng validation tách từ TRAIN; **không chọn tham số bằng TEST**. Thông số Accuracy và Macro-F1 được ghi tự động sau khi train, tại `models/training_report.json`.
- **Quyền dữ liệu:** nguồn UWave gốc có lưu ý về điều kiện tái phân phối. Repo này **không chứa file dữ liệu gốc**. Model được tạo bởi bạn khi chạy script; nhớ ghi nguồn khi báo cáo.

## 3. Cách nâng cấp trên Windows/VS Code

Mở Terminal tại thư mục có `app.py`:

```powershell
cd E:\gesture_ai_web
python -m pip install -r requirements.txt
python train_real_model.py
```

Khi thành công Terminal sẽ in:

```
TRAIN: ... mẫu; TEST độc lập: ... mẫu
Validation | SVM RBF: ...%
Validation | Logistic Regression: ...%
Chọn mô hình theo validation: ...
Accuracy: ...% | Macro-F1: ...%
Đã xuất model THẬT: ...\models\gesture_real_v1.joblib
```

Hai file mới sẽ xuất hiện:

```
models/gesture_real_v1.joblib      # Model thật đã huấn luyện
models/training_report.json        # Thông tin dữ liệu và chỉ số thật
```

Nếu mạng không tải được dữ liệu, mở link Hugging Face ở mục 2, tải hai file:

```
UWaveGestureLibrary_TRAIN.ts
UWaveGestureLibrary_TEST.ts
```

và đặt tại `data_external/UWaveGestureLibrary/` rồi chạy lại `python train_real_model.py`.

### Kiểm tra giao diện với CSV từ dữ liệu thật

```powershell
python export_real_test_examples.py
python app.py
```

Truy cập `http://127.0.0.1:5000`, đăng ký/đăng nhập, rồi tải lần lượt:

```
samples_real/right_real_sample.csv
samples_real/left_real_sample.csv
samples_real/up_real_sample.csv
samples_real/down_real_sample.csv
samples_real/circle_real_sample.csv
```

Chọn **Chạy dự đoán**, hình bàn tay và mũi tên sẽ thay đổi theo kết quả. Script xuất những ví dụ được model dự đoán đúng để tiện **thử giao diện**; **không** dùng chúng làm số liệu đánh giá. Các file này không đưa lên GitHub.

### CSV từ điện thoại

File `.csv`/`.txt` cần có 3 cột `ax,ay,az` hoặc `x,y,z` và từ 20 đến 10.000 dòng. Ví dụ:

```csv
ax,ay,az
0.12,0.25,0.92
0.15,0.28,0.91
0.18,0.32,0.90
```

Lưu ý ví dụ trên có 3 dòng chỉ để minh họa định dạng, **không đủ để dự đoán**. Nên ghi một động tác liên tục và cắt thành một file CSV cho mỗi lần thực hiện. iPhone cần hiệu chỉnh quy ước trục/hướng đặt máy hoặc thu thập thêm dữ liệu riêng để mô hình tổng quát tốt.

## 4. Cập nhật GitHub và Render

Sau khi chạy thành công `python train_real_model.py`, phải kiểm tra `models/gesture_real_v1.joblib` đã tồn tại rồi cập nhật code:

```powershell
cd E:\gesture_ai_web
# Bỏ các file demo mô phỏng cũ đã được Git theo dõi:
git rm -- train_model.py models/logistic_gesture_demo.joblib samples/vuot_phai_demo.csv samples/xoay_co_tay_demo.csv
# Nếu file đã bị xóa trước đó, không cần chạy lại git rm.
git add .gitignore README.md app.py ml.py train_real_model.py export_real_test_examples.py templates/index.html static/app.js static/style.css static/gestures models/gesture_real_v1.joblib models/training_report.json
git commit -m "Train real gesture model and add correct gesture images"
git push origin main
```

Render tự deploy commit mới (nếu bật Auto Deploy). Link hiện có **vẫn giữ nguyên**:

**https://gesture-ai-web.onrender.com**

Nếu có lỗi, xem Render **Logs**. Kiểm tra `https://gesture-ai-web.onrender.com/api/health`: `model_ready` phải là `true` và `training_source` chứa `REAL UWaveGestureLibrary`.

## 5. Các file quan trọng

| File | Vai trò |
|---|---|
| `train_real_model.py` | Tải, huấn luyện, chọn và đánh giá model trên dữ liệu thật |
| `ml.py` | Trích xuất đặc trưng gia tốc cho TRAIN và dự đoán |
| `models/gesture_real_v1.joblib` | **Được tạo sau khi chạy script**, không giả lập |
| `models/training_report.json` | Độ chính xác thực tế do script tính |
| `app.py` | Đọc model thật, xử lý CSV, trả nhãn và mã icon |
| `static/gestures/*.svg` | 5 hình bàn tay + mũi tên đúng hướng |
| `static/app.js` | Dùng `gesture_key` để đổi ảnh khi nhấn Dự đoán |
| `templates/index.html` | Giao diện ba khối và đăng nhập/đăng ký |

## 6. Nguồn tham khảo

- J. Liu và cộng sự, *uWave: Accelerometer-based personalized gesture recognition and its applications*, IEEE PerCom 2009, [bài báo và hình 8 động tác](https://www.yecl.org/publications/liu09percom.pdf).
- [UEA UWaveGestureLibrary](https://timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary).
- [Bộ dữ liệu TS trên Hugging Face, THUML](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/UWaveGestureLibrary).

**Ghi rõ khi nộp thầy:** Đây là mô hình phân loại cử chỉ từ dữ liệu thật UWave (SVM/Logistic Regression được chọn theo validation), **không phải backbone TF-C + DINOv3**. TF-C + DINOv3 là bài tập khác, vẫn ở [repository riêng](https://github.com/vuvandat109/TF-C-DINOv3).
