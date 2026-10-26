# Sensor Logger: cập nhật 26.000 mẫu

Giữ nguyên templates, CSS, JavaScript và đăng ký/đăng nhập.

## Huấn luyện lại

Cài `python -m pip install -r requirements.txt`, sau đó chạy:

```bash
python train_sensor_model.py --data /path/to/SensorLogger_26000.zip
```

Có thể dùng thư mục đã giải nén hoặc trên Kaggle:

```bash
python train_sensor_model.py --data /kaggle/input/sensor-logger-dataset
```

Script yêu cầu 20.800 mẫu không nhãn và 5.200 mẫu có nhãn, 1.040 mỗi lớp.
PCA học biểu diễn từ tập không nhãn; Logistic Regression học phân loại từ tập có nhãn. Đây là baseline học máy, không phải TF-C/DINOv3. Không dùng nhãn trong provenance để huấn luyện PCA.

Các bản tăng cường cùng tổ tiên được giữ chung nhóm train/validation/test. Không có mã phiên thu/người dùng gốc đầy đủ nên kết quả chưa chứng minh khả năng tổng quát trên người hoặc phiên thu mới. Không coi 26.000 cửa sổ là 26.000 lần thu độc lập.

Các file xuất: `models/gesture_real_v1.joblib`, `models/training_report.json`, `models/evaluation_split.csv`. File model được web nạp khi khởi động lại.

Đầu vào web: CSV chứa `acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z` của một cử chỉ, hoặc ZIP Sensor Logger có Accelerometer.csv và Gyroscope.csv. ZIP đồng bộ hai cảm biến bằng thời gian. Với tín hiệu dài hơn 178 dòng, chọn đoạn chuyển động mạnh nhất; cùng một cách trích xuất đặc trưng được dùng khi huấn luyện và dự đoán. Cần thống nhất đơn vị cảm biến và hướng đặt máy với dữ liệu nguồn.

`rotate` được ánh xạ sang mã icon `circle` sẵn có để giữ giao diện, nhãn kết quả là Xoay cổ tay. Không sử dụng tên file hay cột label để dự đoán.

Hướng dẫn UWave trong README và `train_real_model.py` chỉ áp dụng cho mô hình cũ. Với dữ liệu Sensor Logger, dùng hướng dẫn này. Không chạy lại script UWave nếu muốn giữ mô hình Sensor Logger.

## Lần huấn luyện ngày 27/10/2026

Nguồn: file `Sensor logger(1).zip` do người dùng gửi. Đây là bản tín hiệu đã tăng nhiễu nhân tạo; không phải 26.000 bản ghi thu độc lập. Link tải Kaggle không khả dụng trong lần chạy này nên mô hình được huấn luyện từ ZIP đã gửi, không khẳng định đã dùng phiên bản Kaggle mới nhất.

- SHA-256: `451c770fedf11b305a8d03ab5cb79140671d23686bcce6242443f0540918e643`.
- PCA: 20.800 mẫu không nhãn; bộ phân loại: 4.160 mẫu có nhãn; test: 1.040 mẫu.
- Chọn C = 0,1 bằng validation; accuracy test = 49,23%; macro-F1 = 49,31%.
- Kết quả đánh giá trên bản dữ liệu nhiễu; không dùng để kết luận hiệu quả trên cử chỉ thực tế hoặc người dùng mới. Accuracy khác với confidence của từng dự đoán; confidence vẫn có thể cao ở một số mẫu.

Lệnh tái lập: `python train_sensor_model.py --data "Sensor logger(1).zip"`.
