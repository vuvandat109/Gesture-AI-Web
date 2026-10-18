# Gesture AI — Demo nhận diện 5 cử chỉ bằng Logistic Regression

Ứng dụng web ba cột mô phỏng sát thiết kế buổi 1: **Dữ liệu đầu vào → TF-C Model (khung giao diện) → Kết quả**. Có đăng ký/đăng nhập tài khoản, tải CSV/TXT, dự đoán thật bằng model Logistic Regression đã xuất và hiển thị xác suất 5 cử chỉ.

> **Quan trọng:** Đây là **demo giao diện + mô hình Logistic Regression do dự án tạo từ dữ liệu cảm biến mô phỏng**. Không phải TF-C đã train với 5 cử chỉ thực tế. Tiêu đề khối **“TF-C Model”** giữ theo ảnh thiết kế, nhưng mô hình đang chạy **Logistic Regression**, không phải TF-C/DINOv3. Mọi xác suất chỉ là xác suất của model demo, không phản ánh độ chính xác trên dữ liệu thực. Cần tự thu dữ liệu điện thoại và huấn luyện lại trước khi báo cáo chất lượng thực tế.

## Các chức năng

- Giao diện ba cột màu xanh, bố cục giống ảnh mẫu, responsive trên điện thoại.
- Đăng ký / đăng nhập bằng username + password, mật khẩu được băm, session cookie.
- Upload CSV/TXT 3 cột `ax, ay, az` (hoặc `x, y, z` / `acc_x, acc_y, acc_z`), 20–10.000 dòng, không quá 2 MB.
- Trích xuất các đặc trưng thống kê/tín hiệu, Logistic Regression dự đoán 5 nhãn: Vuốt phải, Vuốt trái, Vuốt lên, Vuốt xuống, Xoay cổ tay.
- Hiển thị tên dự đoán, độ tin cậy, biểu đồ xác suất cả 5 lớp, thông báo lỗi.
- Model đã export trong `models/logistic_gesture_demo.joblib`, **không cần train lại khi khởi động app**.

## Chạy ở Windows (VS Code)

Cài **Python 3.13**. Mở Terminal ở thư mục dự án mới này:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Mở: **http://127.0.0.1:5000**. Chọn **Đăng ký**, nhập username và mật khẩu ≥8 ký tự, sau đó chọn `samples/vuot_phai_demo.csv` để thử. Khi đăng nhập, nhấn **Chạy dự đoán**.

Hai file ví dụ: `samples/vuot_phai_demo.csv`, `samples/xoay_co_tay_demo.csv`. Đây là **dữ liệu mô phỏng**. Để tạo lại model: `python train_model.py`.

## Deploy miễn phí/công khai bằng Render

1. Tạo **repository GitHub mới** cho dự án `gesture_ai_web`, push toàn bộ mã nguồn trong thư mục này (bao gồm `models/logistic_gesture_demo.joblib`; không push `.env`, `instance/`). **Không cần đưa vào chung repository TF-C**.
2. Mở [dashboard.render.com](https://dashboard.render.com), đăng nhập bằng GitHub. Chọn **New → Web Service** rồi kết nối repository vừa tạo.
3. Chọn môi trường **Python**. Build command: `pip install -r requirements.txt`. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`.
4. Vào **Environment**, thêm `SECRET_KEY` bằng chuỗi ngẫu nhiên mạnh. Đặt `COOKIE_SECURE=1` để cookie chỉ gửi qua HTTPS. Có thể tạo secret bằng `python -c "import secrets; print(secrets.token_hex(32))"` trên máy.
5. Bấm **Deploy** và chờ build thành công. Render cung cấp URL dạng **https://ten-website.onrender.com** để gửi thầy. Chỉ khi service thực sự deploy thành công thì mới có link thật.

**Giới hạn triển khai:** SQLite trong `instance/users.db` trên hosting không có persistent disk sẽ **mất tài khoản khi service restart/redeploy** và không phù hợp cho nhiều máy chủ. Demo có thể đăng ký/đăng nhập, nhưng để chạy ổn định lâu dài nên chuyển sang PostgreSQL. Render free có thể tự ngủ khi không sử dụng. Chưa có rate limiting/CAPTCHA đầy đủ cho ứng dụng public: không sử dụng thông tin cá nhân hoặc mật khẩu quan trọng.

## Cấu trúc

```text
gesture_ai_web/
├── app.py                  # Flask API, login/register và inference
├── ml.py                   # Đặc trưng tín hiệu, sinh dữ liệu demo
├── train_model.py          # Train Logistic Regression từ dữ liệu giả lập
├── models/
│   └── logistic_gesture_demo.joblib
├── templates/index.html    # Giao diện ba cột
├── static/style.css        # Giao diện theo ảnh mẫu
├── static/app.js           # Upload, biểu đồ, đăng nhập
├── samples/                # CSV mẫu
├── requirements.txt
├── Procfile
└── README.md
```

## Chuẩn bị dữ liệu điện thoại thật trong tương lai

Thu đủ nhiều mẫu gia tốc `ax, ay, az` cho cả 5 cử chỉ, với nhãn tương ứng. Tách dữ liệu train/test theo người hoặc buổi ghi để tránh rò rỉ. **Huấn luyện lại Logistic Regression trên dữ liệu thực, lưu model mới bằng joblib** và đánh giá Accuracy, F1-score trên test độc lập. Không dùng độ tin cậy của bộ demo như kết quả nghiên cứu.

## Kiểm tra triển khai

Sau khi deploy, mở `https://<ten-cua-ban>.onrender.com/api/health` để thấy JSON `status: ok`, rồi truy cập trang chủ, đăng ký và tải CSV mẫu. Nếu website báo lỗi, vào Render → Logs để xem nguyên nhân. **Không có link deploy công khai thực tế cho đến khi bạn tự kết nối repo và bấm Deploy.**
