# Báo cáo kiểm thử Gesture-AI-Web — Buổi 9 và 9 tiếp

Ngày thực hiện: 27/10/2026. Nền mã nguồn: commit `3bba77f668a64969095c9426a2e15e67d52267ac`, cộng bản sửa đăng ký và account storage trên nhánh này.

## Phạm vi và tính trung thực của kết quả

Đã chạy kiểm thử local với model đã retrain. Không load test vào Render production, không tạo tài khoản thật của người dùng. Bộ dữ liệu tải là 50 CSV giả lập có cấu trúc 178×6, chỉ phục vụ kiểm thử phần mềm/hiệu năng; không dùng đánh giá độ chính xác phân loại.

Đã thực thi 24 ca chức năng, 3 ca adapter PostgreSQL dùng mock, 4 unit test cảm biến cũ và 2 kiểm tra JavaScript với DOM mô phỏng. Tất cả đạt. DOM mô phỏng không thay thế kiểm thử trình duyệt thật. PostgreSQL cloud và tài khoản sau Render redeploy chưa được xác nhận vì chưa có kết nối database cloud.

File `functional_results.json`, `acceptance_results.json`, `load_results.json` và `load_requests.json` chứa kết quả chạy thực tế. Không đánh dấu Pass cho các khả năng còn thiếu.

## 1. Bộ Test Case chức năng

Điều kiện chung: chạy app bằng database thử độc lập; lấy CSRF từ GET `/`; các ca upload hợp lệ đăng nhập trước. Các bước chi tiết có thể tái lập bằng `tests/test_application.py`.

| TC | Requirement | Loại | Input | Steps | Expected Result | Actual / Status |
|---|---|---|---|---|---|---|
| F01 | AUTH-01 | Hợp lệ | Tên mới, mật khẩu 11 ký tự | Mở trang → lấy CSRF → POST register | 201; next=login; chưa có session username; predict bị từ chối | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F02 | AUTH-02 | Hợp lệ | Tài khoản vừa đăng ký | POST login đúng mật khẩu | 200; session có username | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F03 | AUTH-01 | Không hợp lệ | Tên đã tồn tại | Đăng ký hai lần cùng tên | 409 ở lần thứ hai | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F04 | AUTH-03 | Biên | Tên dài 2/3/32/33 ký tự | Đăng ký từng độ dài | 2/33 → 400; 3/32 → 201 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F05 | AUTH-03 | Biên | Mật khẩu dài 7/8/128/129 ký tự | Đăng ký tên mới cho từng độ dài | 7/129 → 400; 8/128 → 201 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F06 | AUTH-03 | Không hợp lệ | Tên chứa ký tự SQL và dấu cách | POST register | 400; không tạo tài khoản | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F07 | AUTH-02 | Không hợp lệ | Mật khẩu sai | Đăng ký → login với mật khẩu sai | 401 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F08 | SEC-01 | Không hợp lệ | Thiếu X-CSRF-Token | POST register không có header CSRF | 403 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F09 | AUTH-03 | Không hợp lệ | JSON array/null; username null | POST register các body không hợp lệ | 400 thay vì 500 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F10 | AUTH-04 | Hợp lệ | Tài khoản đã đăng nhập | Logout → gọi predict | Logout 200; predict 401 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F11 | SEC-02 | Hộp trắng | Tài khoản mới | Đăng ký → đọc record trong DB thử | Hash khác mật khẩu gốc và xác minh được | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F12 | DATA-01 | Tích hợp local | DB SQLite thử nghiệm | Đăng ký → mở Python process mới đọc cùng DB | Hash tài khoản còn và xác minh được; chưa chứng minh Render persistence | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F13 | PRED-01 | Hợp lệ | CSV giả lập hợp lệ 178×6 | Đăng nhập → POST file | 200; 5 xác suất; confidence hữu hạn trong [0,100] | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F14 | INPUT-01 | Không hợp lệ | Không gửi file | Đăng nhập → POST predict trống | 400 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F15 | INPUT-01 | Không hợp lệ | Đuôi .exe | Đăng nhập → upload; theo dõi mock predict_proba | 400; model không được gọi | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F16 | INPUT-01 | Không hợp lệ | CSV rỗng | Đăng nhập → upload | 400 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F17 | INPUT-02 | Không hợp lệ | CSV chỉ có 3 cột gia tốc | Đăng nhập → upload | 400 vì thiếu gyroscope | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F18 | INPUT-03 | Biên | 19/20/10000/10001 dòng | Đăng nhập → upload từng file | 19/10001 → 400; 20/10000 → 200 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F19 | INPUT-02 | Không hợp lệ | CSV có NaN | Đăng nhập → upload | 400 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F20 | INPUT-04 | Biên | File >5 MiB | Đăng nhập → upload | 413 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F21 | INPUT-01 | Không hợp lệ | ZIP có nội dung không phải ZIP | Đăng nhập → upload | 400 | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F22 | AI-01 | Lỗi dịch vụ | MODEL=None (mock) | Đăng nhập → upload hợp lệ | 503 kèm thông báo lỗi | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F23 | DATA-02 | Lỗi dịch vụ | PostgreSQL connection error (mock) | Login khi store.find bị lỗi | 503; không lộ chuỗi kết nối | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |
| F24 | OPS-01 | Cấu hình | Chế độ SQLite local | GET /api/health | account_storage=sqlite | PASS: khớp kết quả mong đợi trong phạm vi nêu trên |

### Các ca bổ sung

- P01–P03: SQL PostgreSQL dùng bound parameters, duplicate username và không fallback sang SQLite khi PostgreSQL lỗi. **PASS với mock**; không phải kết quả chạy PostgreSQL thật.
- U01–U02: JavaScript chuyển từ đăng ký về tab đăng nhập, giữ tên và xóa mật khẩu; đăng nhập thành công mới đóng hộp thoại. **PASS trong DOM mô phỏng**.
- 4 unit test trong `test_sensor_model.py`: bỏ qua cột label khi trích xuất đặc trưng, từ chối thiếu gyroscope, đồng bộ ZIP và hình dạng vector đặc trưng. **PASS**.
- `test_postgres_live.py`: được chuẩn bị cho PostgreSQL thật, hiện **NOT RUN/SKIP tại local** do chưa có TEST_DATABASE_URL. Workflow GitHub Actions có PostgreSQL 16 thử nghiệm riêng; xem trạng thái workflow để biết kết quả CI, không suy diễn từ file báo cáo này.

## 2. Performance Test Plan

| Mục | Kế hoạch thực hiện |
|---|---|
| Endpoint | POST /api/predict qua HTTP, có session và CSRF |
| Workload | 5 virtual users; 120 giây; mỗi user chờ 1 giây sau response |
| Test data | 50 CSV giả lập khác nhau, 178 dòng × 6 kênh, khoảng 11 KB/file |
| Setup | Tạo tài khoản thử và đăng nhập trước đồng hồ đo; không tính thời gian setup/cold start |
| Environment | Local loopback; Python 3.12.14; Gunicorn 25.3.0, 2 workers × 2 threads; OPENBLAS_NUM_THREADS=1; SQLite; CPU x86_64 |
| Success criteria | p95 < 3.000 ms; HTTP error rate <1%; response schema hợp lệ 100% |
| AI metrics | Ghi proxy confidence<80%; unknown/fallback rate không có trạng thái để đo nên N/A |
| Phạm vi kết luận | Chỉ cấu hình local, workload và file size ở trên; không đại diện latency qua Internet/Render/cold start |

## 3. Kết quả đo tải

| Metric | Kết quả |
|---|---:|
| Requests | 600 |
| Thời gian đo (s) | 120.00 |
| p50 (ms) | 7.00 |
| p95 (ms) | 14.93 |
| p99 (ms) | 20.59 |
| Throughput (req/s) | 5.00 |
| HTTP error rate | 0.00% |
| Response schema success | 100.00% |
| Proxy confidence <80% | 70.00% |
| Unknown rate / Fallback rate | N/A — chưa có trạng thái sản phẩm tương ứng |

**Performance: PASS** theo tiêu chí local đã đặt. Không có phép đo CPU/RAM theo thời gian nên chưa xác định bottleneck thực tế. Rủi ro chính: Render cold start và giới hạn tài nguyên/database từ xa có thể làm latency cao hơn nhiều. Không dùng proxy confidence thấp làm tỷ lệ dự đoán sai.

## 4. AI Acceptance Test

Các tiêu chí dưới đây là tiêu chí sản phẩm đề xuất theo Buổi 9B, không khẳng định đã nằm trong yêu cầu gốc của demo. Model responses và TimeoutError ở AC01–AC03 được giả lập để kiểm tra logic; AC03 không chứng minh có time budget thực tế.

| TC | Input / Điều kiện | Expected | Actual | Status |
|---|---|---|---|---|
| AC01 | Controlled high-confidence output (mock) | HTTP 200; prediction shown with confidence 95% | HTTP 200, confidence=95.0 | PASS |
| AC02 | Controlled confidence 40% (mock); proposed threshold 80% | UNCERTAIN state and human-review warning | HTTP 200; status=None; confidence=40.0 | FAIL |
| AC03 | Injected TimeoutError; this does not test an actual time budget | FALLBACK state and usable alternative/retry guidance | HTTP 500; no FALLBACK state | FAIL |
| AC04 | Unsupported input file | HTTP 400 and model not called | HTTP 400; model_called=False | PASS |
| AC05 | Review and save prediction | User confirms/corrects prediction before persistent save | No prediction review/save feature in current UI/routes (out of original gesture-demo scope) | FAIL |
| AC06 | Traceability | Version identifier in inference response | model_name exists; immutable model_version absent | FAIL |

**Acceptance: 2/6 PASS, 4/6 FAIL** theo bộ tiêu chí đề xuất. Không kết luận toàn bộ sản phẩm đã đạt yêu cầu Buổi 9B.

## 5. Bug / khoảng trống cần xử lý

| ID | Mức độ | Vấn đề | Bằng chứng | Hướng xử lý |
|---|---|---|---|---|
| GAP-01 | Cao | Confidence thấp vẫn hiển thị như dự đoán hoàn tất, không cảnh báo UNCERTAIN | AC02, confidence 40% | Thêm trạng thái chưa chắc chắn và hướng dẫn người dùng kiểm tra; chọn ngưỡng bằng validation |
| GAP-02 | Cao | Chưa có fallback/time budget AI | AC03, HTTP 500 khi inject timeout | Thiết kế timeout và retry/fallback phù hợp cử chỉ, kiểm thử end-to-end |
| GAP-03 | Trung bình | Chưa có review/save kết quả | AC05, kiểm tra UI/routes | Xác nhận phạm vi với giảng viên; bổ sung nếu yêu cầu bắt buộc |
| GAP-04 | Trung bình | API chưa có model_version bất biến | AC06 | Thêm mã version/hash cho response và log |
| DEPLOY-01 | Cao | Chưa nối PostgreSQL cloud, chưa kiểm chứng Render redeploy | Health local vẫn sqlite | Cấu hình DATABASE_URL và chạy kiểm chứng cloud theo ACCOUNT_STORAGE.md |
| UI-01 | Trung bình | Tiêu đề giao diện vẫn TF-C nhưng model thực tế PCA + Logistic Regression | templates/index.html và training_report.json | Đổi tên giao diện khi thống nhất phạm vi; báo cáo phải ghi đúng model |

## 6. Tái lập

```bash
python -m pip install -r requirements.txt
python -m unittest test_sensor_model.py
python tests/test_application.py
node tests/test_auth_ui.js
python tests/acceptance_probe.py
# Dùng database thử riêng nếu muốn chạy test PostgreSQL thật:
# TEST_DATABASE_URL được đặt trong environment, không commit bí mật.
python tests/test_postgres_live.py
```

Đo tải ở hai terminal, trong thư mục repository (ví dụ Linux/macOS):

```bash
DATABASE_URL='' DATABASE_PATH=/tmp/gesture-load.db SECRET_KEY=local-load-only OPENBLAS_NUM_THREADS=1 python -m gunicorn app:app --bind 127.0.0.1:5055 --workers 2 --threads 2 --timeout 120
python scripts/load_test.py --url http://127.0.0.1:5055 --users 5 --duration 120
```

Dùng môi trường và tài khoản thử; script tạo 5 tài khoản test mới mỗi lần chạy. Khi chạy lại, file kết quả sẽ được cập nhật. Không dùng database chứa tài khoản thật để chạy các test destructive/reset.

## 7. Kết luận

Luồng đăng ký → đăng nhập và các ca kiểm thử chức năng trong phạm vi đã chạy đạt. Hiệu năng local đạt; AI Acceptance chưa đạt đầy đủ. Khả năng tài khoản tồn tại sau Render redeploy vẫn cần cấu hình PostgreSQL thật và kiểm chứng trên môi trường đó. Không thể phục hồi các tài khoản SQLite đã bị xóa trước đây bằng thay đổi mã nguồn này.
