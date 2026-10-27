# Lưu tài khoản lâu dài và luồng đăng ký

Đăng ký thành công trả HTTP 201, chuyển tab về Đăng nhập, giữ tên tài khoản và xóa ô mật khẩu. Phiên đăng nhập chỉ được tạo sau khi người dùng nhập đúng thông tin ở bước Đăng nhập.

## Nơi lưu tài khoản

- Có `DATABASE_URL`: lưu bảng `users` trên PostgreSQL riêng. Web khởi động lại/redeploy không xóa bảng này. Nếu PostgreSQL lỗi, ứng dụng không tự chuyển sang SQLite.
- Chưa có `DATABASE_URL`: dùng SQLite tại `DATABASE_PATH` hoặc `instance/users.db`. Đây chỉ là chế độ phát triển/local; trên Render không có persistent disk thì vẫn có thể mất tài khoản.
- Mật khẩu được lưu dưới dạng hash có salt của Werkzeug; không lưu mật khẩu nguyên văn. Session chứa tên tài khoản, không chứa mật khẩu.

## Cấu hình trên Render (cần thực hiện trước khi kết luận đã lưu bền vững)

1. Tạo hoặc dùng một PostgreSQL database riêng. Với Render Postgres, chọn cùng region với web và lấy **Internal Database URL**.
2. Mở Web Service Gesture-AI-Web → **Environment** và đặt:
   - `DATABASE_URL`: URL của PostgreSQL. Không đưa URL này vào GitHub, ảnh công khai hoặc báo cáo.
   - `SECRET_KEY`: một chuỗi ngẫu nhiên cố định. Có thể tạo ở máy cá nhân bằng `python -c "import secrets; print(secrets.token_hex(32))"`.
   - `COOKIE_SECURE=1` khi chạy HTTPS trên Render.
3. Lưu cấu hình và deploy commit mới. Bảng `users` được tạo tự động. Chỉ deploy code mà không đặt `DATABASE_URL` thì vẫn chưa giải quyết được mất tài khoản.
4. Mở `/api/health`, kiểm tra `account_storage` là `postgresql`.
5. Đăng ký một tài khoản thử, đăng nhập, restart/redeploy web rồi đăng nhập lại. Đây là kiểm chứng cần thực hiện trên Render; kết quả test local không thay thế bước này.

Tài khoản được lưu riêng vẫn phụ thuộc vào vòng đời database. Theo tài liệu Render được kiểm tra ngày 27/10/2026, **Free Render Postgres hết hạn sau 30 ngày**, nên không phù hợp nếu cần lưu lâu dài mà không gia hạn/chuyển gói. Không có database cloud nào được tự động tạo hoặc tính phí bởi bản cập nhật code này.

## Tài khoản đã tạo trước đây

Tài khoản cũ đang nằm trong file SQLite của instance đang chạy. Đổi sang PostgreSQL không tự chuyển dữ liệu cũ. Nếu còn lấy được file `users.db`, giữ bản sao riêng và chạy:

```bash
# Đặt DATABASE_URL trong môi trường trước (không ghi bí mật vào lệnh lưu công khai).
python scripts/migrate_accounts.py /duong/dan/users.db
```

Script chuyển nguyên hash, không cần biết mật khẩu, không ghi đè tài khoản trùng có hash khác; xung đột sẽ rollback toàn bộ lần nhập. Có thể chạy lại nếu các hash trùng khớp. Nếu file SQLite cũ đã mất vì Render restart thì không thể phục hồi bằng code mới; cần đăng ký lại.

## Chạy local

Không đặt DATABASE_URL nếu chỉ thử SQLite. Cài `python -m pip install -r requirements.txt`, đặt SECRET_KEY trong môi trường, rồi `python app.py`. `.env.example` là mẫu cấu hình; app không tự đọc file `.env`.

## Nguồn

- https://render.com/docs/free
- https://render.com/docs/postgresql-creating-connecting
- https://www.psycopg.org/psycopg3/docs/basic/usage.html
