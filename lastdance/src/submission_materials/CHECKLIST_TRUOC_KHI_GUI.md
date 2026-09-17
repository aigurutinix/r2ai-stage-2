# Checklist trước khi gửi email

## Bắt buộc điền

- [ ] Tên đội.
- [ ] Họ tên, email và số điện thoại đại diện.
- [ ] Link dữ liệu Google Drive/OneDrive.
- [ ] Quyền tải dữ liệu đã được kiểm tra bằng tài khoản ngoài đội.
- [ ] Link source code hoặc ZIP source.
- [ ] Link checkpoint hoặc xác nhận sử dụng các link Hugging Face công khai.
- [ ] Link `submission.zip` và artifact liên quan.

## Kiểm tra quyền truy cập

- [ ] Mở từng link trong cửa sổ ẩn danh.
- [ ] Tải thử ít nhất một tệp từ mỗi link.
- [ ] Không yêu cầu đăng nhập tài khoản nội bộ của đội.
- [ ] Không chứa API key, Modal token, Hugging Face token hoặc credential.

## Kiểm tra artifact

- [ ] File bài nộp đúng là `submission.zip` cần gửi.
- [ ] ZIP có `submission.json` và thư mục `data/`.
- [ ] SHA-256 của file gửi được ghi lại.
- [ ] Chạy lại lệnh validation trước khi upload.
- [ ] Link source có README, requirements và configs.

## Lệnh kiểm tra cuối

```bash
shasum -a 256 outputs/baselines/benchmark_locked/submission.zip

PYTHONPATH=src python -m vifinqa.submission_validation \
  --zip outputs/baselines/benchmark_locked/submission.zip \
  --database artifacts/vifinqa.db
```

Kỳ vọng: `valid=true`, `items=1012`, `replayed=1012`, `errors=[]`.

