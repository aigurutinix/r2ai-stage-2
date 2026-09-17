# KINGPRO Docker deployment

Docker stack gồm đúng hai service dùng chung pipeline hiện tại:

- `backend`: ProductService + SSE + durable BatchJobManager;
- `frontend`: Next.js standalone, proxy API và đọc evidence table từ cùng catalog.

Không có database giả và không copy dữ liệu 3 GB vào image. `build/`, `data/`,
`sub_v297_scope2/` được bind-mount read-only; `outputs/` được bind-mount read-write
để batch checkpoint và submission ZIP xuất hiện ngay trên máy host.

## 1. Yêu cầu

- Docker Desktop đang chạy, Linux containers;
- tối thiểu khoảng 8 GB RAM khả dụng;
- repo giữ đủ `build/`, `data/`, `sub_v297_scope2/` và `outputs/`.

## 2. Khởi động một lệnh

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 up -Rebuild
```

Mở `http://127.0.0.1:3000`.

Những lần sau không cần build lại:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 up
```

## 3. Kiểm tra

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 doctor
curl.exe http://127.0.0.1:8080/health
curl.exe http://127.0.0.1:3000/api/health
```

Backend chỉ khởi động nếu đủ mount bắt buộc, `outputs/` ghi được và SHA-256 của
`sub_v297_scope2/submission.json` đúng
`E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85`.

## 4. Model server tùy chọn

Chế độ replay/compiler/fail-closed chạy mà không cần LLM. Nếu dùng model mở trên
máy host:

```powershell
Copy-Item .env.docker.example .env.docker
```

Sửa `.env.docker`, dùng URL dạng
`http://host.docker.internal:<port>/v1`. Không commit `.env.docker`.
Dynamic mode vẫn bị khóa nếu checkpoint/revision chưa được attestation.

## 5. Batch và persistence

Mọi checkpoint, trace, manifest và ZIP sinh trong container được ghi vào
`outputs/product_batch_jobs/` trên host. Dừng/xóa container không xóa kết quả.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 logs
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 down
```

## 6. Security defaults

- chỉ publish lên `127.0.0.1`;
- container chạy non-root;
- backend filesystem read-only, chỉ `outputs/` và `/tmp` ghi được;
- drop toàn bộ Linux capabilities và bật `no-new-privileges`;
- source data/catalog/replay được mount read-only;
- secret chỉ truyền qua environment, không nằm trong image/config được commit.

