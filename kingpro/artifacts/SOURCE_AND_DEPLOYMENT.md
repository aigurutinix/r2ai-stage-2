# Mã nguồn, dependencies và triển khai — KINGPRO V297

## Cấu trúc source bundle

```text
src/kingpro/          core extraction, retrieval, answering, sandbox, product
scripts/              build, audit, batch, release và reproduction commands
frontend/app/         Next.js application và API proxy
frontend/components/  UI primitives cần để build frontend
frontend/lib/         TypeScript contracts/helpers
tests/                unit, integration, mutation, metamorphic tests
config/               provenance allowlists và release policy
docker/               backend/frontend production images và SHA preflight
compose.yaml          health-gated two-service deployment
docs/                 data/model/compliance/reproducibility cards
```

`KINGPRO_V297_SOURCE.zip` loại bỏ `.env`, credentials, node_modules, `.next`,
virtual environments, caches, build tables, raw dataset và historical candidate
payloads. Manifest trong ZIP khóa SHA-256 từng file.

## Môi trường

- Python 3.11 khuyến nghị; scorer compatibility đã kiểm thêm Python 3.9.
- Node.js 20+ cho frontend.
- Windows 10/11 hoặc Linux x86_64.
- GPU chỉ cần nếu bật dynamic Qwen; registry/compiler path chạy CPU.

Python core:

```text
bm25s >=0.2,<0.3
lxml >=4.9,<6
numpy >=1.23,<3
pandas >=1.5,<3
scipy >=1.9,<2
```

Frontend khóa trong `frontend/package-lock.json`: Next.js 16.3.2, React
19.2.8, TypeScript 5.9.x và Lucide React.

## Cài đặt từ đầu

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd frontend
npm ci
npm run lint
npm run build
cd ..
```

## Chạy backend/frontend

Terminal 1:

```bash
set PYTHONPATH=src
python scripts/serve_product.py --host 127.0.0.1 --port 8080
```

Terminal 2:

```bash
cd frontend
npm start -- --hostname 127.0.0.1 --port 3000
```

Mở `http://127.0.0.1:3000`.

## Chạy production bằng Docker

Docker Desktop phải dùng Linux containers. Từ thư mục gốc:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 up -Rebuild
```

Stack gồm backend và Next.js standalone. Frontend chỉ khởi động sau khi backend
healthy. Catalog, BM25, bảng CSV và v297 replay được bind-mount read-only;
`outputs/` được mount read-write để giữ checkpoint, trace và submission ZIP sau
khi container bị thay thế. Backend chạy non-root, filesystem read-only, drop
Linux capabilities và fail-fast nếu SHA replay khác canonical V297.

Kiểm tra hoặc dừng:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 doctor
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 down
```

Xem chi tiết tại `docs/DOCKER_DEPLOYMENT.md`. Dynamic LLM là tùy chọn; nếu model
server chạy trên host, dùng `host.docker.internal` trong `.env.docker` và không
commit secret.

## Security và governance

Product API hỗ trợ token/OIDC RBAC và tenant boundary cho batch, upload, feedback.
Mọi LLM call đi qua endpoint/model allowlist, PII/credential egress mask và audit
hash-chain; prompt chỉ lưu hash/độ dài. Hành động export/approve cần role phù hợp
và xác nhận người dùng. Upload bị quarantine, kiểm MIME/payload/CSV formula và
không được dùng cho retrieval trước approval; production có thể bắt buộc malware
scanner và AES-256-GCM.

Chi tiết biến môi trường, retention, backup/restore, metrics, source lifecycle và
OS sandbox fail-closed nằm tại `docs/SECURITY_AND_GOVERNANCE.md`. Source bundle
không tuyên bố có gVisor/SSO provider/malware scanner khi operator chưa cấu hình
và attest các dịch vụ ngoài đó.

## Kiểm tra private-ready

```bash
python scripts/private_ready_doctor.py --require-live
```

Expected: `status=PASS`. Full validation đã đo trên canonical V297:

- 558 tests + 25 subtests cho product hiện hành;
- 106/106 typed dual execution;
- 29/29 adversarial rejection;
- 3/3 metamorphic invariance;
- 1.012/1.012 cell lineage, 7.299 cells;
- browser audit và console sạch.
