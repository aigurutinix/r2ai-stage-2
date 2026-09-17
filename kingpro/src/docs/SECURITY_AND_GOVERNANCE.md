# KINGPRO security and governance contract

Ngày cập nhật: 2026-08-28

## Authority

Mọi response có `authority`, `operation`, `requires_approval` và `approval`:

- registry/compiler: không gọi LLM, code + Pandas replay quyết định;
- generated path: LLM chỉ đề xuất Pandas, sandbox/replay/citation verifier quyết;
- refusal: deterministic policy gate quyết;
- write/export: cần role phù hợp và `X-Kingpro-Approval: user-confirmed`.

## LLM gateway

`kingpro.answering.llm_client.chat` là cổng model production:

- HTTPS/loopback transport + hostname allowlist;
- model allowlist + deployment attestation;
- PII/credential mask trước egress;
- prompt chỉ được ghi SHA-256 và độ dài, không ghi nội dung;
- append-only audit hash chain, HMAC khi có `KINGPRO_AUDIT_HMAC_KEY`;
- endpoint/model/latency/success/error có trace.

## Authentication and RBAC

`KINGPRO_AUTH_MODE`:

- `off`: chỉ dùng khi bind loopback/Demo;
- `token`: bearer token đọc từ `KINGPRO_API_TOKENS_JSON`;
- `oidc`: RS256 JWT, issuer/audience/JWKS và claim tenant/roles.

Roles:

- viewer: ask/catalog/feedback;
- analyst: batch/upload/status/download;
- approver: submission export và upload approval;
- admin: metrics/operations.

OIDC backend đã có. Browser login phụ thuộc IdP nên triển khai qua reverse proxy
OIDC hoặc frontend session adapter của nhà cung cấp; không tuyên bố có màn hình
SSO khi chưa cấu hình một IdP thật.

## Tenant isolation

- batch inputs/runs: `outputs/product_batch_jobs/{inputs,runs}/<tenant>/`;
- uploads: `outputs/uploads/{quarantine,approved}/<tenant>/`;
- feedback: `outputs/feedback/<tenant>/feedback.jsonl`;
- API lookup job/upload luôn so tenant từ principal.

Kho BTC/VIFinQA là dữ liệu công khai dùng chung. Upload private chưa tự động đi
vào retrieval; approval vẫn để `usable_by_retrieval=false` cho tới khi có bước
ingestion/index tenant-scoped riêng.

## Secure upload

API raw upload dùng `POST /uploads`, header `X-Filename` và Content-Type.

- giới hạn byte;
- allowlist CSV/TXT/JSON;
- đối chiếu MIME/extension;
- từ chối archive/ZIP, NUL/binary, invalid JSON, pathological line và CSV formula;
- ghi quarantine theo tenant;
- scanner ClamAV nếu có;
- scanner không sạch/không có thì không approve khi policy yêu cầu;
- approver + explicit approval mới chuyển sang approved;
- approval không tự ý inject vào retrieval.

## Encryption and secrets

Tenant upload hỗ trợ AES-256-GCM với key urlsafe-base64 32 byte:

- `KINGPRO_DATA_ENCRYPTION_KEY` hoặc `_FILE`;
- `KINGPRO_REQUIRE_DATA_ENCRYPTION=true` để fail-closed;
- LLM/audit secrets cũng hỗ trợ mounted secret file qua hậu tố `_FILE`.

Public BTC tables remain read-only plaintext because they are public competition
data. Confidential tenant payloads are the encryption boundary.

## Retention and deletion

`python scripts/apply_retention_policy.py` is dry-run. `--execute` moves expired
items to `.retention-trash` rather than deleting irreversibly. TTLs are separate
for batch inputs/runs, quarantine and approved uploads.

Permanent purge must be an explicit operator action after the recovery window;
the current tool intentionally does not hard-delete.

## Audit, backup and restore

- Audit JSONL is a SHA-256 chain; configure HMAC for malicious-tamper resistance.
- `scripts/backup_governance_state.py create` builds a deterministic manifest/hash ZIP.
- `verify` checks CRC and every file hash.
- `restore-drill` restores only into a new directory, blocks traversal and
  rechecks every restored hash.

## Metrics and feedback

- `/metrics` requires admin and exposes route status counts, rolling p50/p95/max,
  error rate and simple alert state.
- Feedback stores comment hash/length, not raw comment.
- Adjudication appends a new event; it cannot mutate an answer directly.
- Promotion still requires the ordinary review/release ledger.

## Source lifecycle

`scripts/audit_source_lifecycle.py` fingerprints 1,965 reports/146,246 tables.
Removed or changed report IDs set `review_required=true`; a new data drop must be
reviewed before promotion. This is the financial analogue of legal-document
effective-status monitoring.

## Code isolation

Current guaranteed layer: AST gate + child process + timeout. `isolation_health`
detects nsjail/bwrap/firejail/runsc or accepts an explicitly attested gVisor/Kata/
MicroVM container runtime.

Set `KINGPRO_REQUIRE_OS_SANDBOX=true` in production. If no provider is ready,
dynamic code generation returns `os_sandbox_unavailable`; registry/compiler paths
continue because they replay constrained, audited code. On this Windows machine
no OS sandbox provider is installed, so the product does not claim otherwise.

## Remaining provider/operator work

1. Configure real OIDC issuer/audience/JWKS and frontend login/session.
2. Install/operate malware scanner before approving uploads.
3. Put TLS and managed secret store at ingress.
4. Choose and attest gVisor/Kata/nsjail for dynamic execution.
5. Configure off-host encrypted backup destination and alert receiver.
6. Run independent penetration test before external enterprise pilot.

