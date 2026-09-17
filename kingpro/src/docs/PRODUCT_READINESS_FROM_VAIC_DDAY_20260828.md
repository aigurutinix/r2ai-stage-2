# Product-readiness audit: VAIC-DDAY → KINGPRO

Ngày: 2026-08-28  
Nguồn đối chiếu: `C:\Users\vinh\Downloads\VAIC-DDAY\vaic-2026`.

## Phán quyết

- **Competition/private runner:** sẵn sàng theo contract hiện tại.
- **Demo Day product:** đã đủ để trình diễn end-to-end có bằng chứng.
- **Public pilot có người dùng thật:** chưa nên mở Internet không kiểm soát.
- **Enterprise financial assistant:** còn thiếu identity, tenancy, data governance,
  immutable audit, operational SLO và OS-level code isolation.

Điểm mạnh của KINGPRO hiện nằm ở correctness/provenance. Nút thắt sản phẩm tiếp
theo không phải thêm model mà là **trust boundary**: ai được hỏi, dữ liệu nào được
gửi ra model, ai được tải kết quả, log giữ bao lâu và chuyện gì xảy ra khi dịch vụ
bị lạm dụng hoặc hỏng.

## Những bài học thật từ VAIC-DDAY

| Bài học | Bằng chứng trong PolicyRadar | Áp sang KINGPRO | Tác dụng |
|---|---|---|---|
| Code quyết định, LLM diễn giải | `README.md`: matcher tất định quyết eligibility | Giữ compiler/replay/verifier làm authority | Model không được tự biến một kết quả “nghe hợp lý” thành sự thật |
| PII phải che trước egress | `vn/context.py:82-129`, gọi tại `bff/main.py:1065-1075` | Mask email/phone/CCCD/tài khoản trước dynamic LLM | Dữ liệu người dùng không rời trust boundary dưới dạng nguyên văn |
| Mọi LLM qua một gateway | `gateway/client.py:48-176` | KINGPRO đã có endpoint allowlist; cần thêm audit tập trung | Chặn SSRF/endpoint lạ, đo cost/latency/failure và đổi model có kiểm soát |
| Read và write là hai cấp rủi ro | `ho_so/sinh.py`, `bff/main.py:1318-1340` | Nếu sau này gửi báo cáo/email/ERP phải `requires_approval` | AI không tự thực hiện hành động tài chính có hậu quả |
| Validate dữ liệu nghiệp vụ | `matcher/kiem_ho_so.py` | Gate số âm, tỷ lệ >100%, đơn vị phi lý, kỳ/scope xung đột | Bắt lỗi nhập liệu trước khi retrieval/model hợp thức hóa nó |
| Nguồn có vòng đời | `scripts/cron_giam_sat.py`, cache hiệu lực | Theo dõi filing mới, restatement, superseded report và as-of date | Không trả số đúng từ một báo cáo đã bị thay thế |
| Data rights là product feature | `docs/NGUON-DU-LIEU.md` | Giữ data card, revision và attribution ngay trong UI/export | Giảm rủi ro pháp lý và làm nguồn dữ liệu có thể nghiệm thu |
| Công bố giới hạn/OOD | README công bố guard in-domain tốt nhưng OOD yếu | Health/model card phải tách measured/unmeasured | Không biến benchmark nội bộ thành lời hứa production |
| Action workflow quan trọng hơn chat | Policy → citation → form → draft → approval | Question → evidence → calculation → export/report → approval | Sản phẩm giải quyết công việc, không chỉ tạo một bong bóng trả lời |
| Pilot phải có KPI nghiệp vụ | `docs/LO-TRINH-PILOT.md` | Đo time-to-answer, verify rate, correction rate, analyst acceptance | Biết sản phẩm có tiết kiệm công sức thật hay chỉ đẹp trên sân khấu |

## KINGPRO đã có các lớp bảo mật/tin cậy nào

- CORS chỉ cho origin cấu hình.
- Giới hạn request body chat và batch.
- Prompt-injection/domain policy chạy trước model.
- LLM endpoint/model allowlist và attestation gate.
- Secret redaction trong manifest/logbook; source bundle có secret scan.
- Pandas AST gate, child process và timeout.
- Table path traversal bị chặn.
- Citation dependency, replay verification và cell-level lineage.
- Docker bind localhost, container non-root, filesystem read-only,
  `no-new-privileges`, drop capabilities.
- V297 SHA gate trước khi backend nhận traffic.
- Per-item batch isolation, atomic checkpoint, resume và deterministic archive.
- Trace ID cho mỗi câu; batch có manifest/report/trace.
- Fail-closed khi thiếu entity/year/source hoặc verification thất bại.

Những lớp này đủ tốt cho máy Demo và private runner, nơi người vận hành tin cậy
và dịch vụ chỉ bind loopback.

## Khoảng trống quan trọng

### P0 — nên hoàn thiện trước khi mở một URL public

| Thiếu | Tác dụng cần có | Cách làm đúng mức |
|---|---|---|
| Rate limit + global concurrency cap | Chống spam làm nổ CPU/RAM/SSE/thread | Token bucket theo IP/API key; cap ask streams, batch jobs và số item/job |
| Authentication tối thiểu | Không cho người lạ gọi batch/download | Demo token hoặc reverse-proxy basic/OIDC; download phải kiểm cùng principal |
| API audit có privacy | Điều tra ai hỏi gì và hệ thống trả mode nào | Append-only event: request ID, actor, route, status, latency, mode; chỉ hash/redact question |
| PII/secret egress guard | Chặn người dùng vô tình gửi dữ liệu nhạy cảm sang model ngoài | Mask email/phone/CCCD/account/API-key pattern trước `make_llm_from_env` |
| Security headers | Giảm XSS/clickjacking/content sniffing | CSP, frame-ancestors, nosniff, referrer policy, permissions policy ở Next/reverse proxy |
| Dependency/image scan | Phát hiện thư viện/image có CVE | `pip-audit`, `npm audit`, Trivy; xuất SBOM Syft trong release CI |
| Explicit retention policy | Tránh log/output tích tụ dữ liệu vô hạn | TTL cho batch/trace, endpoint purge và hướng dẫn operator |

### P1 — bắt buộc cho pilot doanh nghiệp

| Thiếu | Tác dụng |
|---|---|
| OIDC/SSO + RBAC | Phân quyền viewer, analyst, approver, admin |
| Tenant isolation | Công ty A không nhìn thấy câu hỏi, file và output công ty B |
| Secure upload/ingestion | MIME sniffing, size/page limit, malware/quarantine, zip-bomb và CSV-formula guard |
| Encryption + managed secrets | TLS ingress, encrypted storage/backups và key rotation |
| Data classification/consent | Phân biệt BCTC công khai, nội bộ, mật; quyết định model nào được phép xử lý |
| Tamper-evident audit | Hash-chain/sign audit log; actor/action/artifact/model revision truy được |
| Backup + restore drill | Không chỉ “có backup”; phải chứng minh restore chạy được |
| Metrics/SLO/alert | p50/p95 latency, refusal/error rate, queue depth, model cost, availability |
| Feedback/adjudication | Người dùng báo sai nguồn; sửa qua review ledger, không hard-code lén |
| Source freshness | Phát hiện báo cáo điều chỉnh, thay thế và dữ liệu vừa được công bố |
| Strong execution isolation | Chạy generated code trong gVisor/nsjail/MicroVM, network off, CPU/RAM/pid quota |

### P2 — khi bán enterprise

- SCIM/user lifecycle, MFA và policy theo nhóm.
- KMS/HSM, customer-managed key và data residency.
- Row/document-level access control tại retrieval.
- Approval workflow cho export, gửi email, tạo báo cáo hoặc tích hợp ERP.
- Signed images, provenance/SLSA, image admission policy.
- Incident-response runbook, breach notification và tabletop exercise.
- Penetration test độc lập; threat model theo STRIDE/OWASP LLM Top 10.
- HA, multi-zone, RPO/RTO và disaster-recovery drill.
- Model/prompt registry: version, evaluator, approver, canary và rollback.

## Những thứ KHÔNG cần làm trước Demo

- PostgreSQL chỉ để nói mình có database.
- Kubernetes/microservices/service mesh.
- Blockchain cho citation.
- MCP server không có consumer/tool thật.
- Multi-agent council cho mọi câu.
- SSO/SCIM hoàn chỉnh khi Demo chỉ chạy local.
- Neural hallucination detector chưa chứng minh OOD.

Các món này tăng bề mặt lỗi và làm loãng câu chuyện bằng chứng của KINGPRO.

## Thứ tự hợp lý nhất

### Gói Demo hardening nhỏ

1. Rate/concurrency caps.
2. API audit redacted.
3. PII/secret egress guard.
4. Security headers.
5. Dependency scan + SBOM.
6. Retention/purge contract.

### Gói pilot sau cuộc thi

1. Auth/RBAC/tenant.
2. Secure report upload + ingestion quarantine.
3. Encrypted persistent store và managed secret.
4. Source freshness/restatement monitor.
5. Metrics/SLO/alerts + backup/restore.
6. OS-level sandbox và independent security review.

## Ba ca nên demo để chứng minh đây là sản phẩm

1. **Câu đúng:** trả số, Pandas, bảng và ô nguồn.
2. **Câu nguy hiểm/thiếu dữ liệu:** fail-closed và nêu field còn thiếu.
3. **Câu chứa prompt injection hoặc secret/PII:** chặn trước model, trace ghi sự
   kiện bảo mật nhưng không ghi dữ liệu nhạy cảm nguyên văn.

## Câu định vị sau khi bổ sung P0

> KINGPRO là trợ lý phân tích tài chính evidence-first: code quyết định, model chỉ
> hỗ trợ ngôn ngữ; dữ liệu nhạy cảm bị chặn trước egress, mọi kết quả có lineage
> và replay, còn mọi hành động hoặc trường hợp thiếu căn cứ đều phải qua policy
> và human approval.

## Lưu ý trung thực về VAIC-DDAY

Không nên sao chép VAIC như một mẫu security hoàn chỉnh. PolicyRadar có PII mask,
egress allowlist, LLM audit và write-gate rất đáng học, nhưng runtime public được
commit không có authentication/rate limiting/tenant isolation; frontend lưu hồ
sơ và lịch sử trong `localStorage`; audit LLM mới là file JSONL và chính tài liệu
deploy thừa nhận hobby hosting chưa có SLA/backup. Bài học đúng là lấy các
**nguyên tắc**, rồi hoàn thiện trust boundary còn thiếu trong KINGPRO.

