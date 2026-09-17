# Runtime attestation — open-model serving

Tài liệu này tách ba tuyên bố thường bị nhập nhằng khi demo:

1. **Public model evidence:** model card chính thức công bố checkpoint, license,
   ngày phát hành và số tham số.
2. **Control-plane attestation:** cấu hình endpoint và template của chính tài
   khoản RunPod cùng trỏ tới checkpoint đó.
3. **Runtime identity attestation:** serving route đang hoạt động và trả đúng
   model ID tại thời điểm kiểm tra.

Chỉ lớp 3 cộng với ảnh deployment/revision do operator giữ mới cho phép bật
free-form generation. Nếu thiếu, hệ thống tiếp tục fail-closed; verified replay
và grounded compiler vẫn hoạt động.

## Model được cấu hình

| Trường | Bằng chứng |
| --- | --- |
| Checkpoint | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Ngày công bố | 12/11/2024, trước cutoff 01/06/2026 |
| Tổng tham số | 14,7B; non-embedding 13,1B |
| License | Apache-2.0 |
| Ngưỡng áp dụng | ≤15B theo xác nhận bằng văn bản của BTC do đội giữ |

Nguồn chính thức:

- [Hugging Face model card](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
- [Qwen release announcement](https://qwenlm.github.io/blog/qwen2.5-coder-family/)
- [RunPod OpenAI compatibility](https://docs.runpod.io/serverless/vllm/openai-compatibility)
- [RunPod endpoint health](https://docs.runpod.io/serverless/endpoints/operation-reference)

## Snapshot control-plane ngày 24/08/2026

`scripts/audit_runtime_attestation.py` đọc metadata qua API quản trị RunPod và
chỉ xuất allowlisted fields. Endpoint ID được thay bằng fingerprint; API key,
user ID, token và nội dung sinh không bao giờ được ghi ra báo cáo.

- endpoint environment: `MODEL_NAME=Qwen/Qwen2.5-Coder-14B-Instruct`;
- template environment: cùng model ID, `DTYPE=auto`, `QUANTIZATION=None`;
- image: `registry.runpod.net/runpod-workers-worker-vllm-main-dockerfile:9e1c48313`;
- GPU allowlist: NVIDIA A40 hoặc RTX A6000, một GPU/worker;
- configuration endpoint/template/model/image: PASS;
- `/health` báo 3 ready, 0 unhealthy nhưng pod inventory độc lập chỉ có 5
  worker `EXITED`, không có worker `RUNNING`;
- native `/run` ở `IN_QUEUE` suốt 30 giây rồi audit hủy đúng job probe;
- `/models` timeout 120 giây và làm queue tăng;
- vì health và inventory mâu thuẫn, `control_plane_attested=false`,
  `runtime_identity_attested=false` và `dynamic_enable_recommended=false`.

Đây là bằng chứng cấu hình mạnh hơn kiểm tra hostname phía client, nhưng không
phải bằng chứng worker đang phục vụ hay hash mật mã của weight bytes đã nạp.
Vì vậy `KINGPRO_LLM_ATTESTED` vẫn phải giữ `false` cho tới khi có ít nhất một
worker `RUNNING`, native probe hoàn tất và serving route trả đúng model ID.

## Chạy lại trước khi demo

```powershell
python scripts\audit_runtime_attestation.py `
  --probe-native `
  --probe-openai `
  --timeout 60 `
  --output build\demo_compliance\runtime_attestation_live.json
```

Chỉ khi báo cáo đồng thời có:

- `control_plane_attested=true`;
- `runtime_identity_attested=true`;
- `dynamic_enable_recommended=true`;
- `secrets_exposed=false`;

thì operator mới lưu ảnh deployment/revision, đặt
`KINGPRO_LLM_ATTESTED=true`, restart backend và chạy lại HTTP smoke. Không có
script nào tự bật cờ này hoặc tự sửa `.env`.
