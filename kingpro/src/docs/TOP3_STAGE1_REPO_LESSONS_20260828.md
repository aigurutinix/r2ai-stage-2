# Đọc sâu Top 1–2–3 R2AI Stage 1 → bài học cho KINGPRO Stage 2

Ngày rà soát: 2026-08-28  
Nguồn: bản open-source chính thức đã checkout tại
`C:\Users\vinh\Downloads\r2ai-stage-1`.

## Kết luận điều hành

Ba đội mạnh theo ba hướng khác nhau:

| Đội | Thứ hạng | Năng lực nổi bật nhất | Một câu kết luận |
|---|---:|---|---|
| mscAI | 1 | Product/operations end-to-end | Pipeline tốt được biến thành dịch vụ có lifecycle, streaming, batch, persistence và deployment rõ ràng. |
| Hung&Fong | 2 | Data quality + forensic experimentation | Không đoán lỗi; họ dùng lịch sử submission và root-cause audit để biết đang nghẽn ở corpus, retriever hay selector. |
| Nguyễn Văn Nghiêm | 3 | Multi-stage retrieval + learned ranking | Thu hẹp noise theo tầng rồi mới dùng reranker/LLM; hard negatives được lấy từ chính các ứng viên “gần đúng nhưng sai”. |

Nguồn xếp hạng chính thức: `r2ai-stage-1/README.md:58-66`.

Không có một repo nào là khuôn mẫu hoàn chỉnh cho Stage 2. KINGPRO cần ghép:

```text
mscAI: lifecycle + serving + batch + UI
    + Hung&Fong: data normalization + root-cause audit + safe selector
    + Nguyễn Văn Nghiêm: candidate cascade + hard-negative reranker
    + KINGPRO: typed financial plan + sandbox + deterministic execution
               + cell-level causal lineage + fail-closed
```

## 1. mscAI — vì sao đây là sản phẩm end-to-end hoàn chỉnh nhất

### Những thứ repo thực sự làm

1. **Nguồn dữ liệu chuẩn có trạng thái**
   - PostgreSQL giữ legal records chuẩn.
   - Chroma giữ dense index persistent.
   - BM25 có cache corpus đã tokenize.
   - Backend đọc dữ liệu và dựng/khôi phục index ngay trong lifecycle startup.
   - Bằng chứng: `mscai/src/backend/docs/REPORT.md:6-11`,
     `mscai/src/backend/src/services/vector_store/index_builder.py:31-89`.

2. **Index có manifest và tự phát hiện stale cache**
   - Manifest chứa nguồn PostgreSQL, số record và cấu hình embedding.
   - Collection count được đối chiếu trước khi reuse.
   - BM25 có manifest riêng.
   - File lock ngăn nhiều Uvicorn worker embedding lại cùng lúc.
   - Bằng chứng: `index_builder.py:20-22,31-39,51-84,178-227,322-353`.

3. **Một core pipeline cho chat, batch và CLI**
   - LangGraph đi qua intent → prepare query → retrieve → rerank → LLM filter →
     answer → format.
   - Competition mode bỏ intent và gọi thẳng legal RAG.
   - Không dựng một logic khác cho giao diện và một logic khác cho bài nộp.

4. **Streaming là telemetry thật, không phải animation giả**
   - SSE phát stage status, token và result.
   - Heartbeat mỗi một giây giữ kết nối và cho biết stage đang chạy.
   - Callback tiến độ được cô lập theo request.
   - Bằng chứng: `mscai/src/backend/src/routers/legal.py:90-220`.

5. **Batch là first-class workload**
   - Semaphore giới hạn concurrency.
   - Mỗi câu lỗi được cô lập.
   - Output partial được ghi lại sau từng câu; có running/final/report.
   - Có SSE riêng cho competition và đường tải kết quả cuối.
   - Bằng chứng: `legal.py:245-496`,
     `services/competition/output.py:81-93`.

6. **Deployment và readiness có contract**
   - Docker Compose gồm PostgreSQL, backend và UI.
   - Có healthcheck và volume cho PostgreSQL/Chroma.
   - UI phụ thuộc backend healthy; backend phụ thuộc PostgreSQL healthy.
   - Bằng chứng: `mscai/src/docker-compose.yml:4-60`.

7. **UI giúp kiểm chứng pipeline**
   - Chat history, tìm/xóa hội thoại, bật/tắt trace.
   - Trace hiển thị stage và retrieval candidates.
   - Có upload competition file và tiến độ từng item.
   - Bằng chứng: `mscai/src/ui/app/page.tsx:133-177,234-373,387-527,595-640`.

### Điều quan trọng: đừng nhầm “có code” với “đã bật trong bản cuối”

Cấu hình được commit của mscAI là:

- `vector_store.mode: chroma`, không phải hybrid;
- HyDE bật;
- rewrite tắt;
- reranker tắt;
- LLM filter bật;
- `top_k: 7`, competition concurrency 12.

Bằng chứng: `mscai/src/backend/config.yaml:28-73`.

Vì vậy không được nói “mscAI thắng nhờ BM25 + Chroma + reranker” chỉ vì repo có
các module đó. Giá trị chắc chắn nhất từ mscAI là **kiến trúc sản phẩm và
operational discipline**, không phải bằng chứng rằng mọi toggle đều tăng điểm.

### Điểm yếu/không nên sao chép nguyên trạng

- API key và mật khẩu PostgreSQL nằm trong YAML mẫu.
- `network_mode: host` và file lock `fcntl` thiên về Linux.
- Short memory dùng `InMemorySaver`, mất khi process dừng.
- MCP trong repo không đăng ký tool có ích và backend không dùng nó.
- LLM filter fail-open giúp giữ recall nhưng vẫn có thể giữ noise.
- Không có sandbox thực thi code vì Stage 1 không phải Text-to-Pandas.
- Một số option cấu hình được phép bỏ qua key lạ, có nguy cơ typo im lặng.

## 2. Hung&Fong — repo đáng học nhất để tăng accuracy có kiểm soát

### Bài học đã được điểm leaderboard xác nhận

1. **Data overhaul tạo bước nhảy sạch lớn nhất**
   - v13 `0.4657` → v14 `0.5170`.
   - Họ chunk theo cấu trúc Điều/Khoản/Điểm, bỏ overlap, normalize tiếng Việt đối
     xứng ở corpus và query, loại record rác, dedup và re-embed 285K chunks.
   - Cả precision lẫn recall cùng tăng.
   - Bằng chứng: `hung&phong/src/docs/SUBMISSIONS.md:23`,
     `POST_SUBMISSION_REVIEW.md:63-83`.

2. **Domain embedding thắng generic embedding**
   - A/B local R@1: Vietnamese embedding `0.80`, bge-m3 `0.52`.
   - Bằng chứng: `SUBMISSIONS.md:17,60`.

3. **F2 không cho phép filter hung hăng**
   - Full LLM judge tăng precision nhưng làm mất recall.
   - Chính sách `top-2 OR judge-yes` tăng đến `0.5716`.
   - Keep-top-3 lại kém hơn, nên đây là kết quả đo chứ không phải quy luật thần kỳ.
   - Bằng chứng: `POST_SUBMISSION_REVIEW.md:122-145`.

4. **Phân rã sâu có thể phá retrieval**
   - Deep decomposition làm query drift, thay đổi intent và lấy thêm ứng viên
     nhiễu; bản v22 giảm mạnh.
   - Giữ original query làm anchor và chỉ dùng decomposition nông khi thật cần.
   - Bằng chứng: `POST_SUBMISSION_REVIEW.md:181-218`.

5. **Chẩn đoán theo vị trí gold trong candidate pool**
   - Gold đã ở pool nhưng không được chọn: lỗi selector/ranking.
   - Gold không ở pool: lỗi data/retrieval.
   - Audit v72 chỉ ra nhiều candidate tốt đã xuất hiện nhưng selector bỏ mất;
     lúc đó bottleneck chính là decision logic.
   - Bằng chứng: `V72_ROOT_CAUSE_AUDIT_2026_06_30.md:67-108`.

6. **Baseline fallback là uncertainty policy hợp lệ**
   - Selector mới chỉ được phép thay baseline khi tín hiệu đủ mạnh.
   - LLM/prompt nên là audit signal, không phải final mutator không kiểm soát.
   - Bằng chứng: `V67_ROOT_CAUSE_AUDIT.md:5-41`,
     `V66_ARCH_PIPELINE_NOTES.md:5-12`.

### Product của Hung&Fong

- FastAPI + SSE.
- Ollama local cho LLM/embedding.
- Qdrant embedded hoặc server.
- BM25 + dense + reranker.
- Có OpenAI-compatible path để ghép Open WebUI.
- Cài local dễ hơn mscAI, nhưng không có frontend riêng/persistence/lifecycle hoàn
  chỉnh bằng mscAI.

Đây không chỉ là mô tả README: mã runtime thực sự normalize query đối xứng,
phân tích metadata để filter-first rồi fallback vector-only, hợp nhất dense và
BM25, ưu tiên reranker nếu bật và dedup candidate. Bằng chứng:
`hung&phong/src/backend/rag.py:69-154`,
`hung&phong/src/backend/query_analyzer.py:1-91`,
`hung&phong/src/backend/main.py:1-82`.

### Những điều không nên bê

- Không dùng “bản luật mới nhất” như boost cứng; gold có thể là văn bản cũ nhưng
  vẫn đúng/ngữ cảnh yêu cầu.
- Không thay nhiều biến cùng một A/B; repo tự ghi nhận có các thử nghiệm bị
  confound.
- Không cho listwise LLM xóa candidates tốt khi retrieval yếu.
- Không dùng các blacklist/manual scope rule như nền tảng cho private.

## 3. Nguyễn Văn Nghiêm — cascade mạnh, nhưng có rủi ro overfit cần nhìn thẳng

### Pipeline công khai

```text
original query + decomposition + HyDE
→ Dense FAISS top-100
BM25S top-150 trên 1.88M chunks
→ intersection khoảng 29 candidates/query
→ fine-tuned Vietnamese reranker top-5
→ Qwen3-8B-AWQ chọn article
→ LoRA classifier filter/add
→ answer generation có grounding prompt
```

Articles F2 được báo cáo cho nhánh LLM top-5 là `0.6056`; tài liệu mô tả
classifier post-process tăng tới khoảng `0.632`.

### Những ý tưởng đáng học

1. **Giảm noise trước khi dùng model đắt**
   - Reranker hoạt động kém trên pool HyDE khoảng 97 candidates nhưng tốt hơn trên
     intersection khoảng 29 candidates.
   - Đây là cascade có chủ đích, không phải “thêm agent”.
   - Bằng chứng: `nguyenvannghiem/README.md:34-57,85-94`.

2. **Synthetic QA là cầu nối ngôn ngữ cho BM25**
   - 1.27M synthetic QA giúp BM25 ceiling từ khoảng 52% lên 93.6%.
   - Cùng dữ liệu đó đưa vào dense lại giảm precision khoảng 10.8%.
   - Bài học: augmentation phải phục vụ đúng retriever; không đổ mọi dữ liệu vào
     mọi index.
   - Bằng chứng: `src/docs/data_description.md:154-193`.

3. **Hard negatives lấy từ vùng gần đúng**
   - Positive là article được cite.
   - Negative lấy ở dense rank 5–12 sau dedup; khoảng 7 negatives/positive.
   - 500 query được hold out cho evaluator reranker.
   - Bằng chứng: `src/docs/model_description.md:74-89`,
     `src/code/train_reranker_v2.py:112-208`.

4. **Tối ưu metric cuối, không tối ưu recall đơn lẻ**
   - ck-18000 recall top-5 `87.1%` nhưng F2 `0.5187`.
   - ck-8000 recall `79.3%` nhưng F2 `0.5347`, nên được chọn.
   - Bằng chứng: `src/docs/model_description.md:91-97`.

5. **Fallback khi model selector lỗi**
   - Nếu LLM không parse được output hoặc request lỗi, code giữ top-3 reranker.
   - Đây là recall guard thực tế.
   - Bằng chứng: `run_llm_rerank_on_reranker_top5.py:90-120`.

### Red flags phải tránh ở private

1. **Classifier được mô tả là overfit có chủ đích trên chính distribution test**
   - Tài liệu nói train trên top-5 candidates của 2,000 câu test, label từ union
     submission 27B và ck-23000 đạt train F1 1.0.
   - Cách này không có giá trị bảo đảm cho private unseen.
   - Bằng chứng: `src/docs/model_description.md:144-177`.

2. **Tài liệu và code classifier không khớp hoàn toàn**
   - Docstring nói label từ union 27B/top-5.
   - Nhưng code được commit đặt `LABEL_FILE = data_final_hard_negatives.jsonl` và
     đọc trực tiếp file đó.
   - Bằng chứng: `train_legal_classifier_v2.py:1-5,40,138-148`.

3. **Eval classifier bị leakage**
   - `train_rows = all_rows` rồi `eval_rows = random.sample(all_rows, 500)`.
   - Eval là tập con của train, nên F1 eval không đo generalization.
   - Bằng chứng: `train_legal_classifier_v2.py:154-162`.

4. **Threshold ensemble được sweep với pseudo-GT của chính LLM output**
   - `load_ground_truth()` đọc submission LLM 8B làm pseudo-GT.
   - Kết quả sweep đo mức giống pseudo-GT, không phải ground truth độc lập.
   - Bằng chứng: `ensemble_classifier_reranker.py:67-81,139-220`.

5. **Intersection cứng có thể giết recall trong Stage 2**
   - Chỉ dùng khi audit chứng minh cả hai retriever có recall ceiling đủ cao.
   - Với bảng tài chính, sai OCR/header có thể khiến BM25 hoặc dense đơn lẻ bỏ sót;
     intersection cứng sẽ xóa luôn candidate đúng.

## 4. Ma trận áp dụng vào KINGPRO

### Đã có hoặc đã áp dụng tốt hơn

| Bài học | Trạng thái KINGPRO |
|---|---|
| Chat và Batch dùng chung core pipeline | DONE |
| SSE stage + heartbeat | DONE |
| Ordered concurrency, per-item isolation | DONE |
| Atomic checkpoint, resume, run manifest | DONE |
| Evidence/source UI và trace | DONE |
| Deterministic release ZIP + replay gate | DONE |
| Sandbox thực thi Pandas | DONE — đây là lớp Top 3 Stage 1 không cần có |
| Fail-closed khi thiếu bằng chứng | DONE |
| Cell-level lineage và causal perturbation | DONE cho 1,012 registry; generated unseen còn PARTIAL |
| Root-cause audit theo câu/bucket | DONE, cần tiếp tục dùng cho hidden-like validation |

### Nên làm tiếp — lợi ích cao, không đụng bản public v297 đã freeze

1. **Index lifecycle contract kiểu mscAI**
   - Manifest phải fingerprint catalog, table CSV, tokenizer, alias map,
     embedding/reranker revision và record count.
   - Lock cross-process khi build.
   - Backend chỉ READY khi count/hash hợp lệ.

2. **Financial hard-negative suite kiểu Hung&Fong + Nghiêm**
   - Đúng metric nhưng sai company.
   - Đúng company nhưng sai year.
   - Hợp nhất vs riêng lẻ.
   - Bảng chính vs thuyết minh.
   - Cùng row label nhưng sai cột kỳ hạn/tổng cộng.
   - Đúng số nhưng sai đơn vị triệu/tỷ/%.
   - Train/eval phải split theo company + year + question type; tuyệt đối không lấy
     eval làm tập con train.

3. **Agreement-aware candidate lattice, không intersection cứng**
   - Metadata exact filter tạo pool nền.
   - BM25/alias, dense và structural retrieval chạy độc lập.
   - Candidate có nhiều đường đồng thuận được boost.
   - Candidate chỉ một đường tìm thấy vẫn được giữ nếu metadata/cell structure mạnh.
   - Chỉ cắt khi recall ceiling trên hidden-like split đã chứng minh an toàn.

4. **Synthetic financial questions chỉ cho lexical bridge**
   - Sinh từ metric alias/row thật, không sinh số đáp án và không gắn public ID.
   - Dùng để mở rộng BM25 label matching.
   - Không đưa tự động vào dense index trước khi A/B chứng minh không giảm precision.

5. **Selector có baseline guard**
   - Model/reranker mới chỉ được thay deterministic baseline khi company/year/
     scope/unit và score margin đều đạt gate.
   - Khi bất đồng, giữ candidate nền hoặc chạy verifier; không để LLM tự xóa.

6. **One-command cold start cho Demo — ĐÃ ÁP DỤNG**
   - `compose.yaml` cùng hai multi-stage Dockerfile dựng backend và frontend,
     frontend chỉ chạy sau khi backend healthy.
   - `scripts/docker_product.ps1 up -Rebuild` là entrypoint một lệnh trên máy Demo.
   - Secrets tách khỏi image/config; evidence read-only, outputs persistent và
     backend khóa SHA v297 trước khi nhận traffic.

### Chỉ thử sau hidden-like validation

- Dense + BM25 weighted fusion.
- HyDE/query rewrite.
- LLM filter/critic.
- Learned reranker.
- Synthetic QA quy mô lớn.

Mọi thử nghiệm trên phải có:

```text
baseline hash
hidden-like split độc lập
candidate recall@k
row/cell recall
execution accuracy
end-to-end accuracy
latency
failure buckets
rollback artifact
```

### Không áp dụng

- Classifier memorize public/test candidates.
- Eval lấy mẫu từ chính train.
- Threshold tuning theo pseudo-GT của submission cần đánh giá.
- LLM làm final mutator mà không có baseline guard.
- Intersection cứng chưa chứng minh recall.
- Boost “mới nhất” hoặc scope bằng rule tuyệt đối.
- MCP shell không có tool thật.
- Thay v297 chỉ vì một module mới nghe có vẻ hay.

## 5. Kiến trúc đích rút ra từ cả ba repo

```text
Question
  → Typed financial plan
  → exact company/year/scope/unit resolver
  → candidate lattice
       metadata + BM25/alias + dense + structural header
  → source-aware table/row/cell reranker
  → deterministic formula compiler (ưu tiên)
       hoặc constrained Pandas generator (fallback)
  → sandbox execution
  → independent typed replay + unit/scope verifier
  → causal cell lineage
  → grounded answer / fail-closed refusal
  → SSE trace + durable batch checkpoint
  → strict replay validator + deterministic submission ZIP
```

Điểm khác biệt của KINGPRO không nên là “nhiều agent hơn”. Điểm khác biệt nên là:

> Mỗi kết quả số chỉ được xuất khi identity của bảng, tọa độ ô, đơn vị, phép tính
> và kết quả replay tạo thành một chuỗi bằng chứng khép kín; mọi đường retrieval
> bất đồng đều được lưu thành tín hiệu rủi ro thay vì bị che bởi câu trả lời tự tin.

## 6. Thứ tự triển khai thực tế

### Trước private test

1. Giữ nguyên v297 public freeze và SHA.
2. Chốt hidden-like split theo company/year/question type.
3. Thêm index manifest/readiness/file lock cho dynamic path.
4. Dựng hard-negative audit set; chưa fine-tune nếu chưa có split sạch.
5. Chạy private doctor + strict exporter + release gate cho mỗi data drop mới.

### Trước Demo Day

1. Chạy `docker_product.ps1 doctor` trên máy có Docker Desktop để nghiệm thu
   image/Compose engine; contract cold-start và health/readiness đã triển khai.
2. Attest model revision/runtime thật hoặc tiếp tục khóa dynamic mode.
3. Hoàn thành manual evidence và rehearsal 5 phút.
4. Demo ba ca: đúng, bất đồng được verifier cứu, thiếu dữ liệu thì fail-closed.

## Phán quyết cuối

- **Bê ngay từ mscAI:** lifecycle, manifest, readiness, durable batch, deployment.
- **Bê ngay từ Hung&Fong:** data normalization đối xứng, forensic audit, baseline
  fallback và phân biệt retrieval-vs-selector.
- **Bê có kiểm soát từ Nguyễn Văn Nghiêm:** hard negatives, cascade giảm noise,
  synthetic lexical bridge và checkpoint selection theo metric cuối.
- **Không bê:** mọi kỹ thuật dựa trên test memorization, pseudo-GT tự sinh hoặc
  LLM filter không có independent validation.
