"""p_pick baseline on the held-out picker slice — the number that gates Stage C.

Measures ONE skill with no training and no submission cost: given a question and a
short list of rendered candidate rows (gold row + mined confusables), does the model
name the right row? The list contains labels only — no values, no arithmetic is
possible, so the id=252 x10 failure class is structurally absent.

Three arms over the same 53 questions:

  zero-shot   base instruct behaviour, the floor
  few-shot    three worked examples from train, measures whether the skill is
              elicitable by context alone (if yes, an adapter should reach higher)
  lexical     the label_match scorer the pipeline already trusts — S4 must beat this
              or it has no reason to exist

Usage:
  python scripts/fresh/probe_ppick.py                 # all arms
  python scripts/fresh/probe_ppick.py --arm zero      # one arm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_match import score as match_score  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-14b"
THINK_RE = re.compile(r"^.*</think>\s*", re.S)
JSON_RE = re.compile(r"\{[^{}]*\}")

SYSTEM = """Bạn chọn DÒNG chứa chỉ tiêu mà câu hỏi nêu, trong một bảng báo cáo tài chính Việt Nam.

Quy tắc:
1. Khớp TỪ NGỮ chỉ tiêu trong câu hỏi với nhãn dòng. Nhãn thừa hoặc thiếu chữ so với
   câu hỏi đều đáng nghi ("Tiền" hỏi chung thì không lấy dòng "Tiền mặt").
2. Câu hỏi nói "tổng/cộng" thì ưu tiên dòng tổng; nói một khoản mục chi tiết thì KHÔNG
   lấy dòng tổng.
3. Không tính toán gì. Chỉ đối chiếu từ ngữ.
4. Nếu hai dòng cùng khớp và câu hỏi không phân vách được, chọn -1.

Trả lời ĐÚNG MỘT DÒNG JSON: {"chon": <số thứ tự lựa chọn, 0-based>; dùng -1 nếu từ chối}"""

FEWSHOT = [
    {
        "q": "Tổng chi phí trả trước dài hạn của CTCP Nông nghiệp BAF Việt Nam (BAF) "
             "đến ngày 31/12/2020 là bao nhiêu đồng?",
        "choices": ["r2 Tiền thuê đất trả trước 1 lần (*)",
                    "r3 Các chỉ phí trả trước dài hạn khác",
                    "r4 Cộng"],
        "a": 2,
        "why": "câu hỏi nói 'TỔNG chi phí trả trước dài hạn' -> dòng tổng 'Cộng'.",
    },
    {
        "q": "Giá trị tài sản cố định vô hình của CTCP Viễn thông FPT (FOX) vào ngày "
             "31/12/2020 là bao nhiêu đồng?",
        "choices": ["r5 Phân loại lại sang tài sản cố định hữu hình",
                    "r7 Số dư cuối năm",
                    "r8 GIÁ TRI HAO MÒN LỦY KẾ",
                    "r12 Phân loại lại sang tài sản cố định hữu hình",
                    "r15 GIÁ TRI CÒN LẠI"],
        "a": 1,
        "why": "'giá trị TSCĐ vô hình' tại NGÀY 31/12 -> 'Số dư cuối năm'; hao mòn lũy kế "
               "và giá trị còn lại là các đo lường khác của cùng nhóm tài sản.",
    },
    {
        "q": "Chi phí dịch vụ mua ngoài của CTCP Tập đoàn PC1 (PC1) năm 2018 là bao nhiêu đồng?",
        "choices": ["r1 Chi phí nhân công", "r2 Chi phí dịch vụ mua ngoài",
                    "r3 Chi phí công cụ dụng cụ"],
        "a": 1,
        "why": "khớp đúng nguyên văn chỉ tiêu.",
    },
]


def call(question: str, choices: list[str], *, examples: int,
         key: str) -> dict:
    user_lines = [f"CÂU HỎI: {question}", "", "CÁC LỰA CHỌN:"]
    for i, label in enumerate(choices):
        user_lines.append(f"[{i}] {label}")
    user_lines.append("")
    if examples:
        for ex in FEWSHOT[:examples]:
            user_lines.append(f"VÍ DỤ:")
            user_lines.append(f"CÂU HỎI: {ex['q']}")
            for i, lab in enumerate(ex["choices"]):
                user_lines.append(f"[{i}] {lab}")
            user_lines.append(f'Trả lời: {{"chon": {ex["a"]}}}  # {ex["why"]}')
            user_lines.append("")
    user_lines.append('Trả lời chỉ một dòng JSON: {"chon": <số>}')

    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 2500,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(user_lines)},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"] or ""
            return {"raw": text}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    return {"raw": "", "error": str(last_err)}


def parse_choice(raw: str, n: int) -> int:
    text = THINK_RE.sub("", raw or "").strip()
    m = JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = int(obj.get("chon"))
            return v if -1 <= v < n else -2
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    nums = re.findall(r"-?\d+", text)
    if nums:
        v = int(nums[-1])
        return v if -1 <= v < n else -2
    return -2


def lexical_arm(pairs: list[dict]) -> tuple[int, int]:
    right = answered = 0
    for p in pairs:
        scores = [match_score([p["question"]], ch["label"]) for ch in p["choices"]]
        best = max(range(len(scores)), key=lambda i: scores[i])
        if max(scores) > 0:
            answered += 1
            right += int(best == p["answer_index"])
    return right, answered


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("zero", "fewshot", "lexical"), default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    key = ""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPEN_ROUTER_KEY="):
            key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("không thấy OPEN_ROUTER_KEY trong .env")

    pairs = [json.loads(line) for line
             in (ROOT / "artifacts" / "fresh" / "picker_heldout.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    arms = [args.arm] if args.arm else ["zero", "fewshot", "lexical"]
    results: dict[str, dict] = {}

    if "lexical" in arms:
        r, a = lexical_arm(pairs)
        results["lexical"] = {"right": r, "answered": a, "n": len(pairs)}

    lock = threading.Lock()
    for arm in ("zero", "fewshot"):
        if arm not in arms:
            continue
        rows: list[dict] = []
        started = time.time()

        def work(p: dict) -> None:
            res = call(p["question"], [ch["label"] for ch in p["choices"]],
                       examples=3 if arm == "fewshot" else 0, key=key)
            pick = parse_choice(res["raw"], len(p["choices"]))
            with lock:
                rows.append({
                    "id": p["id"], "pick": pick, "gold": p["answer_index"],
                    "n_choices": len(p["choices"]), "verified": True,
                })

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, pairs))
        right = sum(1 for r in rows if r["pick"] == r["gold"])
        refused = sum(1 for r in rows if r["pick"] == -1)
        invalid = sum(1 for r in rows if r["pick"] == -2)
        results[arm] = {"right": right, "answered": len(rows) - refused - invalid,
                        "refused": refused, "invalid": invalid, "n": len(rows),
                        "rows": rows}
        print(f"{arm}: {time.time() - started:.0f}s")

    print(f"\n=== p_pick trên {len(pairs)} câu held-out (model {MODEL}) ===")
    for name, r in results.items():
        base = f"{r['right']}/{r['n']} = {r['right'] / r['n']:.1%}"
        extra = ""
        if "answered" in r:
            extra = f"  · trả lời {r['answered']}, đúng/trả-lời " \
                    f"{r['right'] / max(1, r['answered']):.1%}"
        if "refused" in r:
            extra += f", từ chối {r['refused']}, hỏng-parse {r.get('invalid', 0)}"
        print(f"  {name:8s} {base}{extra}")

    out = ROOT / "artifacts" / "fresh" / "ppick_baseline.jsonl"
    dump = []
    for name in ("zero", "fewshot"):
        if name in results:
            for row in results[name].get("rows", []):
                dump.append({**row, "arm": name})
    if dump:
        out.write_text("\n".join(json.dumps(d, ensure_ascii=False)
                                 for d in dump), encoding="utf-8")
        print(f"\nchi tiết -> {out.relative_to(ROOT)}")

    gate = min((results[a]["right"] for a in ("zero", "fewshot") if a in results),
               default=-1)
    print("\nNGƯỠNG đã đăng ký: >=37/53 (70%) xây S4 đầy đủ · "
          "29-36 phá-và-giữ · <29 chết S4")


if __name__ == "__main__":
    main()
