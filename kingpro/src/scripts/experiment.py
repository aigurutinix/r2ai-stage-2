"""TỰ ĐỘNG HOÁ vòng thí nghiệm R2AI — để KHÔNG BAO GIỜ quên bước (nguyên tắc Vô thượng).

Mọi lần muốn nộp một submission dir, chạy DUY NHẤT:
    python scripts/experiment.py package <sub_dir>
Nó sẽ: (1) grader-check bằng .venv-grader (pandas 1.1.5, contract y máy chấm);
       (2) CỔNG CHẶN — nếu code không tái tạo đúng answer đã lưu thì TỪ CHỐI đóng gói;
       (3) zip + copy sang thư mục upload (VAIC-DDAY);
       (4) ghi vào docs/submissions-log.md (không quên kết quả).

Sau khi nộp xong trên leaderboard, ghi điểm lại:
    python scripts/experiment.py result <sub_name> <score> [ghi_chú]

Quy ước EMIT (nhắc để không quên): abs_tol 0.01; % -> round(ratio*100,2); hệ số/lần -> round(2);
tiền -> quy về đơn vị câu hỏi rồi round(2). Grader chỉ cấp dfs/df + biến result, whitelist builtins.
"""
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime

# Each script can be launched independently from a Windows shell or a piped
# subprocess, where the inherited console encoding is often cp1252.  Configure
# this entrypoint explicitly before printing Vietnamese status messages.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from kingpro.experiments import ExperimentLogbook

UPLOAD_DIR = "C:/Users/vinh/Downloads/VAIC-DDAY"   # thư mục chrome-devtools thấy để upload
GRADER_PY = os.path.join(ROOT, ".venv-grader/Scripts/python.exe")
LOG = os.path.join(ROOT, "docs/submissions-log.md")
AUTO_LOG = ExperimentLogbook(os.path.join(ROOT, "knowledge", "vothuong"))


def auto_log(kind, oracle, **fields):
    """Logging must be durable, but a log I/O problem must not corrupt an artifact."""
    try:
        return AUTO_LOG.append_event(kind, oracle=oracle, **fields)
    except Exception as exc:
        print(f"⚠️  Không ghi được Vô Thượng log: {exc}", file=sys.stderr)
        return None


def grader_check(sub_dir):
    out = subprocess.run([GRADER_PY, "scripts/grader_check.py", sub_dir],
                         cwd=ROOT, capture_output=True, text=True)
    txt = out.stdout.strip()
    try:
        start = txt.index("{")
        return json.loads(txt[start:])
    except Exception:
        print(txt[-2000:]); print(out.stderr[-1000:])
        raise SystemExit("grader_check không trả JSON — xem log trên")


def zip_dir(sub_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(sub_dir, "submission.json"), "submission.json")
        ddir = os.path.join(sub_dir, "data")
        for f in os.listdir(ddir):
            if f.endswith(".csv"):
                z.write(os.path.join(ddir, f), f"data/{f}")


def append_log(line):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    head = ""
    if not os.path.exists(LOG):
        head = "# Nhật ký nộp bài R2AI (tự động ghi bởi experiment.py)\n\n| Thời điểm | Bài | code sạch | rỗng | Điểm LB | Ghi chú |\n|---|---|---|---|---|---|\n"
    with open(LOG, "a", encoding="utf-8") as f:
        if head:
            f.write(head)
        f.write(line + "\n")


def package(sub_dir):
    sub_dir = sub_dir.rstrip("/\\")
    name = os.path.basename(sub_dir)
    subj = os.path.join(sub_dir, "submission.json")
    if not os.path.exists(subj):
        raise SystemExit(f"KHÔNG thấy {subj}")
    print(f"[1/4] grader-check {name} (contract y máy chấm)...", flush=True)
    st = grader_check(sub_dir)
    entries = st.get("entries", 0)
    withq = st.get("with_query", 0)
    empty = st.get("empty_query", 0)
    match = st.get("match_stored_answer", 0)
    errs = st.get("errors", {})
    print(f"      entries={entries} with_query={withq} empty={empty} match_stored={match} errors={errs}")
    # CỔNG CHẶN: code phải tái tạo đúng answer đã lưu, không thì grader sẽ chấm khác -> từ chối
    if match < withq:
        auto_log(
            "submission",
            "do-luong",
            name=name,
            stage="package_gate",
            result="thua",
            metric="runtime_match",
            value=f"{match}/{withq}",
            counts=st,
            note="Blocked before packaging because code did not reproduce stored answers.",
        )
        raise SystemExit(f"❌ CỔNG CHẶN: {withq-match} câu code KHÔNG tái tạo đúng answer đã lưu "
                         f"(match {match} < with_query {withq}). SỬA trước khi nộp. Không đóng gói.")
    if entries != 1012:
        print(f"      ⚠️  CẢNH BÁO: entries={entries} != 1012 (định dạng có thể sai).")
    print(f"[2/4] Cổng chặn OK ({match}/{withq} code sạch khớp answer).", flush=True)
    zip_path = os.path.join(ROOT, f"{name}.zip")
    zip_dir(sub_dir, zip_path)
    size = os.path.getsize(zip_path)
    print(f"[3/4] Zip -> {zip_path} ({size//1024} KB)", flush=True)
    up = os.path.join(UPLOAD_DIR, f"{name}.zip")
    import shutil
    shutil.copyfile(zip_path, up)
    print(f"[4/4] Copy -> {up} (sẵn sàng upload qua leaderboard)", flush=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_log(f"| {ts} | {name} | {match}/{withq} | {empty} | (chờ) | packaged |")
    log_record = auto_log(
        "submission",
        "do-luong",
        name=name,
        stage="packaged",
        result="thang",
        metric="runtime_match",
        value=f"{match}/{withq}",
        counts=st,
        artifacts=[zip_path, up],
        archive_bytes=size,
        note="Artifact passed the legacy experiment.py runtime gate and was copied to the local upload folder; not uploaded automatically.",
    )
    print()
    print("=" * 60)
    print(f"SẴN SÀNG NỘP: {name}.zip đã ở {UPLOAD_DIR}")
    print("Bước nộp (leaderboard participate-tab): upload file .zip -> chờ Scoring -> Finished.")
    print(f"Nộp xong ghi điểm: python scripts/experiment.py result {name} <score>")
    if log_record:
        print(f"Vô Thượng event: {log_record['id']}")
    print("=" * 60)


def result(name, score, note=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_log(f"| {ts} | {name} | — | — | **{score}** | {note or 'kết quả LB'} |")
    record = auto_log(
        "experiment",
        "leaderboard",
        name=name,
        metric="execution_accuracy",
        value=score,
        result="thang",
        note=note or "Leaderboard result",
    )
    print(f"Đã ghi: {name} = {score} vào {LOG}")
    if record:
        print(f"Vô Thượng event: {record['id']}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("package", "result"):
        print(__doc__)
        raise SystemExit(1)
    if sys.argv[1] == "package":
        package(sys.argv[2])
    else:
        result(sys.argv[2], sys.argv[3], " ".join(sys.argv[4:]))
