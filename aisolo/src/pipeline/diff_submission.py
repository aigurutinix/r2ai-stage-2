"""So bản nộp HIỆN TẠI với bản TRƯỚC → thay đổi này làm đổi câu nào, tốt lên hay xấu đi.

Vì sao cần: một thay đổi có thể trông "vô hại" theo chỉ số tổng nhưng đổi hàng chục đáp án.
Bản vá target_label từng bị đánh giá thấp (+0.005) vì không so được; khi so mới thấy nó sửa
25 câu từ dòng RÁC ("Mua trong năm", "Số cuối năm") thành đúng chỉ tiêu.

Chạy:  python diff_submission.py                     # so với submission.prev.json
       python diff_submission.py <file_khac.json>    # so với bản tải từ leaderboard
"""
import json, os, re, sys

OUT = os.path.join(os.path.dirname(__file__), "submission_out")
CUR = os.path.join(OUT, "submission.json")
PREV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "submission.prev.json")
LAB = re.compile(r'astype\(str\)==("(?:[^"\\]|\\.)*")')

if not os.path.exists(PREV):
    sys.exit(f"Không có bản trước để so: {PREV}\n(chạy build ít nhất 2 lần, hoặc truyền đường dẫn file)")

cur = {e["id"]: e for e in json.load(open(CUR, encoding="utf-8"))}
prev = {e["id"]: e for e in json.load(open(PREV, encoding="utf-8"))}

def label_of(pq):
    m = LAB.findall(pq or "")
    return json.loads(m[0]) if m else ""

d_ans = [i for i in cur if i in prev and cur[i]["answer"] != prev[i]["answer"]]
d_doc = sum(1 for i in cur if i in prev and cur[i]["relevant_docs"] != prev[i]["relevant_docs"])
d_tab = sum(1 for i in cur if i in prev and cur[i]["relevant_tables"] != prev[i]["relevant_tables"])

print(f"So {os.path.basename(PREV)} → submission.json")
print(f"  đổi ĐÁP ÁN         : {len(d_ans)}")
print(f"  đổi relevant_docs  : {d_doc}")
print(f"  đổi relevant_tables: {d_tab}")
print(f"\n=== {len(d_ans)} câu đổi đáp án (xem DÒNG được chọn để tự đánh giá tốt/xấu) ===")
for i in d_ans:
    print(f"\nid{i}: {prev[i]['answer']} → {cur[i]['answer']}")
    print(f"   Q : {cur[i]['question'][:100]}")
    print(f"   cũ : {label_of(prev[i]['pandas_query'])[:52]!r}")
    print(f"   mới: {label_of(cur[i]['pandas_query'])[:52]!r}")
