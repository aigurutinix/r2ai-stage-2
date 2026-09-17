"""Pipeline trial slice DIRECT-LOOKUP: ingest .txt→long rows (dò cột Mã-số động + unit robust)
+ extractor 2 nhánh (Mã-số main / label notes / %-ownership). Chạy 30 câu direct-lookup để verify."""
import json, re, csv, unicodedata, math
from pathlib import Path
import os as _os
# Mặc định trỏ vào `data_vifinqa/` CẠNH repo — người khác clone về, tải dữ liệu theo README là
# chạy được ngay. Trước đây mặc định là một đường dẫn máy cá nhân; ai không đặt VIFINQA_ROOT sẽ đọc
# một thư mục KHÔNG TỒN TẠI, và vì `glob` trên thư mục vắng chỉ trả rỗng nên bài nộp hỏng IM LẶNG
# (mọi câu đều fallback) thay vì báo lỗi. Đặt VIFINQA_ROOT để trỏ nơi khác.
_BASE = _os.environ.get("VIFINQA_ROOT",
                        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data_vifinqa"))
from collections import defaultdict, Counter
ROOT = Path(_BASE)
if not (ROOT / "financial_statements").is_dir():
    raise SystemExit(
        f"[LOI] Khong thay du lieu ViFinQA tai: {ROOT.resolve()}\n"
        f"       Thieu thu muc con 'financial_statements/'.\n"
        f"       Tai du lieu:  git clone https://huggingface.co/datasets/AIGuruTinix/ViFinQA data_vifinqa\n"
        f"       Hoac tro noi khac:  VIFINQA_ROOT=<duong/dan> python ...")

def strip_vn(s):
    s=s.replace("đ","d").replace("Đ","D")
    return "".join(c for c in unicodedata.normalize("NFD",s) if unicodedata.category(c)!="Mn").lower()
PREFIX=["cong ty co phan","cong ty tnhh","tong cong ty co phan","tong cong ty","ngan hang tmcp",
        "ngan hang thuong mai co phan","tap doan","ctcp","cong ty","ngan hang"]
def core(name):
    n=strip_vn(name)
    for p in sorted(PREFIX,key=len,reverse=True): n=n.replace(p," ")
    return re.sub(r"\s+"," ",n).strip()
tick2core={}
with (ROOT/"code_stock.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f): tick2core[r["Mã CK"].strip()]=core(r["Tên công ty"])
tickers=set(tick2core)
# Clean-core (bo dau gach '-' thua) cho fuzzy fallback: khop CUM LIEN MACH nhu resolve() goc
tick2clean={t:re.sub(r"\s+"," ",re.sub(r"[^a-z0-9]+"," ",c)).strip() for t,c in tick2core.items()}
tick2clean={t:cc for t,cc in tick2clean.items() if len(cc)>=8}
def resolve_fuzzy(text):
    """Fallback ten->ma CK: clean-core (vd 'phan bon dau khi ca mau') xuat hien LIEN MACH trong cau.
    Chi tra khi co ung vien du dai va KHONG nhap nhang (2 ung vien manh -> cau da cong ty -> None)."""
    low=strip_vn(text)
    cands=sorted(((len(cc),t) for t,cc in tick2clean.items() if cc in low), reverse=True)
    if not cands: return None
    strong=[c for c in cands if c[0]>=12]
    if len(strong)>=2: return None  # >=2 ten cong ty day du -> cau so sanh 2 cong ty -> None
    return cands[0][1] if cands[0][0]>=8 else None
def resolve_all_names(text, minlen=8):
    """TẤT CẢ mã CK có tên công ty xuất hiện trong câu (resolve_fuzzy chỉ trả 1 và bỏ cuộc khi
    ≥2 ứng viên mạnh → câu 'Masan, Đại Dương và Vinamilk' mất sạch doc).
    Đo được: 92/1012 câu thiếu công ty trong relevant_docs → DOCS recall 0.886 vs 0.95 của đội khác.
    minlen=8 (khớp resolve_fuzzy gốc): hạ từ 10→8 thêm 23 doc ĐÚNG mà report có thật
    (vd câu nêu thẳng "Tập đoàn VINGROUP" nhưng thiếu VIC, "Tập đoàn Hòa Phát" thiếu HPG).
    Guard: bỏ ứng viên có clean-core NẰM TRONG core của ứng viên dài hơn đã khớp (tránh
    'dầu khí việt nam' kéo theo 'khí việt nam')."""
    low = strip_vn(text)
    cands = sorted(((len(cc), t, cc) for t, cc in tick2clean.items()
                    if len(cc) >= minlen and cc in low), reverse=True)
    keep, taken = [], []
    for _, t, cc in cands:
        if any(cc in longer for longer in taken):   # bị lồng trong tên dài hơn → bỏ
            continue
        keep.append(t); taken.append(cc)
    # BỔ SUNG (không phải fallback độc quyền): alias tên thị trường cho công ty có core < minlen
    # (vd "a chau" 6 ký tự bị loại khỏi tick2clean) — câu có NHIỀU công ty, vài công ty đã khớp qua
    # clean-core, vài công ty khác chỉ khớp được qua alias → phải GỘP, không phải "chỉ dùng khi keep rỗng".
    for t in alias_names(text):
        if t not in keep:
            keep.append(t)
    return keep

# Tên THỊ TRƯỜNG → mã CK. `code_stock.csv` chỉ lưu tên PHÁP LÝ ("quan doi" cho MBBank,
# "xuat nhap khau viet nam" cho Eximbank) nên câu hỏi gọi công ty bằng tên quen thuộc thì mọi
# tầng khớp theo tên đều trượt — đo được 3 câu mất trắng cả relevant_docs lẫn relevant_tables.
# Chỉ dùng khi các tầng kia đã trả rỗng, nên KHÔNG thể phá câu đang khớp đúng (vd "Chứng khoán
# FPT" vẫn ra FTS chứ không bị ép về FPT). Đã kiểm mọi mã dưới đây đều có báo cáo trong kho.
ALIAS = {"mbbank": "MBB", "mb bank": "MBB", "eximbank": "EIB", "kinh bac": "KBC",
         "vinamilk": "VNM", "vietcombank": "VCB", "vietinbank": "CTG", "sacombank": "STB",
         "sai gon thuong tin": "STB",   # code_stock.csv GHI SAI tên STB thành "Sài Gòn Tài Lộc"
                                         # (báo cáo thật là Sacombank/Sài Gòn Thương Tín — đã xác minh
                                         # đọc trực tiếp file OCR) → core() tự động sai theo, phải đè tay.
         "vpbank": "VPB", "bidv": "BID", "vietjet": "VJC", "vingroup": "VIC",
         "the gioi di dong": "MWG", "hoa phat": "HPG", "hoa sen": "HSG", "nam kim": "NKG",
         "petrolimex": "PLX", "sabeco": "SAB", "novaland": "NVL",
         # Core tên < minlen=8 nên bị loại khỏi tick2clean (đo được 16 mã, đây là các mã đã xác nhận
         # gây mất công ty thật trong câu hỏi — xem alias_names()/resolve_all_names()).
         "a chau": "ACB", "nam a": "NAB", "an binh": "ABB", "bac a": "BAB", "viet a": "VAB"}

def alias_names(text):
    low = strip_vn(text)
    return list(dict.fromkeys(t for a, t in ALIAS.items() if a in low and t in tickers))

def resolve(text):
    low=strip_vn(text)
    for m in re.findall(r"\(([A-Za-z0-9]{2,4})\)",text):
        if m.upper() in tickers: return m.upper()
    best,bl=None,0
    for t,c in tick2core.items():
        if c and len(c)>=6 and c in low and len(c)>bl: best,bl=t,len(c)
    if best: return best
    for m in re.findall(r"\b([A-Z0-9]{2,4})\b",text):
        if m in tickers: return m
    for m in re.findall(r"\b([a-z]{3})\b",text):
        if m.upper() in tickers: return m.upper()
    best,bl=None,0
    for t,c in tick2core.items():
        if c and len(c)>=4 and c in low and len(c)>bl: best,bl=t,len(c)
    if best: return best
    rf = resolve_fuzzy(text)
    if rf: return rf
    al = alias_names(text)               # lưới cuối; ≥2 tên → câu so sánh, giữ None như thiết kế cũ
    return al[0] if len(al) == 1 else None
def doctype(t): return "separate" if "công ty mẹ" in t.lower() else "consolidated"
def find_report(tk,year,prefer):
    d=ROOT/"financial_statements"/tk/year
    if not d.exists(): return None
    docs=[p for p in d.glob("*/") if p.is_dir()]
    if not docs: return None
    pref=[p for p in docs if prefer in p.name.lower()]; cons=[p for p in docs if "consolidated" in p.name.lower()]
    ch=(pref or cons or docs)[0]; txts=list(ch.glob("*.txt"))
    return (txts[0],ch.name) if txts else None

CODE=re.compile(r"^\d{2,3}$")
NOTE_RE=re.compile(r"^(?:[IVX]+\.)?\d{1,2}(?:\.\d{1,2})?$")   # số thuyết minh: "6", "12", "V.01", "5.1"
_note_cache={}
def note_table_line(txt, note):
    """Bảng THUYẾT MINH ứng với số hiệu `note` → số dòng <table>, hoặc None.

    Tìm tiêu đề mục dạng '6. Tiền và các khoản tương đương tiền' rồi lấy <table> đầu tiên
    đứng sau nó. Cho tín hiệu chọn bảng thứ ba, độc lập với BM25 lẫn anchor line-item."""
    if not note: return None
    key=(id(txt) if len(txt)<1 else hash(txt), note)
    if key in _note_cache: return _note_cache[key]
    num=note.split(".")[-1].lstrip("0") or note      # "V.01" → "1"
    res=None
    m=re.search(rf"^\s*{re.escape(num)}\.\s+\S[^\n]{{5,70}}$", txt, re.M)
    if m:
        t=re.search(r"<table>", txt[m.end():])
        if t: res=txt.count("\n",0,m.end()+t.start())+1
    _note_cache[key]=res
    return res
MONEY=re.compile(r"^\(?-?\d{1,3}(?:\.\d{3})+\)?$")
def money(s):
    s=s.strip()
    if not MONEY.match(s): return None
    neg=s.startswith("("); return (-1 if neg else 1)*int(s.strip("()").replace(".",""))
def cells(tr): return [re.sub(r"<[^>]*>","",c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>",tr,re.S)]

_SPAN=re.compile(r'(rowspan|colspan)\s*=\s*"?(\d+)"?',re.I)
EXPAND_SPAN=_os.environ.get("EXPAND_SPAN","0")=="1"
def expand_rows(body):
    """<tr> → lưới ô CHỮ NHẬT, đã mở rộng rowspan/colspan (bước BTC mô tả ở slide 14).

    Vì sao: đo trên corpus có 6.411 ô mang thuộc tính span, 60/60 file đều dính, và 826 dòng bị
    THIẾU ô so với dòng trước — tức mọi ô sau đó lệch sang trái, giá trị tiền gán nhầm cột.
    Ô rowspan chiếm chỗ ở đúng cột đó trong các dòng kế tiếp; ô colspan nhân bản sang phải.
    """
    out=[]; pend={}                       # cột -> [nội dung, số dòng còn chiếm]
    for tr in re.findall(r"<tr>(.*?)</tr>",body,re.S):
        row=[]; col=0
        def drain():
            nonlocal col
            while col in pend:
                row.append(pend[col][0]); pend[col][1]-=1
                if pend[col][1]<=0: del pend[col]
                col+=1
        for attr,raw in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>",tr,re.S):
            drain()
            sp={k.lower():int(v) for k,v in _SPAN.findall(attr)}
            txt=re.sub(r"<[^>]*>","",raw).strip()
            for _ in range(max(1,sp.get("colspan",1))):
                row.append(txt)
                if sp.get("rowspan",1)>1: pend[col]=[txt,sp["rowspan"]-1]
                col+=1
        drain()
        out.append(row)
    return out

def stmt_type(tabtext):
    # So khớp trên bản ĐÃ BỎ DẤU. OCR đọc "NỢ" thành "NỘ" ở 27% lần xuất hiện cụm "nợ phải trả"
    # (đo trên 10 doanh nghiệp: 454/1654 lần, 5 biến thể). Bản so chuỗi CÓ DẤU trượt hết, và bảng
    # NGUỒN VỐN rơi xuống nhánh PL vì nó chứa "Lợi nhuận sau thuế chưa phân phối" (mã 421) — tức
    # nửa dưới bảng cân đối bị gắn nhầm loại. Đo trên 1887 báo cáo: số cặp công ty-năm kiểm được
    # đẳng thức kế toán 32.1% -> 76.5%, tỷ lệ thoả giữ nguyên ~98%. Xem probe_bs_identity.py.
    l=strip_vn(tabtext).lower()
    if "luu chuyen tien" in l: return "CF"
    if "tong cong tai san" in l or "tong tai san" in l or "no phai tra" in l: return "BS"
    if "loi nhuan sau thue" in l or "doanh thu thuan" in l or "gia von" in l: return "PL"
    return "NOTE"

def find_total_assets(tables_rows):
    for rows in tables_rows:
        for r in rows:
            j=" ".join(r).lower()
            if "tổng cộng tài sản" in j or ("tổng tài sản" in j and len(r)>=3):
                vals=[money(c) for c in r if money(c) is not None]
                if vals: return vals[0]
    return None

SUBITEM=re.compile(r"^\s*[-+•·–—]\s*\S")     # nhãn cấp con: "- Bằng VND", "+ Ngắn hạn"
def _row_label(r):
    """Nhãn của dòng = ô có chữ dài nhất (bỏ ô mã số / ô số tiền)."""
    lab=""
    for c in r:
        if strip_vn(c) and re.search(r"[a-z]{3}",strip_vn(c)) and money(c) is None and len(c)>len(lab): lab=c
    return lab

TA_LO,TA_HI=1e10,2e16       # tổng tài sản doanh nghiệp niêm yết VN: 10 tỷ .. 20 triệu tỷ VND
def _plaus(ta,f): return bool(ta) and TA_LO<=abs(ta*f)<=TA_HI

def detect_unit(txt,tables_rows):
    low=txt.lower()
    ctx=r"(số (?:cuối|đầu) (?:năm|kỳ)|đơn vị[:\s]*tính?|làm tròn đến hàng)[^<\n]{0,25}?"
    ta=find_total_assets(tables_rows)                       # suy theo độ lớn total assets (an toàn hơn loose-marker)
    if re.search(ctx+r"triệu\s*đồng",low): base=(1e6,"marker-triệu")
    elif re.search(ctx+r"(?:nghìn|ngàn)\s*đồng",low): base=(1e3,"marker-nghìn")
    elif ta and ta>0 and len(str(int(abs(ta))))<11: base=(1e6,"magnitude")
    else: base=(1,"VND")                                     # BỎ loose-marker: "triệu đồng" trong prose gây false-positive ×1e6
    # CHỐT TỰ KIỂM: hệ số nào làm TỔNG TÀI SẢN ra ngoài khoảng hợp lý thì sai — marker bắt trúng chữ
    # "triệu đồng" trong văn xuôi từng thổi TTS lên 2.6e19 (VGC, SNZ). Chỉ sửa khi có hệ số KHÁC đưa
    # được TTS về khoảng hợp lý; không có thì GIỮ NGUYÊN, vì thà giữ nguyên còn hơn đoán bừa.
    if ta and not _plaus(ta,base[0]):
        for f in (1,1e3,1e6,1e9):
            if _plaus(ta,f): return f,base[1]+"-sua"
    return base

def ingest(txt):
    """→ (rows, unit_factor, unit_src). rows: dict(table_id, statement, ma_so, label, cur, prev)."""
    tmatch=list(re.finditer(r"<table>(.*?)</table>",txt,re.S))
    tables=[m.group(1) for m in tmatch]
    table_lines=[txt.count("\n",0,m.start())+1 for m in tmatch]   # SỐ DÒNG (1-based) nơi <table> bắt đầu trong file OCR = "vị trí" BTC
    tables_rows=([expand_rows(tb) for tb in tables] if EXPAND_SPAN
                 else [[cells(tr) for tr in re.findall(r"<tr>(.*?)</tr>",tb,re.S)] for tb in tables])
    uf,src=detect_unit(txt,tables_rows)
    rows=[]
    for tid,(tabtext,trows) in enumerate(zip(tables,tables_rows)):
        st=stmt_type(tabtext)
        # NGỮ CẢNH CHA trong bảng (ctx): bảng thuyết minh hay có nhãn con TRÙNG HỆT nhau, chỉ phân
        # biệt được bằng dòng cha ("Tiền gửi có kỳ hạn" → "- Bằng VND" vs "Cho vay TCTD" → "Bằng VND").
        # Không dùng quy tắc "tiêu đề = dòng không có số" vì trong dữ liệu OCR này dòng cha CŨNG có số.
        labs=[_row_label(r) for r in trows]
        dup={l for l in labs if l and labs.count(l)>1}
        # dò cột code: cột có >=3 dòng khớp ^\d{2,3}$
        colc=defaultdict(int)
        for r in trows[:30]:
            for j,c in enumerate(r):
                if CODE.match(c.strip()): colc[j]+=1
        codecol=next((j for j,c in sorted(colc.items()) if c>=3),None)
        # HEADER cột số: dòng đầu tiên KHÔNG có số tiền nào. Chỉ dùng khi số ô header (bỏ ô nhãn)
        # KHỚP ĐÚNG số cột tiền của dòng dữ liệu — lệch thì bỏ qua, thà giữ hành vi cũ còn hơn gán bừa.
        nmoney=[sum(1 for c in r if money(c) is not None) for r in trows if r]
        ncol=Counter(n for n in nmoney if n>0).most_common(1)[0][0] if any(nmoney) else 0
        hdrs=[]
        if ncol>2:
            hrow=next((r for r in trows if r and not any(money(c) is not None for c in r)),None)
            if hrow:
                cand=[c for c in hrow if strip_vn(c)] or hrow
                if len(cand)==ncol: hdrs=cand
                elif len(cand)==ncol+1: hdrs=cand[1:]        # ô đầu là cột nhãn
        parent=""                       # nhãn cha gần nhất (dòng cấp 0, không phải nhãn bị lặp)
        for r in trows:
            if not r: continue
            ma=r[codecol].strip() if (codecol is not None and codecol<len(r) and CODE.match(r[codecol].strip())) else ""
            lab=_row_label(r)
            vals=[money(c) for c in r if money(c) is not None]
            child=bool(SUBITEM.match(lab)) or (lab in dup)   # cấp con: có gạch đầu dòng HOẶC nhãn bị lặp
            ctx=parent if child else ""
            if lab and not child: parent=lab
            if not lab and not ma: continue
            # cột "Thuyết minh" đứng ngay sau cột Mã số ("110" → "6") — trỏ tới bảng thuyết
            # minh chi tiết của chính chỉ tiêu đó. Liên kết này do BCTC tự khai, chính xác
            # tuyệt đối và độc lập hoàn toàn với BM25.
            note=""
            if codecol is not None and codecol+1<len(r):
                nc=r[codecol+1].strip()
                if NOTE_RE.match(nc): note=nc
            rows.append({"tid":tid,"line":table_lines[tid],"st":st,"ma":ma,"label":lab,"note":note,"ctx":ctx,
                         "hdrs":(hdrs if len(vals)==len(hdrs) else []),"vals":[v*uf for v in vals],
                         "cur":(vals[0]*uf if vals else None),"prev":(vals[1]*uf if len(vals)>1 else None)})
    return rows,uf,src

MAIN_CODE={"tong tai san":"270","von chu so huu":"400","tien va cac khoan tuong duong tien":"110",
    "no phai tra":"300","doanh thu thuan":"10","gia von hang ban":"11","loi nhuan gop":"20",
    "loi nhuan sau thue":"60","loi nhuan truoc thue":"50","tai san ngan han":"100",
    "tai san dai han":"200","hang ton kho":"140","tong cong tai san":"270"}
# Mệnh đề thời gian ĐẦU CÂU ("Đến ngày 31/12/2022, ...", "Năm 2024, ..."): phải cắt TRƯỚC,
# vì hàm dưới chỉ cắt mốc thời gian đứng SAU chỉ tiêu. Đo được 69 câu bị dính nguyên cụm ngày
# tháng vào target_label — làm hỏng cả locate() (chọn dòng) LẪN query BM25 (chọn bảng).
LEAD_TIME=re.compile(r"^\s*(?:đến ngày|vào ngày|tính đến ngày|tại thời điểm|trong giai đoạn|"
                     r"trong năm|vào cuối năm|vào đầu năm|vào năm|năm|cuối năm|đầu năm)\s+[^,]{0,40},\s*",re.I)
def target_label(q):
    q=LEAD_TIME.sub("",q,count=1)
    # cắt ở ranh giới CÔNG TY (của công ty mẹ/ctcp/ngân hàng/tập đoàn/<TICKER>) hoặc mốc THỜI GIAN,
    # KHÔNG cắt ở "của" giữa cụm metric ("phải thu ngắn hạn CỦA khách hàng")
    m=re.search(r"\scủa (công ty mẹ|ctcp|ngân hàng tmcp|ngân hàng|tập đoàn|tổng công ty)\b"
                r"|\scủa [A-Z]{2,4}\b"
                r"|\s(năm |cuối (năm|kỳ)|đầu (năm|kỳ)|vào ngày|đến ngày|tại thời điểm|trong năm|vào cuối)",q,re.I)
    lab=q[:m.start()] if m else q
    return re.sub(r"^(số dư|số tiền|số|tổng số|tổng cộng|tổng|giá trị|tổng giá trị)\s+","",lab.strip(),flags=re.I).strip()
def toks(s): return set(w for w in strip_vn(s).split() if len(w)>=2)
def toks_list(s): return [w for w in strip_vn(s).split() if len(w)>=2 and not w.isdigit()]   # giữ trùng lặp (TF) cho BM25

# pyvi word-segmentation CHỈ cho TABLE RETRIEVAL (ghép từ ghép 'lợi_nhuận_sau_thuế' → token discriminative
# cho BM25 tiếng Việt, như baseline 0.89). KHÔNG dùng cho answer-path (locate) → non-regressive answer.
try:
    from pyvi import ViTokenizer
    _USE_PYVI = True
except Exception:
    # Chạy nhầm Python không có pyvi thì BM25 đổi hoàn toàn mà không báo lỗi — đã làm hỏng
    # một lượt nộp (07/08: tưởng anchor kém hơn, thực ra thí nghiệm nhiễu 2 biến).
    import sys as _sys
    print("[CANH BAO] KHONG co pyvi -> BM25 chon bang SAI so voi ban da do.\n"
          "           Cai bang: pip install pyvi (nho chay dung venv cua du an)", file=_sys.stderr)
    _USE_PYVI = False
def toks_seg(s):
    seg = ViTokenizer.tokenize(s) if _USE_PYVI else s
    return [w for w in strip_vn(seg).split() if len(w) >= 2 and not w.isdigit()]

_tt_cache = {}   # memoize theo report (pyvi CRF chậm; nhiều câu chung 1 report)
def table_tokens(txt):
    """Per-table token-bag (CHỈ text cell trong bảng, word-segmented pyvi) + số dòng.
    Đo A/B tỷ lệ trúng top-1: chỉ-nội-dung-bảng 36.1% > kèm-context-1500-ký-tự 33.7%
    (context kéo theo header/đoạn văn của bảng trước → các bảng trông giống nhau, loãng IDF).
    pyvi giúp rõ rệt: +7 điểm so với token thường."""
    h = hash(txt)
    c = _tt_cache.get(h)
    if c is not None: return c
    out = []
    ms = list(re.finditer(r"<table>(.*?)</table>", txt, re.S))
    for tid, m in enumerate(ms):
        line = txt.count("\n", 0, m.start()) + 1
        raw = re.sub(r"<[^>]*>", " ", m.group(1))
        out.append({"tid": tid, "line": line, "toks": toks_seg(raw)})
    _tt_cache[h] = out
    return out

def bm25_scores(query_toks, docs_toks, k1=1.5, b=0.75):
    """BM25 điểm cho từng 'doc' (bảng) theo query. IDF tính TRONG report (từ metric hiếm→cao, từ chung→thấp)."""
    N=len(docs_toks)
    if N==0: return []
    df=defaultdict(int)
    for d in docs_toks:
        for w in set(d): df[w]+=1
    avgdl=(sum(len(d) for d in docs_toks)/N) or 1
    qset=set(query_toks); out=[]
    for d in docs_toks:
        dl=len(d); tf=defaultdict(int)
        for w in d: tf[w]+=1
        s=0.0
        for w in qset:
            f=tf.get(w,0)
            if not f: continue
            idf=math.log(1+(N-df[w]+0.5)/(df[w]+0.5))
            s+=idf*f*(k1+1)/(f+k1*(1-b+b*dl/avgdl))
        out.append(s)
    return out

def qdir_of(q):
    ql=q.lower()
    return ("cuoi" if re.search(r"cuối (năm|kỳ)|đến ngày|vào ngày|31/12|tại thời điểm",ql)
            else "dau" if re.search(r"đầu (năm|kỳ)|01/01",ql) else None)

def clean_metric(s):
    """Bỏ prefix tính toán để lấy đúng cụm metric (cho câu multi-year/company)."""
    s=re.sub(r"^\s*so (sánh|với)\s+[^,]{0,40},\s*","",s,flags=re.I).strip()
    s=re.sub(r"^\s*(tính\s+)?((tỷ lệ|phần trăm|tốc độ|mức)\s+)?(tăng trưởng|tăng|giảm)\s+(của\s+)?","",s,flags=re.I).strip()
    s=re.sub(r"^\s*(hiệu số|chênh lệch)\s+(giữa\s+)?","",s,flags=re.I).strip()
    return s

def locate_like(rows, maso, label):
    """Tìm ĐÚNG cùng line-item với anchor (ma_so ưu tiên, else label Jaccard cao) — đồng bộ giữa report."""
    if maso:
        cand=[r for r in rows if r["ma"]==maso and r["cur"] is not None]
        if cand:
            r=cand[0]
            return {"maso":maso,"label":r["label"],"tid":r["tid"],"cur":r["cur"],"prev":r["prev"]}
    at=toks(label); alow=strip_vn(label); best=None
    for r in rows:
        if r["cur"] is None or not r["label"]: continue
        rlow=strip_vn(r["label"])
        if rlow==alow:   # khớp label tuyệt đối
            return {"maso":r["ma"],"label":r["label"],"tid":r["tid"],"cur":r["cur"],"prev":r["prev"]}
        rt=toks(r["label"]); j=len(at&rt)/max(1,len(at|rt))   # Jaccard: phạt token thừa/thiếu (có/không kỳ hạn)
        if best is None or j>best[0]: best=(j,r)
    if best and best[0]>=0.7:   # đủ giống anchor mới nhận
        r=best[1]
        return {"maso":r["ma"],"label":r["label"],"tid":r["tid"],"cur":r["cur"],"prev":r["prev"]}
    return None

def parse_pct(s):
    m=re.search(r"(\d{1,3}(?:[.,]\d+)?)\s*%",s)
    return float(m.group(1).replace(",",".")) if m else None

def own_tables(txt):
    """Bảng 'công ty con/liên kết' có cột Tỷ lệ lợi ích / quyền biểu quyết → [{tid,line,rows:[{name,loi,vote}]}]."""
    out=[]
    for tid,m in enumerate(re.finditer(r"<table>(.*?)</table>",txt,re.S)):
        body=m.group(1); line=txt.count("\n",0,m.start())+1
        trows=[cells(tr) for tr in re.findall(r"<tr>(.*?)</tr>",body,re.S)]
        header=None; hi=0
        for i,r in enumerate(trows[:3]):
            j=" ".join(r).lower()
            if "tỷ lệ" in j and ("lợi ích" in j or "biểu quyết" in j or "sở hữu" in j): header=r; hi=i; break
        if not header: continue
        cl=next((j for j,c in enumerate(header) if "lợi ích" in c.lower() or "sở hữu" in c.lower()),None)
        cv=next((j for j,c in enumerate(header) if "biểu quyết" in c.lower()),None)
        cn=next((j for j,c in enumerate(header) if "tên" in c.lower() or "công ty" in c.lower()),None)
        if cn is None: cn=1 if len(header)>1 else 0   # fallback: cột sau Stt
        rows=[]
        for r in trows[hi+1:]:
            if cn>=len(r): continue
            name=r[cn].strip()
            if len(name)<6 or parse_pct(name) is not None: continue
            loi=parse_pct(r[cl]) if (cl is not None and cl<len(r)) else None
            vote=parse_pct(r[cv]) if (cv is not None and cv<len(r)) else None
            if loi is None and vote is None: continue
            rows.append({"name":name,"loi":loi,"vote":vote})
        if rows: out.append({"tid":tid,"line":line,"rows":rows})
    return out

def own_extract(q, txt):
    """Câu 'tỷ lệ sở hữu/biểu quyết X của Y': match tên công ty con trong câu → % đúng cột."""
    tabs=own_tables(txt)
    if not tabs: return None
    qc=set(core(q).split()); vote="biểu quyết" in q.lower()
    best=None
    for t in tabs:
        for r in t["rows"]:
            nc=set(core(r["name"]).split())
            if not nc: continue
            ov=len(nc&qc)/max(1,len(nc))   # tên công ty con xuất hiện trong câu (chuẩn hoá theo token tên)
            if best is None or ov>best[0]: best=(ov,t,r)
    if not best or best[0]<0.6: return None
    _,t,r=best; val=r["vote"] if vote else r["loi"]
    if val is None: val=r["loi"] if r["loi"] is not None else r["vote"]
    if val is None: return None
    return {"line":t["line"],"tid":t["tid"],"name":r["name"],"val":val,"rows":t["rows"],"vote":vote}

def count_extract(qt, rows, uf):
    """Câu đếm SỐ LƯỢNG cổ phiếu/cổ phần → dòng share-count, trả {label,tid,raw}.
    raw = cur/uf (số cổ phiếu KHÔNG phải tiền → hoàn tác hệ số đơn vị); lọc [1e5,1e11] tách khỏi giá trị tiền."""
    ql = strip_vn(qt)
    if "binh quan" in ql or "gia quyen" in ql:
        pats = ["binh quan"]
    elif "dang luu hanh" in ql:
        pats = ["dang luu hanh"]
    else:
        pats = ["dang luu hanh", "da phat hanh", "pho thong"]
    best = None
    for r in rows:
        if r["cur"] is None or not r["label"]:
            continue
        lab = strip_vn(r["label"])
        if not ("co phieu" in lab or "co phan" in lab):
            continue
        if not any(p in lab for p in pats):
            continue
        raw = r["cur"] / uf
        if not (1e5 < abs(raw) < 1e11):   # số cổ phiếu hợp lý (loại giá trị tiền)
            continue
        if best is None or "so luong" in lab:   # ưu tiên dòng bắt đầu "số lượng"
            best = {"label": r["label"], "tid": r["tid"], "cur": r["cur"]}
    return best

# Trọng số ngữ cảnh cha. ĐÃ THỬ BẬT 0.25 (13/08) rồi TẮT LẠI — bài học về phạm vi phép đo:
# trên dev set nó cho +2 câu và 0 ca tụt, NHƯNG dev set chỉ phủ ĐƯỜNG TRA CỨU CHÍNH. Bật toàn cục
# thì nó đổi cả lựa chọn dòng ở nhánh RATIO-PAIR và làm hỏng id731 (tỷ lệ LDR của HDB: 96.14% →
# 70524.74%). Lợi ích nằm trong biên nhiễu, tác hại ở nhánh không đo được là có thật ⇒ để 0.
# Muốn bật lại thì phải đo được cả nhánh ratio/diff, không chỉ nhánh lookup.
CTX_W=float(_os.environ.get("CTX_W","0"))
# TÁCH HAI HƯỚNG CHỨA NHAU — đã nộp thử và GIỮ LẠI (14/08).
#   CONT_IN  nhãn CHỨA TRỌN target ("Vốn cổ phần đã phát hànhCổ phiếu phổ thông" ⊇ "Vốn cổ phần
#            đã phát hành") — khớp đúng thứ được hỏi rồi mô tả thêm.
#   CONT_OF  nhãn chỉ là KHÚC CẮT ("Vốn cổ phần" ⊂ "Vốn cổ phần đã phát hành") — đã rụng định ngữ.
# Trước đây cộng +0.15 NHƯ NHAU cho cả hai hướng, nên hai nhãn cách nhau đúng 0.001 điểm.
#
# Đường đi của quyết định này đáng giữ lại vì nó ngược với trực giác:
#   - dev set chia đôi: dấu ĐÚNG và ỔN ĐỊNH (mọi ô đều đẩy nửa nghiệm thu lên, không ô nào tụt)
#     nhưng độ lớn chỉ +3/312 (~1%) — DƯỚI ngưỡng đã nâng lên sau khi thử 4 giả thuyết cùng ngày.
#   - soi tay 22 câu đổi đáp án: khoảng 8-9 ca tốt lên (sửa mảnh nhãn OCR vụn: "Băng VND" →
#     "Tiền mặt bằng VND"), 7-8 ca tệ đi (id957 "TỔNG TÀI SẢN" → "Nguyên giá TSCĐ") ⇒ trông như
#     xáo bài, và kỳ vọng bị hạ xuống ~50/50 TRƯỚC khi nộp.
#   - leaderboard (id 3023): Execution 0.3300 → 0.3340 (+0.0040, ~2 câu/506); TABLES F2 nhích
#     xuống 0.4506 → 0.4487 vì đổi dòng chọn thì kéo theo đổi bảng trả về; DOCS giữ 0.9451.
# ⇒ Bài học: thước đo nội bộ không đủ ĐỘ PHÂN GIẢI cho hiệu ứng cỡ 1%, và soi tay cũng không —
#   chỉ trọng tài thật phân xử được. Nhưng +0.004 là nhỏ, ĐỪNG suy rộng thành "hướng này còn dư địa".
CONT_IN =float(_os.environ.get("CONT_IN","0.25"))
CONT_OF =float(_os.environ.get("CONT_OF","0.05"))
# TIE_LONG: khi HOA DIEM tuyet doi thi nhan DAI thang thay vi nhan NGAN.
# Ly do: diem `ov` da la F1 nen da PHAT token thua cua nhan dai roi; uu tien nhan ngan o khoa phu
# la phat lan thu hai tren cung mot truc. Do duoc (AGENTS.md 18/08): o 19 cau hoa tuyet doi, cach
# hien tai trung 0/19 — te hon boc tham (~25%), tuc tin hieu dang bi dung NGUOC dau.
# KET QUA — DA DO BANG GOLD THAT, AM (private sub 3998, 01/09/2026): TIE_LONG=1 doi 56/1012
# dap an ma diem GIU NGUYEN 0.3320 = 168/506, rong dung 0 cau. Ghep voi 0/19 tren dev:
# nhan ngan trung 0, nhan dai cung trung 0 => DO DAI NHAN KHONG MANG TIN HIEU o vung hoa,
# khong phai bi dung nguoc dau. DUNG THU LAI moi bien the tie-break theo do dai nhan.
# Mac dinh "0" = hanh vi cu, de ban 0.3320 van dung lai trung byte.
TIE_LONG=_os.environ.get("TIE_LONG","0")=="1"
def score_cands(rows, target, qdir=None):
    """Chấm mọi dòng theo target → [(ov,row)] đã sắp xếp giảm dần. Tách ra từ locate() để
    locate_alt() dùng lại đúng cùng cách chấm — KHÔNG đổi hành vi locate."""
    tt=toks(target); tn_s=strip_vn(target); cands=[]              # nhánh label: gom ứng viên có overlap>0
    for r in rows:
        if r["cur"] is None or not r["label"]: continue
        ls=strip_vn(r["label"]); lt=toks(r["label"])
        inter=len(tt&lt)
        if inter==0: continue
        # F1(precision,recall) thay vì recall-only: PHẠT token thừa của label → label dài
        # chứa đủ chữ không còn thắng oan (vd hỏi "cam kết cho thuê hoạt động" ra "Tiền chi khác cho hoạt động KD").
        prec=inter/max(1,len(lt)); rec=inter/max(1,len(tt))
        ov=2*prec*rec/(prec+rec)
        # Chứa nguyên cụm → tín hiệu mạnh. Tách HAI HƯỚNG chứa nhau (mặc định bằng nhau = hành vi cũ):
        #   CONT_IN  nhãn CHỨA TRỌN target  ("Vốn cổ phần đã phát hànhCổ phiếu phổ thông" ⊇ "Vốn cổ phần đã phát hành")
        #   CONT_OF  nhãn chỉ là KHÚC CẮT   ("Vốn cổ phần" ⊂ "Vốn cổ phần đã phát hành")
        # Hướng đầu khớp đúng thứ được hỏi rồi mô tả thêm; hướng sau đánh rơi định ngữ.
        if tn_s:
            if tn_s in ls: ov+=CONT_IN
            elif ls in tn_s: ov+=CONT_OF
        if CTX_W and r.get("ctx"):
            # NGỮ CẢNH CHA giải thích phần câu hỏi mà nhãn con không nói tới. Cộng theo RECALL
            # (không chia cho độ dài ctx) để nhãn cha dài không bị phạt oan như khi ghép vào label.
            ov+=CTX_W*len((tt-lt)&toks(r["ctx"]))/max(1,len(tt))
        if lt<=tt or tt<=lt: ov+=0.001
        if qdir=="cuoi" and ("dau nam" in ls or "dau ky" in ls): ov-=0.5
        if qdir=="dau" and ("cuoi nam" in ls or "cuoi ky" in ls): ov-=0.5
        if ov>0: cands.append((ov,r))
    cands.sort(key=lambda x:(-x[0], -len(x[1]["label"]) if TIE_LONG else len(x[1]["label"])))
    return cands

def locate_alt(rows, target, qdir=None, exclude=(), rel=0.75, k=2):
    """Các bảng KHÁC cũng chứa dòng khớp target gần bằng dòng tốt nhất → list số dòng <table>.

    Một chỉ tiêu thường xuất hiện ở nhiều bảng (bảng chính + thuyết minh + bảng tổng hợp).
    locate() chỉ giữ dòng tốt nhất vì answer chỉ cần một giá trị, nhưng retrieval thì mọi
    bảng chứa chỉ tiêu đều có thể là gold. Ngưỡng rel để không vơ bảng chỉ khớp lỏng lẻo."""
    cands=score_cands(rows,target,qdir)
    if not cands: return []
    top=cands[0][0]; seen=set(exclude); out=[]
    for ov,r in cands:
        if ov < rel*top or len(out) >= k: break
        if r["tid"] in seen: continue
        seen.add(r["tid"]); out.append(r["line"])
    return out

GAP_THR=float(_os.environ.get("GAP_THR","0.05"))   # dưới ngưỡng này coi như HOÀ điểm từ vựng
def locate(rows, target, qdir=None, embedder=None, tabtoks=None, gap=None, rrank=None):
    """Tìm dòng khớp target → {maso,label,tid,cur,prev,ov} hoặc None.
    embedder(target, labels)->[cosine]: nếu có, rerank top-K ứng viên lexical bằng semantic (HYBRID).

    TIE-BREAK CẤP BẢNG khi hai ứng viên đầu gần như HOÀ điểm từ vựng — hai nguồn tín hiệu:
      rrank   = danh sách SỐ DÒNG <table> đã xếp bởi Qwen3-Reranker (rerank_cache.json), tốt nhất trước.
      tabtoks = P.table_tokens(text) → BM25. ĐÃ ĐO VÀ LOẠI, xem bên dưới. Đừng dùng lại.

    *** BM25 (tabtoks): ĐÃ ĐO VÀ LOẠI (13/08) — ĐỪNG BẬT ***
    Chia đôi dev set: nửa chỉnh −7 câu, nửa nghiệm thu +3 → TRÁI DẤU, ròng −4. Lý do gốc đo được:
    trong vùng hoà (119/312 câu, hiện đúng 48.7%), BM25 xếp ĐÚNG bảng ở 58/85 ca = 68%, mà cách
    chọn hiện tại cũng đúng 58/85 = 68% — hai tín hiệu CHÍNH XÁC NGANG NHAU nhưng sai ở những ca
    KHÁC nhau, nên hoán đổi chỉ xáo lại tập câu đúng, thêm phương sai mà không thêm kỳ vọng.
    Cận trên của MỌI tín hiệu bảng chỉ 27 câu (34 ca trong vùng hoà không có ứng viên nào đúng).
    Kết luận ĐÚNG của phép đo đó là "BM25 cấp bảng không đủ mạnh", KHÔNG phải "tín hiệu bảng vô
    dụng" — BM25 chính là retriever yếu nhất trong bảng đo của BTC (Recall@10 47.4%, so với 80.8%
    khi có reranker). Vì vậy `rrank` được thử riêng: cùng cơ chế, tín hiệu mạnh hơn hẳn.

    *** rrank (Qwen3-Reranker): ĐÃ ĐO VÀ LOẠI (14/08) — ĐỪNG BẬT ***
    Chia đôi dev set (eval_rr.py): TRÁI DẤU ở CẢ SÁU ngưỡng — nửa chỉnh âm (−1..−6), nửa nghiệm thu
    dương (+2..+6). Nửa A nói ngưỡng tốt nhất là 0 (tức tắt hẳn). Ròng tốt nhất +3/312, trong nhiễu.
    Chẩn đoán (diag_rr.py, GAP=0.05) cho biết vì sao, và con số này mới là thứ đáng nhớ:
      vùng hoà 119 câu → 34 câu (29%) KHÔNG ứng viên nào mang giá trị đúng ⇒ trần mọi tín hiệu bảng
      là 85 câu; trong 85 câu đó cách hiện tại đã đúng 58 (68%), reranker đúng 60 (71%), hai cách
      chọn khác nhau ở 41 ca mà chỉ ròng +2 ⇒ gần như tung đồng xu trên phần bất đồng.
    Vì sao lợi thế Recall@10 47.4%→80.8% KHÔNG chuyển hoá: Recall@10 chỉ hỏi "bảng gold có nằm đâu
    đó trong top-10 không" — mục tiêu thô và rộng lượng. Ở đây cần xếp ĐÚNG BẢNG LÊN ĐẦU giữa các
    bảng cùng một report mang nhãn gần như trùng chữ. Reranker chưa từng được đo cho việc đó. Cùng
    một lớp thất bại với dense retrieval (xem USE_DENSE: "dense kém phân biệt bảng TRONG 1 report").
    ⇒ Nghẽn ở vùng hoà KHÔNG phải chọn bảng mà là 29% số ca ứng viên đúng không hề lọt vào nhóm hoà.

    Cả hai mặc định None = hành vi y hệt trước khi có tham số; không call site nào truyền vào."""
    tn=strip_vn(target)
    code=next((c for k,c in MAIN_CODE.items() if k in tn),None)   # nhánh main: Mã số (deterministic, không cần embed)
    if code and not any(x in tn for x in ["chua phan phoi","thang du"]):
        cand=[r for r in rows if r["ma"]==code and r["cur"] is not None]
        if cand:
            r=cand[0]
            return {"maso":code,"label":r["label"],"tid":r["tid"],"cur":r["cur"],"prev":r["prev"],"ov":None,
                    "hdrs":r.get("hdrs") or [],"vals":r.get("vals") or []}
    cands=score_cands(rows,target,qdir)
    if not cands: return None
    best,best_ov=cands[0][1],cands[0][0]
    g=GAP_THR if gap is None else gap
    if (rrank or tabtoks) and len(cands)>1 and cands[0][0]-cands[1][0]<g:   # THẾ HOÀ → phân xử bằng bảng
        tied=[(ov,r) for ov,r in cands if cands[0][0]-ov<g]
        if len(tied)>1:
            if rrank:
                # Thứ hạng reranker; bảng không nằm trong danh sách xếp CUỐI (không phải hạng 0).
                pos={ln:i for i,ln in enumerate(rrank)}; far=len(rrank)+1
                best_ov,best=max(tied,key=lambda x:(-pos.get(x[1]["line"],far),x[0]))
            else:
                bm=bm25_scores(toks_seg(target),[t["toks"] for t in tabtoks])
                tsc={t["tid"]:s for t,s in zip(tabtoks,bm)}
                # ưu tiên bảng liên quan hơn; hoà bảng thì giữ nguyên thứ tự cũ (điểm từ vựng cao hơn)
                best_ov,best=max(tied,key=lambda x:(tsc.get(x[1]["tid"],0.0),x[0]))
    if embedder and len(cands)>1:                                # HYBRID: rerank top-K lexical bằng embedding
        top=cands[:15]
        try:
            sims=embedder(target,[r["label"] for _,r in top])
            resc=sorted(zip(top,sims), key=lambda x:-(0.4*x[0][0]+0.6*x[1]))
            best,best_ov=resc[0][0][1],resc[0][0][0]
        except Exception:
            pass
    return {"maso":"","label":best["label"],"tid":best["tid"],"cur":best["cur"],"prev":best["prev"],"ov":round(best_ov,2),
            "hdrs":best.get("hdrs") or [],"vals":best.get("vals") or []}

MOVE=("tang","giam","phat sinh","trong ky","trong nam","du phong","loai tru","dieu chinh")
def pick_col(qt, loc, qdir):
    """Bảng >2 cột số → chọn cột theo HEADER thay vì theo vị trí. Trả (index, giá trị) hoặc None.

    Vì sao: `cur` mặc định lấy số tiền ĐẦU TIÊN của dòng. Đúng với bảng 2 cột (kỳ này/kỳ trước —
    90.4% số ca) nhưng SAI với bảng dạng 'Số đầu năm | Tăng | Giảm | Số cuối năm' hay bảng chia theo
    mảng/khu vực có cột 'Tổng cộng'. Ca thật id226: hỏi số thuế phải nộp ĐẾN 31/12, pipeline nộp cột
    'Số đầu năm'. Dev set KHÔNG bắt được lớp lỗi này vì trọng tài được cho ăn chính giá trị sai đó.
    """
    hdrs=loc.get("hdrs") or []; vals=loc.get("vals") or []
    if len(hdrs)<3 or len(hdrs)!=len(vals): return None
    qs=strip_vn(qt); qt_toks=toks(qt); yrs=set(re.findall(r"\b20\d{2}\b",qt))
    best=(None,-9.0); s0=0.0
    for i,h in enumerate(hdrs):
        hs=strip_vn(h); s=0.0
        if yrs & set(re.findall(r"\b20\d{2}\b",h)): s+=2.0        # header ghi đúng năm câu hỏi
        if qdir=="cuoi" and ("cuoi nam" in hs or "cuoi ky" in hs): s+=2.0
        if qdir=="dau"  and ("dau nam"  in hs or "dau ky"  in hs): s+=2.0
        if qdir=="cuoi" and ("dau nam"  in hs or "dau ky"  in hs): s-=2.0
        if qdir=="dau"  and ("cuoi nam" in hs or "cuoi ky" in hs): s-=2.0
        # 'Tổng cộng' là cột đúng mặc định của bảng chia mảng/khu vực/ngoại tệ, kể cả khi câu hỏi
        # không có chữ 'tổng' — trừ khi câu hỏi nêu đích danh một mảng (lúc đó khớp từ sẽ thắng).
        if "tong cong" in hs or hs=="tong": s+=1.5 if ("tong" in qs or "toan" in qs) else 1.2
        if any(m in hs for m in MOVE) and not any(m in qs for m in MOVE): s-=2.0   # cột biến động
        # Khớp từ với header phải ĐỦ NẶNG để cột nêu đích danh trong câu hỏi thắng cột 'Tổng cộng'
        # (ca thật id765: hỏi doanh thu 'hoạt động ngân hàng' mà nhảy sang Tổng cộng → sai).
        s+=1.5*len(qt_toks & toks(h))/max(1,len(toks(h)))
        if s>best[1]: best=(i,s);
        if i==0: s0=s
    i=best[0]
    # BIÊN AN TOÀN: chỉ đổi khi hơn hẳn cột mặc định. Hai cột gần điểm nhau (id115: 'Năm 2018 trình
    # bày lại' vs 'Năm 2018 số đã công bố') thì giữ nguyên còn hơn chọn bừa.
    if i is None or i==0 or best[1]-s0<0.75: return None
    return (i, vals[i])

def pick_value(loc, qdir):
    """'đầu năm' cho item balance (label không tự chứa từ hướng) → cột prev."""
    has_dir=any(x in strip_vn(loc["label"]) for x in ["dau nam","cuoi nam","dau ky","cuoi ky"])
    return loc["prev"] if (qdir=="dau" and not has_dir and loc["prev"] is not None) else loc["cur"]

def extract(q):
    tk=resolve(q); yrs=sorted(set(re.findall(r"\b(20\d{2})\b",q)))
    if not tk or not yrs: return {"err":"no tk/year","tk":tk}
    ry=yrs[-1]; fr=find_report(tk,ry,doctype(q))
    if not fr: return {"err":"no report","tk":tk}
    rep,docname=fr
    rows,uf,src=ingest(rep.read_text(encoding="utf-8",errors="replace"))
    qdir=qdir_of(q)
    loc=locate(rows,target_label(q),qdir)
    if not loc:
        return {"tk":tk,"docname":docname,"uf":int(uf),"usrc":src,"mode":"miss","val":None}
    return {"tk":tk,"docname":docname,"uf":int(uf),"usrc":src,
            "mode":("ma_so" if loc["maso"] else "label"),"val":pick_value(loc,qdir),
            "tid":loc["tid"],"ov":loc["ov"],"ev":loc["label"][:40],
            "maso":loc["maso"],"flabel":loc["label"],"ry":int(yrs[-1])}

def rank_tables(q):
    """Xếp hạng bảng theo điểm table-level (max overlap dòng) → phục vụ đo F2 retrieval. Trả [(tid,score)]."""
    tk=resolve(q); yrs=sorted(set(re.findall(r"\b(20\d{2})\b",q)))
    if not tk or not yrs: return [],None
    fr=find_report(tk,yrs[-1],doctype(q))
    if not fr: return [],None
    rep,docname=fr; rows,uf,src=ingest(rep.read_text(encoding="utf-8",errors="replace"))
    target=target_label(q); tn=strip_vn(target); tt=toks(target)
    code=next((c for k,c in MAIN_CODE.items() if k in tn),None) if not any(x in tn for x in ["chua phan phoi","thang du"]) else None
    per=defaultdict(float)
    for r in rows:
        if r["cur"] is None: continue
        sc=len(tt&toks(r["label"]))/max(1,len(tt)) if r["label"] else 0
        if code and r["ma"]==code: sc=max(sc,1.5)   # bảng chứa Mã số đúng → boost
        per[r["tid"]]=max(per[r["tid"]],sc)
    ranked=sorted(per.items(),key=lambda x:-x[1])
    return ranked,docname

# chọn 30 câu direct-lookup, spread
Q=[json.loads(l) for l in (ROOT/"questions/questions.jsonl").open(encoding="utf-8") if l.strip()]
def n_tickers(q): return len(set(m for m in re.findall(r"\b([A-Z0-9]{2,4})\b",q) if m in tickers))
AGG=["cao nhất","thấp nhất","lớn nhất","nhỏ nhất","tăng nhiều nhất","giảm nhiều nhất","trung bình","nhiều nhất","ít nhất"]
COND=["trong các năm","trong những năm","trong giai đoạn","các công ty có","doanh nghiệp có mức","vào năm mà","năm ngay sau","xét các","trong ngành"]
RATIO=["tỷ suất","biên lợi nhuận","roe","roa","tỷ lệ nợ","vòng quay","hệ số","tốc độ tăng trưởng",
       "tỷ trọng","tỷ lệ"," trên ","chiếm bao nhiêu"]  # ratio/tính toán, KHÔNG direct
COMPUTE=["chênh lệch","bình quân","trung bình mỗi","hiệu số","phần trăm"]
OWN=["tỷ lệ sở hữu","tỷ lệ biểu quyết","quyền biểu quyết","tỷ lệ lợi ích"]
GROWTH=["tăng trưởng","so với năm","tăng bao nhiêu","giảm bao nhiêu"]
def is_direct(q):
    low=q.lower(); yrs=re.findall(r"\b20\d{2}\b",q)
    if any(k in low for k in OWN+COND+AGG+RATIO+GROWTH+COMPUTE): return False
    if n_tickers(q)>=2 or len(set(yrs))>=2: return False
    return True
direct=[q for q in Q if is_direct(q["question"])]
sample=direct[::len(direct)//30][:30]

if __name__=="__main__":
    print(f"Direct-lookup: {len(direct)} câu · chọn {len(sample)} để verify\n")
    for q in sample:
        a=extract(q["question"])
        print(f"id{q['id']}: {q['question'][:82]}")
        if "err" in a: print(f"   ERR {a['err']} tk={a.get('tk')}\n"); continue
        v=a["val"]; vs=f"{int(v):,}" if v is not None else "—"
        extra=f" ov={a.get('ov')}" if a["mode"]=="label" else ""
        print(f"   {a['tk']} · {a['mode']}{extra} · unit={a['usrc']}(×{a['uf']}) · ANS={vs}")
        print(f"   tid={a['tid']} ev='{a['ev']}' · doc={a['docname'][:38]}\n")
