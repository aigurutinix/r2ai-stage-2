"""Create a truthful post-review appendix without modifying any submission."""
from __future__ import annotations
import hashlib
import io
import json
import math
import re
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from road2ai_vifinqa.corpus import Corpus
from road2ai_vifinqa.html_tables import parse_html_table
from road2ai_vifinqa.text import parse_vn_number
from synera_recompute_eight import IDS, audit_package

OUT = ROOT / 'output' / 'synera_appendix_8_samples'
PDF = ROOT / 'output' / 'pdf' / 'synera_phu_luc_8_mau.pdf'
BASE = ROOT / 'submission_vn79.zip'
def sha(b):
    return hashlib.sha256(b).hexdigest()
def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evidence_original').mkdir(exist_ok=True)
    (OUT / 'source_tables').mkdir(exist_ok=True)
    (OUT / 'source_code_snapshot').mkdir(exist_ok=True)
    with zipfile.ZipFile(BASE) as archive:
        selected = [r for r in json.loads(archive.read('submission.json')) if r['id'] in IDS]
        for row in selected:
            path = row['evidence'][0]['csv_path']
            (OUT / 'evidence_original' / Path(path).name).write_bytes(archive.read(path))
    write_json(OUT / 'submitted_samples.json', selected)
    shutil.copy2(ROOT / 'tools/synera_recompute_eight.py', OUT / 'synera_recompute_eight.py')
    for name in ('pipeline.py', 'hard_solver.py', 'panel.py', 'html_tables.py', 'text.py'):
        shutil.copy2(ROOT / 'src/road2ai_vifinqa' / name, OUT / 'source_code_snapshot' / name)
    reports = audit_package(OUT)
    source_rows = []
    docs, tables = {}, {}
    with Corpus() as corpus:
        for report in reports:
            qid = report['id']
            frame = pd.read_csv(OUT / 'evidence_original' / f'q{qid:04d}_df1.csv')
            for s in frame.to_dict('records'):
                doc, tid = s['doc_id'], int(s['table_id'])
                if doc not in docs:
                    path = Path(corpus.document(doc).source_path)
                    if not path.is_absolute():
                        path = ROOT / path
                    binary = path.read_bytes()
                    fragments = list(re.finditer(br'<table\b[^>]*>.*?</table>', binary, re.I | re.S))
                    docs[doc] = (binary, fragments)
                binary, fragments = docs[doc]
                key = (doc, tid)
                if key not in tables:
                    match = fragments[tid - 1]
                    fragment = match.group().decode('utf-8')
                    grid = parse_html_table(fragment)
                    line = binary.count(b'\n', 0, match.start()) + 1
                    context = corpus.table(doc, tid).context
                    name = f'{doc}__table_{tid}.json'
                    write_json(OUT / 'source_tables' / name, {
                        'doc_id': doc, 'document_sha256': sha(binary), 'table_id_internal': tid,
                        'physical_line_one_based': line, 'html_sha256': sha(match.group()),
                        'html': fragment, 'expanded_rows': grid, 'context_from_local_index': context})
                    tables[key] = grid, line, name
                grid, line, name = tables[key]
                raw = grid[int(s['row_idx'])][int(s['col_idx'])]
                number = parse_vn_number(raw)
                stored_raw = str(s['raw'])
                factor = None if number is None or number == 0 else float(s['value']) / number
                source_rows.append({'id': qid, 'source_id': s['source_id'], 'doc_id': doc,
                    'table_id_internal': tid, 'physical_line_one_based': line,
                    'row_idx_zero_based': int(s['row_idx']), 'col_idx_zero_based': int(s['col_idx']),
                    'ticker': s['ticker'], 'year': int(s['year']), 'metric': s['raw_column'],
                    'evidence_raw': stored_raw, 'source_raw': raw, 'raw_exact_match': raw == stored_raw,
                    'source_number': number, 'normalized_value': float(s['value']),
                    'value_over_raw_number': factor, 'source_table_file': 'source_tables/' + name})
    for r in reports:
        src = [s for s in source_rows if s['id'] == r['id']]
        r['source_cells'] = len(src)
        r['source_raw_matches'] = sum(s['raw_exact_match'] for s in src)
        r['scale_factors_observed'] = dict(Counter(str(s['value_over_raw_number']) for s in src))
    write_json(OUT / 'source_cell_audit.json', source_rows)
    write_json(OUT / 'recompute_results.json', reports)
    write_json(OUT / 'provenance.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'classification': 'Post-review reconstruction; not original submitted query; not official gold answers',
        'local_baseline': BASE.name, 'local_baseline_sha256': sha(BASE.read_bytes()),
        'organizer_pdf_sha256': sha(Path('C:/Users/DELL/Downloads/synera_evidence_1.pdf').read_bytes()),
        'organizer_sample_ids': list(IDS),
        'warning': 'Local VN79 is the comparison baseline. PDF samples match query/shape/results up to floating precision. Final uploaded full archive identity has not been confirmed.',
        'raw_matches': sum(s['raw_exact_match'] for s in source_rows), 'source_cells': len(source_rows),
        'source_documents': len(docs), 'source_tables': len(tables)})
    return reports, source_rows


DESCRIPTIONS = {
363: ('Chọn năm D/E cao nhất và tính khả năng thanh toán lãi vay',
      'Với từng năm 2016-2020, D/E = nợ phải trả / vốn chủ sở hữu. Chọn năm có D/E lớn nhất. Tại năm đó, tính (lợi nhuận trước thuế + chi phí lãi vay) / chi phí lãi vay.'),
364: ('Lọc CFO dương, chọn tăng trưởng doanh thu cao nhất',
      'Giữ doanh nghiệp có CFO > 0 ở cả 2020 và 2021. Tăng trưởng = (doanh thu 2021 / doanh thu 2020 - 1) × 100. Chọn mức lớn nhất; tỷ số dồn tích = (LNST 2021 - CFO 2021) / [(tài sản 2020 + tài sản 2021) / 2] × 100. Đây là công thức đang được mã nguồn sử dụng.'),
367: ('Lọc CFO và doanh thu giảm, tính chênh lệch biên bình quân',
      'Giữ doanh nghiệp có CFO dương ở cả hai năm 2024, 2025 và doanh thu 2025 < 2024. Mỗi doanh nghiệp: chênh lệch biên = (lợi nhuận gộp - LNST) / doanh thu thuần × 100. Lấy trung bình số học trên các doanh nghiệp còn lại; đơn vị điểm phần trăm.'),
368: ('Lọc dưới trung vị thanh toán nhanh, tính biên ròng bình quân',
      'Năm 2022: thanh toán nhanh = (tài sản ngắn hạn - hàng tồn kho) / nợ ngắn hạn. Tính trung vị của cả bốn công ty, giữ các tỷ số nhỏ hơn trung vị. Tính LNST / doanh thu thuần × 100 cho từng công ty được giữ, rồi lấy trung bình số học.'),
369: ('Lọc trung vị 2022, chọn tăng biên gộp, tính hệ số 2023',
      'Lọc công ty có thanh toán nhanh 2022 dưới trung vị nhóm. Tính biên gộp = lợi nhuận gộp / doanh thu thuần × 100; chọn công ty có biên gộp 2023 - 2022 lớn nhất trong tập đã lọc. Trả (LNTT 2023 + lãi vay 2023) / lãi vay 2023.'),
370: ('Lọc CFO ba năm, chọn CAGR, tính biên ròng 2024',
      'Giữ công ty có CFO > 0 ở cả 2022, 2023 và 2024. CAGR hai khoảng năm = [(doanh thu 2024 / doanh thu 2022)^(1/2) - 1] × 100. Chọn CAGR lớn nhất; trả LNST 2024 / doanh thu 2024 × 100.'),
371: ('Lọc CFO dương, chọn biên gộp, tính khả năng trả lãi',
      'Trong năm 2024, giữ các công ty có CFO > 0. Tính lợi nhuận gộp / doanh thu thuần × 100, chọn mức lớn nhất. Trả (LNTT + chi phí lãi vay) / chi phí lãi vay của công ty đó.'),
372: ('Chọn năm thanh toán nhanh thấp nhất, lấy năm kế tiếp',
      'Tính (tài sản ngắn hạn - tồn kho) / nợ ngắn hạn của VRE trong 2021-2024. Chọn năm nhỏ nhất; ở năm ngay sau đó, trả CFO / nợ ngắn hạn. Không chọn giá trị CFO trong chính năm đạt cực tiểu.'),
}

def pdf_report(reports):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    pdfmetrics.registerFont(TTFont('ArialVN', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('ArialVNBold', 'C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFontFamily('ArialVN', normal='ArialVN', bold='ArialVNBold')
    styles = getSampleStyleSheet()
    for s in styles.byName.values():
        s.fontName = 'ArialVN'
    styles.add(ParagraphStyle(name='TitleVN', fontName='ArialVNBold', fontSize=20, leading=25, textColor=colors.HexColor('#17384e'), spaceAfter=14))
    styles.add(ParagraphStyle(name='BodyVN', fontName='ArialVN', fontSize=10, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallVN', fontName='ArialVN', fontSize=8, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name='CellVN', fontName='ArialVN', fontSize=7.1, leading=9))
    story = []
    def para(text, style='BodyVN'):
        return Paragraph(text, styles[style])
    def table(headers, rows):
        data = [[para(escape(str(x)), 'CellVN') for x in headers]]
        for row in rows:
            vals=[]
            for x in row:
                if isinstance(x, bool): x='Có' if x else 'Không'
                elif isinstance(x, (int,float)) and not isinstance(x,bool): x=f'{x:.9g}'
                vals.append(para(escape(str(x)), 'CellVN'))
            data.append(vals)
        t = Table(data, colWidths=[499 / len(headers)] * len(headers), repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2edf2')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#bdcbd1')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        return t
    story += [para('PHỤ LỤC ĐỐI CHIẾU 8 MẪU', 'TitleVN'), para('Đội synera | Hồ sơ kỹ thuật bổ sung sau rà soát | 05/09/2026'),
        para('<b>Mục đích.</b> Làm rõ dữ liệu và phép tính của tám mẫu BTC nêu trong synera_evidence_1.pdf. Mã tái tính trong hồ sơ này được viết sau khi nhận phản ánh. Nó không phải pandas_query nguyên bản và không chứng minh bài nộp cũ đã đáp ứng yêu cầu.'),
        para('<b>Cơ sở đối chiếu.</b> Evidence được sao chép nguyên byte từ bản VN79 đang lưu tại máy. Tám mẫu khớp PDF của BTC về query, kích thước và kết quả trong sai số biểu diễn số thực. Chưa xác nhận danh tính toàn bộ ZIP đã tải lên cuối cùng; kết luận ở đây giới hạn trong các mẫu được đối chiếu.'),
        para('<b>Query nguyên bản của cả tám mẫu:</b>'),
        para("float(df1.loc[df1['source_id'] == 's1',<br/>'computed_answer'].iloc[0])", 'SmallVN'),
        para('Query này đọc kết quả đã ghi sẵn. Trong pipeline.py, hàm _hard_source_rows ghi result.answer vào computed_answer; adapt_hard xuất query đọc cột đó. Việc có dữ liệu nguồn trong CSV không thay thế yêu cầu query phải thực hiện phép tính.'),
        table(['ID', 'Kết quả đã nộp', 'Tái tính', 'Khớp'], [[r['id'],r['submitted'],r['recomputed'],r['matches']] for r in reports]), Spacer(1,12),
        para('<b>Cách kiểm chứng.</b> synera_recompute_eight.py chỉ dùng ticker, year, raw_column và value để tính. Cột computed_answer bị loại khỏi đầu vào phép tính. Sau khi tính xong mới đọc answer để so sánh. source_cell_audit.json ghi tọa độ, chuỗi nguồn, hệ số chuẩn hóa quan sát được; source_tables/ chứa trích đoạn HTML của bảng và SHA-256 báo cáo gốc.', 'SmallVN'),
        para('<b>Giới hạn.</b> “Khớp” nghĩa là tái tạo số đã nộp, không phải khớp đáp án bí mật của BTC. Đối chiếu tọa độ xác nhận chuỗi nguồn; chưa phải thẩm định độc lập toàn bộ cách chọn chỉ tiêu, dấu, đơn vị hoặc logic tài chính.', 'SmallVN')]
    for r in reports:
        qid=r['id']; title,method=DESCRIPTIONS[qid]
        story += [PageBreak(), para(f'MẪU {qid}', 'TitleVN'), para(escape(title)),
            para('<b>Câu hỏi:</b> '+escape(r['question'])),
            para(f"<b>Evidence nguyên bản:</b> data/q{qid:04d}_df1.csv; {r['shape'][0]} hàng × {r['shape'][1]} cột. Query nguyên bản là câu lệnh đọc computed_answer tại trang đầu."),
            para('<b>Phép tái tính bổ sung:</b> '+escape(method)),
            para('<b>Bảng trung gian tái tính</b>')]
        trace=r['trace']
        # Large CFO magnitudes are not needed in the compact decision table.
        if qid==364:
            trace=[{k:v for k,v in x.items() if k not in ('CFO_2020','CFO_2021')} for x in trace]
        story += [table(list(trace[0]), [list(x.values()) for x in trace]), Spacer(1,12),
            para('<b>Lựa chọn:</b> '+escape(r['selection'])),
            para(f"<b>Kết quả:</b> {r['recomputed']:.17g}. Đã nộp: {r['submitted']:.17g}. Sai lệch tuyệt đối: {r['absolute_error']:.3g}."),
            para(f"<b>Đối chiếu nguồn:</b> {r['source_raw_matches']}/{r['source_cells']} chuỗi raw trong evidence khớp ô của bảng HTML được trích lại từ TXT nguồn. Xem các bản ghi id={qid} trong source_cell_audit.json để kiểm tra từng số liệu."),
            para('<b>Kiểm tra phụ thuộc dữ liệu:</b> xóa computed_answer hoặc thay toàn bộ cột bằng số giả vẫn cho cùng kết quả tái tính. '+
                 f"Tăng 10% các giá trị {r['sensitivity_metric']} trong một bản sao bộ nhớ làm kết quả đổi thành {r['sensitivity_result']:.15g}. Evidence nguyên bản không bị sửa."),
            para(f"<b>Mã tái thực thi:</b> nhánh qid == {qid} trong synera_recompute_eight.py. Công thức và phép lọc cũng được đối chiếu với nhánh tương ứng trong bản sao hard_solver.py hiện có; các tỷ số trong panel.py. Bản sao mã nguồn hiện tại không tự chứng minh thời điểm triển khai trước hạn nộp.", 'SmallVN'),
            para('<b>Giới hạn của mẫu:</b> kết quả này chứng minh khả năng tái tính từ các giá trị evidence, không hợp thức hóa query cũ. Các giá trị value đã qua chuẩn hóa; hệ số value/raw và các trường hợp đổi dấu được công khai trong bảng kiểm toán. Không coi việc truy vấn một cột đổi tên là biện pháp khắc phục.', 'SmallVN')]
    PDF.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont('ArialVN',8); canvas.setFillColor(colors.HexColor('#536b79'))
        canvas.drawString(48,25,'synera | Phụ lục tái tính sau rà soát | Không phải query đã nộp')
        canvas.drawRightString(A4[0]-48,25,str(doc.page))
    SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    reports, sources=prepare()
    pdf_report(reports)
    shutil.copy2(PDF, OUT / PDF.name)
    readme = '''PHỤ LỤC 8 MẪU - SYNERA

Hồ sơ được tạo sau phản ánh của BTC. Không phải bản submission mới.
Đọc synera_phu_luc_8_mau.pdf trước khi sử dụng.

Tái thực thi (Python 3, pandas):
    python synera_recompute_eight.py --package .

Tệp:
- evidence_original/: 8 CSV nguyên byte lấy từ ZIP VN79 tại máy.
- submitted_samples.json: nguyên bản 8 bản ghi tương ứng.
- synera_recompute_eight.py: mã tái tính bổ sung, không đọc computed_answer.
- recompute_results.json: kết quả, bảng trung gian và kiểm tra phụ thuộc dữ liệu.
- source_cell_audit.json: tọa độ tất cả ô và hệ số chuẩn hóa quan sát được.
- source_tables/: HTML nguồn, bảng mở rộng rowspan/colspan, vị trí dòng và hash.
- source_code_snapshot/: bản sao mã nguồn hiện tại, không phải chứng nhận thời điểm.
- provenance.json, manifest_sha256.json: phạm vi và dấu vân tay dữ liệu.

Chỉ số table_id là thứ tự bảng 1-based trong mã nội bộ; row_idx/col_idx
là tọa độ 0-based sau mở rộng ô gộp. physical_line_one_based là số dòng
thực trong TXT. Không tráo lẫn hai loại vị trí này.

Tái tạo được số đã nộp không chứng minh đúng đáp án BTC, tính hợp lệ query
cũ, hoặc tính đúng của mọi phép chuẩn hóa. Không sửa file gốc hay lịch sử.
'''
    (OUT / 'README.txt').write_text(readme,encoding='utf-8')
    write_json(OUT / 'manifest_sha256.json',{p.relative_to(OUT).as_posix():sha(p.read_bytes())
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest_sha256.json'})
    archive=ROOT / 'output' / 'synera_phu_luc_8_mau.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(OUT).as_posix())
    print(json.dumps({'pdf':str(PDF),'zip':str(archive),'source_cells':len(sources),
        'raw_matches':sum(x['raw_exact_match'] for x in sources),
        'results':[{k:r[k] for k in ('id','recomputed','matches','selection','source_raw_matches','source_cells')} for r in reports]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
