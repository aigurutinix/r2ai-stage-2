PHỤ LỤC 8 MẪU - SYNERA

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
