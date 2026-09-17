# Overview

## Bối cảnh bài toán

Bối cảnh bài toán là tra cứu chỉ số tài chính (doanh thu, lợi nhuận, ROE, ROA, tăng trưởng...) từ hàng trăm Báo cáo Tài chính (BCTC) dạng bảng tốn nhiều thời gian. Cuộc thi hướng tới xây dựng hệ thống **Financial Table Retrieval & Text-to-Pandas Query Generation** tự động hóa quy trình này trên BCTC doanh nghiệp niêm yết tại Việt Nam.

**Truy hồi bảng dữ liệu (Table Retrieval)** là nhiệm vụ cốt lõi đầu tiên, liên quan đến việc xác định bảng dữ liệu nào phù hợp nhất với một truy vấn cho trước. Nhiệm vụ có thể được hình thức hoá như sau: Cho một tập câu hỏi $Q = \{q_1, q_2, ..., q_n\}$ và một kho báo cáo tài chính $D = \{d_1, d_2, ..., d_n\}$ (mỗi báo cáo gồm nhiều bảng: Bảng cân đối kế toán, Báo cáo kết quả kinh doanh, Báo cáo lưu chuyển tiền tệ, thuyết minh), nhiệm vụ yêu cầu xác định một tập con bảng $D' \subset D$ trong đó mỗi bảng $d_i \in D'$ được coi là "liên quan" đến câu hỏi tương ứng $q$. Chúng tôi gọi một bảng dữ liệu là "Liên quan" nếu bảng đó chứa (một phần hoặc toàn bộ) số liệu cần thiết để tính ra câu trả lời.

**Sinh truy vấn Pandas (Text-to-Pandas)** dựa trên các bảng đã truy hồi, hệ thống cần sinh ra câu lệnh pandas thực thi được để tính toán và trả về đúng số liệu cho câu hỏi tài chính tương ứng. Mục tiêu của nhiệm vụ là xây dựng các hệ thống AI có khả năng không chỉ tìm đúng bảng dữ liệu căn cứ mà còn hiểu và chuyển hoá đúng logic tính toán tài chính thành code, đảm bảo kết quả có thể kiểm chứng và tái lập.

## Mục tiêu cuộc thi

Cần xây dựng hệ thống AI có khả năng:

### 1. Truy hồi dữ liệu chính xác
- Xác định đúng công ty, đúng năm, đúng bảng dữ liệu chứa số liệu cần thiết.
- Tìm kiếm và truy xuất chính xác vị trí bảng dữ liệu từ kho BCTC được cung cấp.
- Ưu tiên khả năng retrieval và grounding chính xác trên dữ liệu dạng bảng.

### 2. Hiểu truy vấn tài chính bằng tiếng Việt
- Hiểu ngôn ngữ tự nhiên tiếng Việt về các chỉ số và thuật ngữ tài chính.
- Xử lý được câu hỏi so sánh nhiều công ty, nhiều năm, hoặc chỉ số dẫn xuất (ROE, ROA, tăng trưởng...).

### 3. Sinh truy vấn pandas & tính toán chính xác
- Sinh câu lệnh pandas chạy được, đúng logic, đúng schema dữ liệu.
- Trả về đúng số liệu, đúng đơn vị, đúng kỳ báo cáo được hỏi.

### 4. Dẫn nguồn minh bạch
- Trích dẫn công ty, năm, tên báo cáo, tên bảng và vị trí (trang/mục) chứa số liệu gốc.
- Hiển thị rõ nguồn tham chiếu để đảm bảo khả năng kiểm chứng thông tin.
- Hạn chế việc trả lời không có căn cứ dữ liệu.

### 5. Kiểm soát nội dung sai lệch
- Hạn chế việc AI sinh ra số liệu sai lệch (hallucination).
- Tránh bịa bảng dữ liệu hoặc nguồn tham chiếu không tồn tại.
- Tăng độ tin cậy của câu trả lời dựa trên dữ liệu được cung cấp.

## Quy định về dữ liệu bên ngoài và mô hình ngôn ngữ huấn luyện trước (PLMs)

Người tham gia được phép sử dụng dữ liệu từ các nguồn bên ngoài, tuy nhiên phải trích dẫn rõ ràng và cung cấp đầy đủ thông tin về nguồn gốc dữ liệu để Ban tổ chức có thể kiểm tra, xác minh khi cần thiết. 

Bạn có thể sử dụng các mô hình ngôn ngữ huấn luyện trước và các LLM có dữ liệu huấn luyện và/hoặc mô hình được công khai (ví dụ: Hugging Face hoặc các trang tương tự), nhưng **bạn không được sử dụng các LLM có mô hình đóng (ví dụ: GPT-4o, Gemini, ...)**. 

Ngoài ra, bạn chỉ được sử dụng các mô hình được phát hành **trước ngày 1 tháng 6 năm 2026 (giờ Việt Nam)** có kích thước **nhỏ hơn hoặc bằng ~14B áp dụng cho mỗi mô hình (không có giới hạn cho tổng pipeline)**. 
