<p align="center">
  <img src="./images/aiguru_icon.png" alt="AI Guru Logo" height="90" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./images/r2ai_icon.png" alt="R2AI Logo" height="90" />
</p>

<p align="center">
  🇻🇳 <b>Tiếng Việt</b> | 🇺🇸 <a href="./README.en.md">English</a>
</p>

---

# Road to AI (R2AI) – Stage 2

## 1. Giới thiệu cuộc thi

**Road to AI (R2AI)** là cuộc thi và cộng đồng về AI Engineering tại Việt Nam do **AI Guru** tổ chức, hướng tới việc xây dựng các sản phẩm AI có khả năng ứng dụng trong môi trường doanh nghiệp thực tế. **R2AI – Stage 2** mang đến thử thách **Text-to-Pandas – Trợ lý truy vấn & phân tích báo cáo tài chính**, tập trung vào bài toán truy hồi bảng dữ liệu và sinh truy vấn pandas để trả lời câu hỏi tài chính tiếng Việt.

### Bối cảnh bài toán

Nhà đầu tư, chuyên viên phân tích và doanh nghiệp tại Việt Nam thường mất nhiều thời gian tra cứu thủ công các chỉ số tài chính như doanh thu, lợi nhuận, ROE, ROA, tỷ lệ nợ/vốn chủ sở hữu hoặc tăng trưởng theo giai đoạn. Các số liệu này thường nằm rải rác trong nhiều báo cáo tài chính dạng bảng của các công ty niêm yết qua nhiều năm.

Trợ lý AI Text-to-Pandas được xây dựng nhằm hỗ trợ tự động hóa việc tra cứu, tổng hợp và tính toán các chỉ số tài chính từ dữ liệu báo cáo tài chính gốc. Trong bối cảnh các mô hình ngôn ngữ lớn như ChatGPT, DeepSeek và Qwen phát triển mạnh, nhu cầu xây dựng hệ thống có khả năng chuyển đổi câu hỏi ngôn ngữ tự nhiên thành truy vấn dữ liệu bảng ngày càng trở nên quan trọng, đặc biệt với dữ liệu tài chính tiếng Việt.

Cuộc thi hướng tới bài toán **Truy hồi Bảng dữ liệu & Sinh truy vấn Pandas trên Báo cáo tài chính doanh nghiệp niêm yết (Financial Table Retrieval & Text-to-Pandas Query Generation)**. Các hệ thống cần xác định đúng bảng dữ liệu liên quan, sinh câu lệnh pandas có thể thực thi, tính toán đúng kết quả và cung cấp căn cứ kiểm chứng rõ ràng.

### Truy hồi bảng dữ liệu (Table Retrieval)

Truy hồi bảng dữ liệu là nhiệm vụ xác định bảng nào trong kho báo cáo tài chính phù hợp nhất với câu hỏi cho trước. Với một tập câu hỏi $Q = \{q_1, q_2, ..., q_n\}$ và kho báo cáo tài chính $D = \{d_1, d_2, ..., d_n\}$, trong đó mỗi báo cáo gồm nhiều bảng như Bảng cân đối kế toán, Báo cáo kết quả kinh doanh, Báo cáo lưu chuyển tiền tệ và Thuyết minh báo cáo tài chính, hệ thống cần xác định tập bảng liên quan $D' \subset D$ chứa số liệu cần thiết để tính ra câu trả lời.

### Sinh truy vấn Pandas (Text-to-Pandas)

Dựa trên các bảng đã truy hồi, hệ thống cần sinh câu lệnh pandas có thể chạy lại trên dữ liệu đã chuẩn hóa để tính toán và trả về đúng số liệu cho câu hỏi tài chính tương ứng. Mục tiêu không chỉ là tìm đúng bảng dữ liệu mà còn hiểu đúng logic tính toán tài chính, đúng schema dữ liệu, đúng đơn vị và đúng kỳ báo cáo.

### Mục tiêu cuộc thi

Các đội thi cần xây dựng hệ thống AI có khả năng:

1. **Truy hồi dữ liệu chính xác**:
   * Xác định đúng công ty, đúng năm, đúng bảng dữ liệu chứa số liệu cần thiết.
   * Tìm kiếm và truy xuất chính xác vị trí bảng dữ liệu từ kho báo cáo tài chính.
   * Ưu tiên khả năng retrieval và grounding chính xác trên dữ liệu dạng bảng.
2. **Hiểu truy vấn tài chính bằng tiếng Việt**:
   * Hiểu ngôn ngữ tự nhiên tiếng Việt về chỉ số và thuật ngữ tài chính.
   * Xử lý được câu hỏi so sánh nhiều công ty, nhiều năm hoặc chỉ số dẫn xuất như ROE, ROA, tăng trưởng.
3. **Sinh truy vấn pandas & tính toán chính xác**:
   * Sinh câu lệnh pandas chạy được, đúng logic và đúng schema dữ liệu.
   * Trả về đúng số liệu, đúng đơn vị và đúng kỳ báo cáo được hỏi.
4. **Dẫn nguồn minh bạch**:
   * Trích dẫn công ty, năm, tên báo cáo, tên bảng và vị trí chứa số liệu gốc.
   * Hiển thị rõ nguồn tham chiếu để đảm bảo khả năng kiểm chứng thông tin.
   * Hạn chế việc trả lời không có căn cứ dữ liệu.
5. **Kiểm soát nội dung sai lệch**:
   * Hạn chế việc AI sinh ra số liệu sai lệch.
   * Tránh bịa bảng dữ liệu hoặc nguồn tham chiếu không tồn tại.
   * Tăng độ tin cậy của câu trả lời dựa trên dữ liệu được cung cấp.

---

## 2. Kết quả cuộc thi & Danh sách các đội

Dưới đây là bảng vinh danh kết quả các đội thi tại R2AI Stage 2 cùng liên kết đến mã nguồn và dữ liệu tương ứng đã được tổng hợp trong repository này:

| Hạng giải | Đội thi | Thư mục dự án |
| :--- | :--- | :--- |
| 🥇 **Giải Nhất** | LASTDANCE | [lastdance](./lastdance) |
| 🥈 **Giải Nhì** | ARCANE | [arcane](./arcane) |
| 🥉 **Giải Ba** | KINGPRO | [kingpro](./kingpro) |
| 🏅 **Giải Khuyến khích** | SYNERA | [synera](./synera) |
| 🏅 **Giải Khuyến khích** | VILAMIU | [vilamiu](./vilamiu) |
| 🏅 **Giải Khuyến khích** | Nguyễn Vũ Hoàng Long | [nguyenvuhoanglong](./nguyenvuhoanglong) |
| 🏅 **Giải Khuyến khích** | OVERFITTING | [overfitting](./overfitting) |
| 🏅 **Giải Khuyến khích** | IDIOT | [idiot](./idiot) |
| 🏅 **Giải Khuyến khích** | AISOLO | [aisolo](./aisolo) |

*Mỗi thư mục dự án của các đội đã được tổng hợp theo cấu trúc thống nhất, trong đó `src` chứa mã nguồn gốc của đội thi. Các dữ liệu, artifact hoặc tài liệu bổ sung chỉ được giữ khi có trong bộ nguồn được cung cấp.*

---

## 3. Thông tin liên hệ Ban Tổ chức

**AI Guru – Công ty CP Công nghệ và Truyền thông Dagoras**

* **Địa chỉ**: Tầng 8, số 80 Duy Tân, Cầu Giấy, Hà Nội
* **Đầu mối liên hệ**:
  * **Nguyễn Thị Minh Nguyệt**: Điện thoại: `0981544974` | Email: `nguyetntm@dagoras.io`
  * **Vũ Thị Thùy Linh**: Điện thoại: `0961891198` | Email: `linhvtt@dagoras.io`
* **Website**: [r2ai.aiguru.com.vn](https://r2ai.aiguru.com.vn)
