/**
 * Kiểm code Pandas app sinh ra CÓ CHẠY ĐƯỢC trên file gốc và ra ĐÚNG số app hiển thị.
 *
 * Khác `verify-pandas.mjs` (E2E, cần dev server + Playwright): file này gọi thẳng `computeAnswer`
 * nên chạy được trong mọi hoàn cảnh, đủ nhanh để dùng như một bộ kiểm hồi quy.
 *
 * Vì sao đây là rủi ro lớn nhất của đợt sửa này: `pandasCode` đọc FILE GỐC trên đĩa, còn app đọc
 * bảng ĐÃ chuẩn hoá trong bộ nhớ. Ba thứ vừa thêm đều làm hai đường lệch nhau nếu quên phản ánh:
 *   - bỏ qua dòng tiêu đề rác  → Pandas phải có `skiprows`
 *   - sửa Mã số lệch chuẩn     → mã app dùng KHÔNG còn khớp mã trong file ⇒ phải lọc theo tên
 *   - suy Mã số từ tên         → cột Mã số không tồn tại ⇒ phải lọc theo tên
 * Và đây chính là đoạn người đọc mã nguồn copy ra để kiểm chứng, nên lệch là lỗi thấy được.
 *
 * Chạy: node verify-pandas-gen.mjs <thư-mục-đã-compile>
 */
import { createRequire } from "node:module";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { execFileSync } from "node:child_process";
import * as XLSX from "xlsx";

const dir = process.argv[2];
if (!dir) {
  console.error("Thiếu tham số: node verify-pandas-gen.mjs <thư-mục-đã-compile>");
  process.exit(2);
}
const require_ = createRequire(import.meta.url);
const { normalize } = require_(path.resolve(dir, "normalize.js"));
const { computeAnswer } = require_(path.resolve(dir, "agent.js"));

// Python nao: .venv cua repo neu co, khong thi lay tren PATH. Ban truoc ghi cung duong dan
// tuyet doi cua mot may nen script chet ngay tren may khac.
const REPO_ = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, "$1")), "..");
const PY = [path.join(REPO_, ".venv", "Scripts", "python.exe"),
            path.join(REPO_, ".venv", "bin", "python")].find((p) => fs.existsSync(p))
         ?? (process.platform === "win32" ? "python" : "python3");
const fails = [];
let pass = 0;
const check = (ok, msg) => (ok ? pass++ : fails.push(msg));

function toMatrix(text) {
  const wb = XLSX.read(text, { type: "string" });
  const name = wb.SheetNames[0];
  const aoa = XLSX.utils.sheet_to_json(wb.Sheets[name], {
    header: 1,
    raw: false,
    defval: "",
    blankrows: false,
  });
  return { matrix: aoa.map((r) => (r ?? []).map((c) => (c == null ? "" : String(c)))), sheet: name };
}

/**
 * Dựng file CSV tạm, chạy code Pandas app sinh ra, so kết quả với giá trị app tính.
 * Chạy trong thư mục tạm vì code sinh ra dùng TÊN FILE TRẦN.
 */
function runCase(note, csvText, fileName, plan) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pdgen-"));
  try {
    fs.writeFileSync(path.join(tmp, fileName), csvText, "utf8");
    const { matrix, sheet } = toMatrix(csvText);
    const tidy = normalize(matrix, sheet);
    tidy.fileName = fileName;
    const r = computeAnswer(tidy, plan, "test");
    if (!r.ok) {
      fails.push(`[${note}] app không tính được: ${r.error}`);
      return;
    }
    fs.writeFileSync(path.join(tmp, "run.py"), `${r.pandasCode}\nprint(repr(float(result)))\n`, "utf8");
    let out;
    try {
      out = execFileSync(PY, ["run.py"], { cwd: tmp, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
    } catch (e) {
      fails.push(`[${note}] code Pandas KHÔNG CHẠY ĐƯỢC:\n${String(e.stderr || e).split("\n").slice(-6).join("\n")}`);
      return;
    }
    const got = Number(out);
    const want = Number(r.value);
    const ok = Number.isFinite(got) && Math.abs(got - want) <= Math.max(1e-6, Math.abs(want) * 1e-9);
    check(ok, `[${note}] Pandas ra ${got}, app hiển thị ${want} — HAI BÊN LỆCH NHAU`);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

const direct = (metric, year) => ({ intent: "direct_retrieval", metric, years: [year], tickers: [] });

// 1 — File chuẩn: đường cũ, phải không hồi quy.
runCase(
  "file chuẩn (lọc theo Mã số)",
  "Ma_So,Chi_Tieu,Year_2024\n270,Tổng tài sản,1000000\n100,Tài sản ngắn hạn,400000\n",
  "chuan.csv",
  direct("total_assets", "2024"),
);

// 2 — Có dòng tiêu đề rác: Pandas phải `skiprows`, nếu không sẽ đọc sai toàn bộ tên cột.
runCase(
  "có 2 dòng tiêu đề rác (skiprows)",
  "BAO CAO TAI CHINH,,\nDon vi: dong,,\nMa_So,Chi_Tieu,Year_2024\n270,Tổng tài sản,1000000\n",
  "corac.csv",
  direct("total_assets", "2024"),
);

// 3 — Mã lệch chuẩn TT200: app dùng 220 nhưng file ghi 210 ⇒ phải lọc theo TÊN.
runCase(
  "Mã số lệch chuẩn (lọc theo tên)",
  "Ma_So,Chi_Tieu,Year_2024\n210,Tài sản cố định,500000\n270,Tổng tài sản,1000000\n",
  "lech.csv",
  direct("fixed_assets", "2024"),
);

// 4 — Không có cột Mã số: cột đó không tồn tại trong file ⇒ phải lọc theo TÊN.
runCase(
  "không có cột Mã số (lọc theo tên)",
  "Chi_Tieu,Year_2024\nTổng tài sản,1000000\nHàng tồn kho,250000\n",
  "khongma.csv",
  direct("inventory", "2024"),
);

// 5 — Có cột Ticker + giá trị bẩn: đường phổ biến nhất khi demo.
runCase(
  "có Ticker + ô bẩn",
  'Ticker,Ma_So,Chi_Tieu,Year_2024\nC1,270,Tổng tài sản,"1,000,000 VND"\nC2,270,Tổng tài sản,"2,000,000 VND"\n',
  "ticker.csv",
  { intent: "direct_retrieval", metric: "total_assets", years: ["2024"], tickers: ["C1"] },
);

// 6 — Dấu trừ Unicode: phải ra ÂM ở cả hai bên (lỗi sai dấu vừa vá).
runCase(
  "dấu trừ Unicode (âm ở cả hai bên)",
  "Ma_So,Chi_Tieu,Year_2024\n60,Lợi nhuận sau thuế,\u2212250000\n",
  "amunicode.csv",
  direct("net_income", "2024"),
);

// 7 — Sheet khai đơn vị chung: app quy đổi về VND, Pandas đọc file gốc nên phải nhân lại.
runCase(
  "đơn vị chung của sheet (Pandas phải nhân lại)",
  "Đơn vị tính: triệu đồng,,\nMa_So,Chi_Tieu,Year_2024\n270,Tổng tài sản,1000000\n",
  "donvi.csv",
  direct("total_assets", "2024"),
);

// 8 — Đơn vị chung + ô TỰ KHAI đơn vị: ô tự khai KHÔNG được nhân hệ số sheet (nếu không là nhân đôi).
runCase(
  "ô tự khai đơn vị trong sheet có đơn vị chung",
  'Đơn vị tính: triệu đồng,,\nMa_So,Chi_Tieu,Year_2024\n110,Tiền và các khoản tương đương tiền,"80 tỷ"\n',
  "tukhai.csv",
  direct("cash", "2024"),
);

// 9 — File mẫu "CSV lạ" thật: gộp cả bốn lỗi dàn dựng, chạy trên chính file sẽ dùng khi demo.
{
  const f = "public/samples/01_messy_financial_statements.csv";
  if (fs.existsSync(f)) {
    const text = fs.readFileSync(f, "utf8");
    const mk = (metric) => ({ intent: "direct_retrieval", metric, years: ["2024"], tickers: ["CORP_MESSY"] });
    const name = "01_messy_financial_statements.csv";
    runCase("mẫu CSV lạ — Mã số lệch chuẩn (lọc theo tên)", text, name, mk("fixed_assets"));
    runCase("mẫu CSV lạ — Mã 01 giữ số 0 đầu", text, name, mk("gross_revenue"));
    runCase("mẫu CSV lạ — ô tự khai 'tỷ' trong sheet đơn vị triệu", text, name, mk("cash"));
    runCase("mẫu CSV lạ — dấu trừ Unicode", text, name, mk("revenue_deductions"));
  }
}

const tong = pass + fails.length;
console.log(`pandas sinh ra: ${pass}/${tong} ca đạt`);
if (fails.length) {
  console.log("\nCA HỎNG:");
  for (const f of fails) console.log("  " + f);
  process.exit(1);
}
console.log("Tất cả ca đạt — code Pandas chạy được và khớp số app.");
