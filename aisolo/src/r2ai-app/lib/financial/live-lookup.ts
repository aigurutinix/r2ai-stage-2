import "server-only";
import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";

/**
 * Tra cứu TRỰC TIẾP một chỉ tiêu trên báo cáo OCR gốc — tầng dự phòng của kho trích sẵn.
 *
 * Vì sao cần: kho (`public/corpus/`) chỉ giữ 30 chỉ tiêu registry mỗi doanh nghiệp, đổi lại có
 * đẳng thức kế toán xác nhận từng niên độ. Báo cáo gốc có ~650 dòng, và câu hỏi thật hay rơi vào
 * phần còn lại — "Lãi tiền gửi", "Chi phí xây dựng cơ bản dở dang", "Số dư cho vay ngành Thương
 * mại". Không có tầng này thì app đành trả lời "chưa hỗ trợ chỉ số" dù số liệu nằm ngay trong báo
 * cáo.
 *
 * Cái giá: dùng khâu chọn dòng theo từ vựng của pipeline — đo được **67,6%** mỗi lần tra — và
 * KHÔNG có đẳng thức nào xác nhận. Nên kết quả luôn mang `verified: false`, và giao diện phải nói
 * rõ. Đây là điểm phân biệt hai tầng: "lấy từ đâu" khác với "và con số này tự kiểm được".
 *
 * AN TOÀN: đây KHÔNG phải thực thi mã do mô hình sinh. Nó chạy một script CỐ ĐỊNH
 * (`pipeline/lookup_live.py`), tham số truyền qua mảng argv — không qua shell nên không có đường
 * tiêm lệnh — và ticker/năm còn được kiểm khuôn trước khi truyền.
 */

export interface LiveHit {
  ok: true;
  value: number;
  label: string;
  maSo: string | null;
  doc: string;
  table: string;
  line: number | null;
  unitFactor: number;
  /** Luôn false: không có đẳng thức nào xác nhận giá trị này. */
  verified: false;
}
export interface LiveMiss {
  ok: false;
  error: string;
}
export type LiveResult = LiveHit | LiveMiss;

const REPO = path.join(process.cwd(), "..");
const SCRIPT = path.join(REPO, "pipeline", "lookup_live.py");
const RE_TICKER = /^[A-Z0-9]{2,6}$/;
const RE_YEAR = /^(19|20)\d{2}$/;

/** Python nào: biến môi trường trước, rồi .venv của repo (README hướng dẫn tạo), cuối cùng PATH. */
function pythonBin(): string {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  for (const p of [
    path.join(REPO, ".venv", "Scripts", "python.exe"),
    path.join(REPO, ".venv", "bin", "python"),
  ]) {
    if (fs.existsSync(p)) return p;
  }
  return process.platform === "win32" ? "python" : "python3";
}

export async function liveLookup(
  ticker: string,
  year: string,
  query: string,
  doctype: "consolidated" | "separate" = "consolidated",
): Promise<LiveResult> {
  const tk = ticker.trim().toUpperCase();
  const yr = String(year).trim();
  const q = query.trim();
  if (!RE_TICKER.test(tk) || !RE_YEAR.test(yr) || !q || q.length > 200) {
    return { ok: false, error: "Tham số tra cứu không hợp lệ" };
  }
  if (!fs.existsSync(SCRIPT)) {
    return { ok: false, error: "Chưa có pipeline/lookup_live.py — tra cứu trực tiếp không khả dụng" };
  }

  const args = [SCRIPT, "--ticker", tk, "--year", yr, "--doctype", doctype, "--query", q];
  const out = await new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
    const ps = spawn(pythonBin(), args, { cwd: REPO, env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
    let stdout = "";
    let stderr = "";
    const kill = setTimeout(() => ps.kill(), 20_000);
    ps.stdout.on("data", (d) => (stdout += d));
    ps.stderr.on("data", (d) => (stderr += d));
    ps.on("error", (e) => {
      clearTimeout(kill);
      resolve({ code: -1, stdout: "", stderr: String(e) });
    });
    ps.on("close", (code) => {
      clearTimeout(kill);
      resolve({ code: code ?? -1, stdout, stderr });
    });
  });

  if (out.code !== 0 || !out.stdout.trim()) {
    // stderr thường là thông báo "không thấy dữ liệu ViFinQA" của pipeline.py — nói lại nguyên văn
    // thay vì nuốt đi, vì đó chính là thứ người dùng cần biết để sửa.
    const msg = out.stderr.trim().split("\n").slice(0, 2).join(" ").slice(0, 200);
    return { ok: false, error: msg || "Không chạy được tra cứu trực tiếp" };
  }
  try {
    return JSON.parse(out.stdout) as LiveResult;
  } catch {
    return { ok: false, error: "Kết quả tra cứu không đọc được" };
  }
}
