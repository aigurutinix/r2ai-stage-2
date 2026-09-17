/**
 * Kiểm BỘ KIỂM hợp đồng agent (lib/financial/pipeline-contract.ts).
 *
 * Vì sao cần: `checkPipelineOrder` chạy lúc nạp module `agents.ts`, và chuỗi thật hiện đang ĐÚNG —
 * nên nó không bao giờ ném lỗi trong lúc dùng bình thường. Một bộ kiểm chưa từng kêu thì không có
 * bằng chứng nào cho thấy nó biết kêu. File này bắt nó phải kêu ở những ca dàn dựng, và phải im ở
 * chuỗi đúng.
 *
 * Chạy:  npx tsc lib/financial/pipeline-contract.ts --outDir <dir> --module commonjs \
 *          --target es2020 --moduleResolution node --skipLibCheck
 *        node verify-pipeline-contract.mjs <dir>
 */
import { createRequire } from "node:module";
import path from "node:path";

const dir = process.argv[2];
if (!dir) {
  console.error("Thiếu tham số: node verify-pipeline-contract.mjs <thư-mục-đã-compile>");
  process.exit(2);
}
const { checkPipelineOrder } = createRequire(import.meta.url)(path.resolve(dir, "pipeline-contract.js"));

const SEED = ["tidy", "question", "activeTicker", "prevPlan"];
/** Chuỗi THẬT của app, chép đúng thứ tự trong agents.ts. */
const THAT = [
  { id: "understand", requires: ["tidy", "question"], provides: ["plan", "provider"] },
  { id: "retrieve", requires: ["tidy", "plan", "provider"], provides: ["result"] },
  { id: "audit", requires: ["tidy", "result"], provides: [] },
  { id: "chart", requires: ["tidy", "plan", "result"], provides: [] },
  { id: "narrate", requires: ["tidy", "question", "result"], provides: [] },
];

const run = (chain) => {
  try {
    checkPipelineOrder(SEED, chain);
    return null;
  } catch (e) {
    return e.message;
  }
};

const CASES = [
  ["chuỗi thật phải ĐI QUA", THAT, false],
  ["đảo retrieve lên trước understand → phải KÊU",
    [THAT[1], THAT[0], THAT[2], THAT[3], THAT[4]], true],
  ["bỏ retrieve → audit mất `result` → phải KÊU",
    [THAT[0], THAT[2], THAT[3], THAT[4]], true],
  ["chèn agent mới cần `insight` mà chưa ai sinh → phải KÊU",
    [...THAT, { id: "moi", requires: ["insight"], provides: [] }], true],
  ["agent chỉ cần seed → phải ĐI QUA",
    [{ id: "chi_can_seed", requires: ["question"], provides: [] }], false],
  ["chuỗi rỗng → phải ĐI QUA", [], false],
];

let pass = 0, fail = 0;
for (const [name, chain, phaiKeu] of CASES) {
  const err = run(chain);
  const ok = phaiKeu ? err !== null : err === null;
  if (ok) pass++; else fail++;
  console.log(`${ok ? "  OK  " : "  FAIL"} ${name}${err ? `\n         -> ${err}` : ""}`);
}
console.log(`\n${pass}/${pass + fail} ca đạt`);
process.exit(fail === 0 ? 0 : 1);
