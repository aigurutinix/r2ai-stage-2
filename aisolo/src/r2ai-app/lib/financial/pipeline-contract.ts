/**
 * HỢP ĐỒNG GIỮA CÁC AGENT — chặn lỗi sai thứ tự ngay lúc khởi động, không đợi tới lúc chạy.
 *
 * Vì sao cần: `Ctx` là một object dùng chung với nhiều trường optional, và mỗi agent ngầm giả định
 * các agent trước đã chạy. Giả định đó không được ghi ở đâu cả, nên đổi thứ tự trong `PIPELINE`
 * hoặc chèn thêm một agent vào sai chỗ sẽ **không báo lỗi gì** — nó chỉ đọc phải `undefined` rồi
 * hỏng ở một chỗ khác, xa nguyên nhân.
 *
 * Đây là lớp lỗi đã gặp thật: agent `audit` được đặt trước `narrate` nên nó soi `result.insight`
 * lúc trường đó còn chưa tồn tại. Lần ấy phát hiện được nhờ đọc lại code, không nhờ công cụ nào.
 *
 * Tách riêng khỏi `agents.ts` để KIỂM ĐƯỢC: file này không kéo theo Strands SDK hay model nào,
 * nên `verify-pipeline-contract.mjs` biên dịch và chạy nó một mình được. Một bộ kiểm mà bản thân
 * nó không được kiểm thì chỉ là chỗ dựa tinh thần.
 */

/** Phần hợp đồng của một agent — chỉ gồm thứ cần để kiểm thứ tự, không gồm phần `run`. */
export interface AgentContract {
  id: string;
  /** Trường của Ctx mà agent này ĐỌC. Phải có agent trước đó sinh ra, hoặc nằm trong seed. */
  requires: readonly string[];
  /** Trường của Ctx mà agent này GHI RA. */
  provides: readonly string[];
}

/**
 * Kiểm thứ tự chuỗi agent. Ném lỗi ngay nếu một agent cần thứ chưa ai sinh ra.
 *
 * @param seed  các trường có sẵn từ đầu (do runAgent khởi tạo)
 * @param chain chuỗi agent theo đúng thứ tự sẽ chạy
 */
export function checkPipelineOrder(seed: readonly string[], chain: readonly AgentContract[]): void {
  const have = new Set<string>(seed);
  for (const a of chain) {
    for (const need of a.requires) {
      if (!have.has(need)) {
        throw new Error(
          `Chuỗi agent sai thứ tự: "${a.id}" cần "${need}" nhưng chưa agent nào phía trước sinh ra. ` +
            `Đang có: [${[...have].join(", ")}].`,
        );
      }
    }
    for (const p of a.provides) have.add(p);
  }
}
