import { Agent, TextBlock, type Message } from "@strands-agents/sdk";
import { OpenAIModel } from "@strands-agents/sdk/models/openai";
import { GoalLoop, type Validator } from "@strands-agents/sdk/vended-plugins/goal";
import { OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL, noThinkFetch } from "@/lib/providers";
import { leadingDirection, hasReversedRange } from "./audit";
import { insightSystem } from "./prompts";
import type { AgentResult } from "./types";

/**
 * VercelModel (bridge sang Vercel AI SDK) KHÔNG dùng được ở đây: @strands-agents/sdk 1.12.0
 * (bản mới nhất) chỉ hỗ trợ LanguageModelV3, nhưng @ai-sdk/openai-compatible@3.0.14 đã lên
 * @ai-sdk/provider@4.0.3 (V4) — lệch phiên bản thật, không phải lỗi cấu hình. Dùng thẳng
 * OpenAIModel (api:'chat', vì Ollama chỉ có /v1/chat/completions, không có /v1/responses)
 * với clientConfig trỏ Ollama — độc lập với Vercel AI SDK nên tránh được lệch phiên bản này.
 * Tái dùng đúng noThinkFetch từ providers.ts để giữ nguyên cách tắt-thinking đã xác minh đúng.
 */
function insightModel() {
  return new OpenAIModel({
    api: "chat",
    modelId: OLLAMA_MODEL,
    temperature: 0.3,
    apiKey: OLLAMA_API_KEY,
    clientConfig: { baseURL: OLLAMA_BASE_URL, fetch: noThinkFetch },
  });
}

function textOf(message: Message): string {
  return message.content
    .flatMap((b) => (b instanceof TextBlock ? [b.text] : []))
    .join("")
    .trim();
}


/**
 * Validator thuần code (không gọi thêm LLM để tự phê bình chính nó — GoalLoop hỗ trợ
 * "Programmatic Validator" đúng cho việc này). Gộp luật cũ (bắt ký tự Hán rò từ Qwen) +
 * chặn rỗng + enforce 2 luật ĐÃ ghi trong insightSystem nhưng trước đây chưa kiểm
 * (KHÔNG markdown, KHÔNG mở đầu bằng "Nhận định") + kiểm CHIỀU tăng/giảm.
 *
 * Luật chiều bật theo `result.comparison` (do computeAnswer đặt khi thực sự tính so sánh hai kỳ),
 * KHÔNG theo `plan.intent`. Đây là một BUG THẬT đã sửa: nhánh tính YoY chạy khi
 * `intent==="yoy_growth" HOẶC years.length>=2`, còn luật này trước đây chỉ bật khi
 * `intent==="yoy_growth"` ⇒ câu "doanh thu 2023 và 2024" (planner trả direct_retrieval + 2 năm)
 * vẫn ra % có dấu mà bộ kiểm chiều BỊ TẮT — model tự do viết ngược chiều, không ai chặn.
 *
 * Dùng CHUNG `leadingDirection`/`hasReversedRange` với Auditor để hai nơi không lệch luật.
 * Luật cũ ("có từ giảm VÀ không có từ tăng") bỏ lọt mọi câu hai vế kiểu "doanh thu tăng nhưng
 * biên lợi nhuận giảm" — nay xét chiều NÊU RA ĐẦU TIÊN, đó mới là điều người đọc tiếp nhận.
 */
function makeValidator(result: AgentResult): Validator {
  return (response) => {
    const text = textOf(response);
    if (!text || text.length <= 4) {
      return { passed: false, feedback: "Chưa có nhận định hoặc quá ngắn. Viết 1-2 câu nhận định thật." };
    }
    if (/[一-鿿]/.test(text)) {
      return { passed: false, feedback: "Có lẫn ký tự Hán/Trung. Viết lại HOÀN TOÀN bằng tiếng Việt thuần." };
    }
    if (/[*_#`]|^Nhận định/i.test(text)) {
      return { passed: false, feedback: "Không dùng markdown (*, _, #, `) và không mở đầu bằng 'Nhận định'." };
    }
    const cmp = result.comparison;
    if (cmp) {
      const said = leadingDirection(text);
      if (said && said !== cmp.direction) {
        const dung = cmp.direction === "up" ? "TĂNG" : "GIẢM";
        return {
          passed: false,
          feedback: `SAI CHIỀU: số liệu thực tế ${dung} ${Math.abs(result.value ?? 0)}% từ ${cmp.from} sang ${cmp.to}, nhưng bạn viết theo chiều ngược lại. Viết lại đúng chiều ${dung}.`,
        };
      }
      if (hasReversedRange(text)) {
        return {
          passed: false,
          feedback: `SAI THỨ TỰ THỜI GIAN: kỳ gốc là ${cmp.from}, kỳ sau là ${cmp.to}. Viết "${cmp.from} sang ${cmp.to}", không viết ngược lại.`,
        };
      }
    }
    return true;
  };
}

/** Nhận định ngắn kiểu analyst — best-effort qua Qwen3.5-4B local, có tự kiểm + sinh lại (tối đa 2 lần). */
export async function genInsight(question: string, result: AgentResult): Promise<string | null> {
  try {
    // timeout 30s (không phải 20s): đo được production có lần mất 25,9s cho 1 lần sinh (cold model);
    // 20s từng khiến lần cần retry-vì-sai-chiều bị cắt ngang giữa chừng → trả null oan dù model
    // đã sửa đúng ở attempt 2, chỉ là chưa kịp xong trong ngân sách cũ.
    const goalLoop = new GoalLoop({ goal: makeValidator(result), maxAttempts: 2, timeout: 30000 });
    // printer:false — Agent mặc định TỰ IN output ra console server (thấy rõ khi debug: mỗi lần
    // GoalLoop sinh lại, bản cũ lẫn bản mới đều bị in ra, rác log). Không ảnh hưởng giá trị trả về.
    const agent = new Agent({ model: insightModel(), systemPrompt: insightSystem, plugins: [goalLoop], printer: false });
    // Đưa CẢ dữ liệu biểu đồ vào, không chỉ mỗi con số. Bản trước chỉ gửi câu hỏi + một số rồi
    // yêu cầu "thêm góc nhìn" — tức bắt model đánh giá trong khi giấu dữ liệu để đánh giá, nên nó
    // buộc phải đoán. Đo được trên ảnh demo: hỏi cơ cấu nguồn vốn, model viết "mức độ tự chủ tài
    // chính cao, giảm phụ thuộc vốn vay" trong khi biểu đồ NGAY CẠNH cho thấy nợ chiếm 75,4% và
    // Nợ/VCSH = 3,07x — kết luận ngược hẳn dữ liệu. Có cơ cấu trong tay thì nó không phải đoán.
    const parts = [`Câu hỏi: ${question}`, `Kết quả: ${result.answer}`];
    if (result.chart?.config) {
      parts.push(`Số liệu kèm theo (dùng để so sánh): ${JSON.stringify(result.chart.config).slice(0, 700)}`);
    }
    const r = await agent.invoke(parts.join("\n"), {
      cancelSignal: AbortSignal.timeout(30000),
    });
    const text = textOf(r.lastMessage);
    const outcome = goalLoop.lastResult(agent);
    return outcome?.passed && text ? text : null;
  } catch {
    return null;
  }
}
