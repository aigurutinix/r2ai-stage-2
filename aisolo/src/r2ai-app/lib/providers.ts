import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

/**
 * Provider LLM: Qwen3.5-4B chạy LOCAL qua Ollama (endpoint OpenAI-compatible).
 * Hợp lệ cuộc thi: open-weights (Apache-2.0), ~4.66B (≤14B), repo GGUF upload 02/03/2026 (≤ 31/05).
 *
 * Tắt "thinking" bằng cách chèn `reasoning_effort:"none"` vào body mọi request chat.
 * BẮT BUỘC: trên endpoint /v1 của Ollama, nếu KHÔNG tắt thinking thì Qwen3.x trả `content`
 * RỖNG (text bị nhét vào field `reasoning`) → parse JSON vỡ. (`think:false` vô tác dụng trên
 * /v1 — xem Ollama #15288 / #14820.) Dùng custom fetch vì SDK v3 không nhận `extraBody`.
 */
/** Export để tái dùng ở nơi cần gọi Ollama trực tiếp (vd insight-agent.ts qua Strands OpenAIModel)
 * — tránh viết lại logic tắt-thinking (đã tốn công tìm ra tham số đúng, đừng lặp lại lỗi cũ). */
export const noThinkFetch = (async (url, options) => {
  if (options && typeof options.body === "string") {
    try {
      const body = JSON.parse(options.body);
      if (body && typeof body === "object" && "messages" in body) {
        body.reasoning_effort = "none";
        options = { ...options, body: JSON.stringify(body) };
      }
    } catch {
      // body không phải JSON — để nguyên
    }
  }
  return globalThis.fetch(url, options);
}) as typeof globalThis.fetch;

export const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434/v1";
export const OLLAMA_API_KEY = process.env.OLLAMA_API_KEY || "ollama";
export const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M";

const ollama = createOpenAICompatible({
  name: "ollama",
  baseURL: OLLAMA_BASE_URL,
  apiKey: OLLAMA_API_KEY,
  fetch: noThinkFetch,
});

export const primaryModel = () => ollama(OLLAMA_MODEL);
