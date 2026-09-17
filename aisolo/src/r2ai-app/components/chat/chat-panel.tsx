"use client";
import { useEffect, useRef, useState } from "react";
import { Paperclip, SendHorizontal, Loader2, Sparkles } from "lucide-react";
import { AssistantResult } from "./message-view";
import { StepTimeline, TraceReview } from "./step-timeline";
import type { ChatMsg } from "@/components/workspace";

interface Props {
  messages: ChatMsg[];
  sending: boolean;
  /** Có bảng tra cứu được — quyết định gợi ý câu hỏi mẫu và lời mời ở màn hình trống. */
  hasData: boolean;
  /**
   * Được phép GỬI câu hỏi. Tách khỏi `hasData` có chủ ý: khi người dùng đã nạp file mà app không
   * dựng được bảng, họ vẫn phải hỏi được — và nhận lời giải thích thiếu gì. Khoá ô nhập lúc đó
   * chính là ngõ cụt cũ: gõ câu hỏi mà không có phản hồi nào.
   */
  canAsk: boolean;
  onSend: (q: string) => void;
  onAttach: () => void;
}

const SUGGESTIONS = [
  "Tổng tài sản năm 2024 là bao nhiêu?",
  "ROE năm 2024?",
  "Doanh thu thuần tăng bao nhiêu % so với năm trước?",
  "Tỷ lệ nợ trên vốn chủ sở hữu?",
];

export function ChatPanel({ messages, sending, hasData, canAsk, onSend, onAttach }: Props) {
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function submit(e?: React.FormEvent) {
    e?.preventDefault();
    const q = input.trim();
    if (!q || sending || !canAsk) return;
    onSend(q);
    setInput("");
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <EmptyState hasData={hasData} onPick={(q) => !sending && hasData && onSend(q)} />
        ) : (
          <div className="mx-auto flex max-w-2xl flex-col gap-4">
            {messages.map((m) => (
              <MessageBubble key={m.id} m={m} />
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <form onSubmit={submit} className="shrink-0 border-t border-hairline p-3">
        <div className="mx-auto flex max-w-2xl items-center gap-2">
          <button
            type="button"
            onClick={onAttach}
            title="Tải lên Excel/CSV"
            className="flex h-11 shrink-0 items-center gap-1.5 rounded-sm px-3 text-sm font-medium text-muted hover:bg-surface-soft hover:text-primary"
          >
            <Paperclip className="h-4 w-4" />
            <span>Tải lên</span>
          </button>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              hasData ? "Hỏi về báo cáo tài chính…" : canAsk ? "Hỏi xem vì sao chưa đọc được file…" : "Đính kèm file để bắt đầu…"
            }
            className="h-11 flex-1 rounded-sm border border-hairline bg-canvas px-3.5 text-[16px] text-ink placeholder:text-muted-soft focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/25"
          />
          <button
            type="submit"
            disabled={sending || !input.trim() || !canAsk}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-sm bg-primary text-on-primary hover:bg-primary-hover disabled:opacity-50"
          >
            {sending ? <Loader2 className="h-5 w-5 animate-spin" /> : <SendHorizontal className="h-5 w-5" />}
          </button>
        </div>
      </form>
    </div>
  );
}

function MessageBubble({ m }: { m: ChatMsg }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-lg rounded-br-sm bg-primary px-4 py-2.5 text-[16px] text-on-primary">
          {m.text}
        </div>
      </div>
    );
  }
  return (
    <div className="flex justify-start">
      <div className="max-w-[90%] rounded-lg rounded-bl-sm border border-hairline bg-canvas px-4 py-3 shadow-[var(--shadow-card)]">
        {m.result ? (
          <div className="space-y-3">
            <AssistantResult result={m.result} />
            {m.steps && m.steps.length > 0 && <TraceReview steps={m.steps} />}
          </div>
        ) : m.text ? (
          <p className="text-body">{m.text}</p>
        ) : (
          <StepTimeline steps={m.steps ?? []} />
        )}
      </div>
    </div>
  );
}

function EmptyState({ hasData, onPick }: { hasData: boolean; onPick: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
      <span className="flex h-14 w-14 items-center justify-center rounded-full bg-primary-soft text-primary">
        <Sparkles className="h-7 w-7" />
      </span>
      <div>
        <h2 className="text-lg font-semibold text-ink">Hỏi về báo cáo tài chính</h2>
        <p className="mt-1 max-w-sm text-sm text-muted">
          {hasData
            ? "Chọn một câu gợi ý hoặc tự đặt câu hỏi bằng tiếng Việt."
            : "Chọn một doanh nghiệp niêm yết ở panel phải để hỏi ngay, hoặc đính kèm file Excel/CSV của bạn."}
        </p>
      </div>
      {hasData && (
        <div className="flex max-w-md flex-wrap justify-center gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => onPick(s)}
              className="rounded-full bg-primary-soft px-3 py-1.5 text-sm font-medium text-primary hover:bg-primary/15"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
