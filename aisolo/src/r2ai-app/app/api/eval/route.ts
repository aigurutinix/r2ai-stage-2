// Dev-only: đo F2 macro của retrieval skeleton trên mock repo. Không gọi LLM.
import { runRetrievalEval } from "@/lib/financial/eval";

export const runtime = "nodejs";

export async function GET() {
  return Response.json(runRetrievalEval());
}
