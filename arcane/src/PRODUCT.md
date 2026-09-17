# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary user: the presenter, operating the app live during a competition
demo-day presentation and the Q&A after it — typing questions, reading the
pipeline trace, and pointing at cited evidence in real time.

Secondary, first-class viewer: the audience/judges watching the same screen
(projector or shared view) without touching the keyboard. They do not
interact, but the interface must read clearly to them, not only to the
presenter.

## Product Purpose

Demonstrate, inside a five-minute presentation window plus live Q&A, that the
ViFinQA retrieval pipeline answers a fresh, unseen Vietnamese financial-
statement question correctly and transparently — not by replaying a graded
submission, but by actually running the gate, retrieval, reranking, selection,
and grounded-answer stages against a question typed live.

## Positioning

Unlike a fluent LLM answer with no verifiable source, every number this app
shows is the output of an executed, validated Pandas expression run against a
specific cited source table and cell — one the pipeline's own retrieval
methodology selected. That methodology (deterministic document gate, BM25 +
optional dense retrieval, graph/note expansion, Qwen reranking, RRF fusion,
budgeted slot selection) is the same one that produced the best table- and
document-retrieval scores on the competition's private leaderboard among four
teams. A competitor's demo showing only a plausible-sounding answer cannot
truthfully make the same traceability claim.

## Operating Context

Presenter's machine runs the FastAPI demo server and browser UI. Two model
services sit behind it over HTTP: a planner/answer model (OpenAI-compatible
chat completions) and a table reranker (a small custom `/rerank` endpoint).
For the real event these run on a rented GPU; a CPU-only "dev tier" of smaller
models exists for rehearsing the UI and pipeline wiring beforehand, with the
known limitation that the small planner/answer model cannot reliably pick the
correct source cell (retrieval and reranking are real either way).

One question is in flight at a time (the server enforces this). A live run
against real models can take from a few seconds (GPU, warmed up) to several
minutes (CPU dev tier) — rehearsal and warm-up before presenting are part of
normal use, not an edge case. Chat history persists only in the browser
(localStorage); there is no account system or server-side session store.

## Capabilities and Constraints

Built: deterministic document gate; report-scoped BM25 retrieval with an
optional same-triple dense arm (off by default — measured to contribute under
1% of selected tables against its VRAM cost); graph and note-link table
expansion (off by default, matching the graded methodology); HTTP-based
reranking with RRF fusion; budgeted, arity-gated slot selection reproducing
the graded submission's own packaging step; sandboxed, allowlisted Pandas
execution for the final answer; citations derived only from the cells the
executed query actually read (never a stored or guessed value); a streaming
pipeline trace showing each real stage as it completes, with retrieved and
reranked candidate tables clickable into a real formatted table view; a
fixture mode that exercises the whole UI with a labelled placeholder answer
and no models, clearly banner-marked so it is never mistaken for a live run.

Constraints: the corpus and every question/answer/label are Vietnamese —
Vietnamese is the interface's working language throughout (prompts, labels,
error text), not a placeholder to be swapped for English later. Model
serving is via the OpenAI-compatible chat-completions shape plus one custom
rerank endpoint; no other model transport is supported. No production users,
authentication, or persistence beyond the browser exist, and none are planned
— this is a demo-day tool, not a shipped product.

## Evidence on Hand

Real ViFinQA corpus on disk: 1,973 Vietnamese financial-statement reports.
Private leaderboard result for the retrieval methodology this app reproduces:
best Tables F2 (0.6293) and best Docs F2 (0.9768) among four competing teams,
though 4th of 4 overall (0.6451) because of a separate, lower-scoring
answer/execution stage on the graded submission — a gap this app's own
grounded-answer stage is a fresh attempt to close, not a repeat of it. No
organizer-held gold labels exist locally, so any local accuracy figure is
agreement with the team's own prior artifacts, not a claim of ground-truth
correctness. As of the app's current state, its answer stage has been
exercised end-to-end only with small CPU-tier models (which cannot reliably
pick the correct cell); a full-size-model rehearsal on rented GPU hardware
has not yet been recorded, and no future work should present one as done
without running it.

## Product Principles

- Every displayed number is executed from a real, validated expression against
  a cited source cell — never a stored constant, a cached answer, or an
  unexecuted guess. This constraint is not to be traded away for a smoother
  or faster-feeling interface.
- Show the real pipeline, not a performance of one: only content the backend
  actually produced appears on screen. A slow or waiting stage may say so
  honestly; it is never given fabricated "thinking" text.
- A failed or degraded stage says so visibly and specifically. It is never
  hidden, silently retried into a false success, or reworded to look better
  than what happened.
- The presenter and the audience are both first-class viewers of the same
  screen. Design for someone reading over a shoulder or a projector, not only
  for the person with their hands on the keyboard.
- Vietnamese end-to-end is a fixed product constraint, not a temporary
  default awaiting localization.

## Brand Commitments

Existing name and mark already in the shipped interface: "VIFin", with a
single-letter monogram mark ("V") and the tagline "Trợ lý tài chính"
(financial assistant). Treat these as established rather than open for a
casual rename.
