# Pipeline design

Written as a design, not a plan of work. It says what the stages are, what each one is
allowed to decide, why the split falls where it does, and how each stage can be checked
without a leaderboard slot.

---

## The one number the design is built around

The organisers measured their own system two ways: hand it the gold table and
chain-of-thought scores **87%**; hand it ten retrieved tables and the same model scores
**64%**. Our retrieval already puts the answer's table inside the top nine about 96% of the
time. So those 23 points are not missing tables. They are the cost of *choosing*.

That is also what their error budget says — 54.7% of end-to-end failures are the wrong
cell, against 0.9% arithmetic and 0.5% formula. The model is not bad at finance. It is bad
at picking one row out of a shortlist, and everything else it does is nearly free of error.

**So the design has one job: decide the cell before anything depends on reading it, and
decide it with evidence that is not another model opinion.**

Everything that failed on 21–22/08 failed because it added another opinion — a second
reader, a reordered prompt, a self-audit, a different model family as auditor. Opinions
correlate. Arithmetic does not.

---

## The evidence that is not an opinion

Three checks are facts about the documents. None of them shares code with the retriever or
the reader, which is why they can both *select* candidates and *score* the pipeline
offline.

**Printed identities.** Vietnamese statements print their own arithmetic in the row labels:
`TỔNG TÀI SẢN (270 = 100 + 200)`, `50 = 30 + 40`, `440 = 300 + 400`. If a candidate cell is
the `270` line of a balance sheet, the `100` and `200` lines of the same column must sum to
it. Measured earlier at 97–99.6% agreement where it applies. It confirms the row and the
column together. It cannot confirm the unit, because it is scale-invariant.

**Cross-document comparative.** "Số cuối năm" of report *Y* is "Số đầu năm" of report
*Y+1*, same Mã số. Measured at 90.9% over 9,793 pairs. This one *does* confirm the unit,
because the two reports are independent OCR of two documents.

**Note-to-statement tie.** A note's column total equals the statement line it details.
31,564 such ties exist in the corpus. It confirms that a note table belongs to the
indicator a question names — which is exactly the failure that beat us by hand: a question
about *phải thu khách hàng* answered from the note on *phải thu khác*, both of which have a
row called "Bên thứ ba".

The first two apply only to the three main statements. The third extends coverage to notes.
Together they will not cover every question, and the design does not pretend otherwise —
unverified questions fall through to what we ship today.

---

## The stages

    question
      │
      ▼
    A  spec        ← the only model call
      │            company/companies, year(s), scope, indicator(s) + label variants,
      │            operation, unit, period
      ▼
    B  candidates  ← code: metadata filter, then label matching per operand
      │            top ~5 (table, row, col) per operand, each with its declared scale
      ▼
    C  verify      ← code: identities, cross-document, note tie
      │            candidates ranked by external evidence, not by similarity
      ▼
    D  compute     ← code: the locked operation, one unit conversion, one rounding
      │
      ▼
    E  gate        ← code: answer-type rules; else fall back to today's vote answer

### A. Spec — the only place a model is used

Input: the question text alone. No tables. Output, as JSON:

    {"cong_ty": ["ASM"], "nam": [2025], "pham_vi": "separate",
     "chi_tieu": [{"ten": "chi phí quản lý doanh nghiệp",
                   "bien_the": ["Chi phí quản lý doanh nghiệp",
                                "Chi phí quản lý DN", "CPQLDN"],
                   "ky": "nam_nay"}],
     "phep_tinh": "single", "don_vi": "triệu đồng"}

Two design points that matter more than the schema:

*`bien_the` — the label variants.* The reason to ask a model here at all is that
Vietnamese indicator wording in a question rarely matches the row label verbatim, and a
matcher that tries one phrasing missed the row on 18% of questions in an earlier
measurement. Generating three or four ways the label might be printed is something a model
does well and a rule does badly. The matching itself stays in code, where it is
deterministic and inspectable.

*`phep_tinh` — from the organisers' own closed vocabulary:*
`single | difference | ratio | growth | share | sum | average | minimum | maximum | argmin
| argmax | count`. Closed means checkable: anything outside it is a malformed spec.

This stage is checkable without gold. The ticker must resolve against the corpus; the years
must appear in the question; the unit must be one the question actually names; the operation
must be in the vocabulary; the number of indicators must be consistent with the operation
(a `ratio` needs two, a `single` needs one). A spec that fails any of those is rejected and
the question falls through. **Cost: one short call per question — roughly $0.15 for all
1012, against about $3 for the three sources we vote over today.**

### B. Candidates — per operand, not per question

Today's retrieval returns nine tables for a *question*. That is the wrong unit: a question
about four companies needs one cell per company, and a ratio needs two cells that may live
in different tables.

So Stage B runs once per operand in the spec. The metadata filter — company, year, scope —
already works and stays exactly as it is. Within what it leaves, rows are matched against
the label variants from Stage A. Output per operand: about five `(table, row, col)` triples,
each carrying the row label, the note heading printed above the table, and the scale that
table declares.

The column is decided here too, from `ky` and the column headers, not left to the reader —
"Số cuối năm" against "Số đầu năm" is a lookup, not a judgement.

### C. Verify — the new stage, and the point of the whole design

Each candidate is scored by the checks above:

* candidate sits on a Mã số that participates in a printed identity → test it;
* the same Mã số exists in the neighbouring year's report → test current against prior;
* candidate is in a note → test the note's total against the statement line it details.

A candidate that passes a check is promoted; one that contradicts a check is eliminated
outright. Where exactly one candidate survives, the pipeline is in the *oracle* condition
for that operand — the 87% setting rather than the 64% one, for reasons that have nothing
to do with how good the reader is.

Where several survive or none does, record that. **The distribution of "how many operands
end up uniquely verified" is the number that tells us whether this design is worth
shipping, and it can be computed before any submission.** That is the first honest offline
metric in this project, because the checks share no code with the retriever, the renderer or
the reader.

### D. Compute — deterministic, no model

Apply the operation from Stage A to the verified operands. Convert with the scale the table
declares. Round once, at the end, to two decimals. The organisers' error budget puts
arithmetic and formula errors at 1.4% combined: there is nothing to gain here from a model
and there is the whole of the unit-conversion regression to lose. `vote7` lost 24 questions
by letting a scale decision be made in the wrong place; this stage exists so that decision
has exactly one home.

### E. Gate and fallback

The answer-type rules already written: a "năm nào" answer must be one of the years the
question lists; a count must be an integer no larger than the companies named; a percentage
must be in range. Then: any question the chain did not resolve keeps its current answer from
the three-source vote. The pipeline is **additive** — it can only replace an answer where it
has external evidence, so a stage that turns out to be weak costs coverage, not score.

---

## Multi-hop, which is 216 questions we currently score near zero on

The Hard tier is defined by an intermediate result choosing the next lookup: *"in the year
with the lowest CFO-to-profit ratio, what was the quick ratio?"* A one-shot pipeline cannot
answer that shape at all, which is why we are near zero there while the best frontier model
gets 32.9%.

This design handles it without a new mechanism, because Stage A already produces operands
rather than an answer. A Hard spec has a **selector**: an operand set to evaluate, a
criterion to apply, and an outer operand whose identity the criterion decides. Run B→D over
the inner set, apply the criterion in code, instantiate the outer operand, run B→D again.
Two passes of the same machinery.

Worth noting what makes this tractable: a selector needs only the *ordering* of the inner
operands to be right, not their absolute values. Unit errors and scale errors cancel inside
a comparison. So the inner pass is a much easier problem than a normal read, and it is
exactly the pass where our known weaknesses do not bite.

---

## What this design is betting, stated so it can be wrong

1. **That verification narrows candidates to one often enough to matter.** If Stage C
   uniquely verifies only 15% of operands, the pipeline is a rounding error on top of the
   current vote. Measurable before spending anything.
2. **That the identities hold in OCR'd text as well as they did on the subset measured
   earlier** (97–99.6%). OCR damage to a Mã số or a digit breaks a check and eliminates a
   correct candidate — the failure mode is lost coverage, not wrong answers.
3. **That a single model call can produce a usable spec.** If specs are wrong 30% of the
   time, everything downstream inherits it. Checkable against the closed vocabulary and the
   corpus, per Stage A.

The first is the one to measure first, and it costs nothing but the code.

---

## Why this is more automatable than what we have

Today's 0.415 is a majority vote over three sources, one of which (`aimed.zip`) cannot be
regenerated. That is not a pipeline; it is an heirloom. Any change means re-running two
expensive passes and hoping the vote lands well, and it cannot be validated offline at all.

The design above is one model call plus deterministic code. It runs for well under a dollar,
end to end, in minutes. It can be re-run after every change. And because Stages A, C and E
each reject their own malformed output, a regression shows up as *coverage falling* rather
than as a score dropping silently a day later on the leaderboard.

That last property is the real argument for it. Not that it will reach 0.6 — I have no
measurement that says it will. But it is the first shape where being wrong is visible
before we submit.

---

# Why this design does not need patches

## What a patch is, in this project specifically

Not a general complaint — a list. `MAGNITUDE_CODES` takes the absolute value of eight
income-statement codes because 17% of them print in brackets. `PERIOD_HINT` is a regex over
"đầu năm|đầu kỳ|1/1/" to pick a column. `AGGREGATE_RE` catches "trung bình|bình quân"
because two averaged questions slipped into the single-cell list. `answer_gate` encodes four
answer shapes. `fix_units` rewrites a scale when a power of ten looks wrong.

Every one was added after seeing a specific failure, covers the cases that were seen, and
has an unbounded tail of cases that were not. Twenty of them produced 0.415. That is what
patching buys.

## Why this data invites patching so strongly

Because the chaos is real: OCR merges cells, drops thousands separators, loses column
headers, and the unit is declared in prose above the table sometimes, in the header other
times, and nowhere at all in roughly 40% of documents. Each of those is a special case
asking for a rule.

But the chaos is in the **representation**, not in the information. And the information is
massively over-determined:

* a statement line sums with its siblings into its parent, and the sum is printed in the
  label: `TỔNG TÀI SẢN (270 = 100 + 200)`;
* the same line appears again as the prior-period column of next year's report;
* a detailed line points at the note that details it, and the note's total comes back to it;
* every cell in a column shares one unit, so one cell's scale constrains the others.

A financial statement is a redundant object. **That redundancy is the substitute for
patches**, because a wrong reading almost always contradicts something, while a right one
contradicts nothing.

## The printed index nobody used

Measured on 400 documents of the corpus: 297 have a statement table carrying **both** a
`Mã số` column and a `Thuyết minh` column. Of 8,726 rows with a Mã số, **2,875 also carry a
note pointer** — a third of them, which is about the share of statement lines that have a
note at all.

    1. Phải thu ngắn hạn của khách hàng | 131 | 5.2
    2. Trả trước cho người bán ngắn hạn | 132 | 5.3
    3. Phải thu ngắn hạn khác           | 136 | 5.4

This is the answer to the failure that beat every mechanism tried so far. Question 135 asks
for receivables from *customers*; we answered from the note on *other* receivables. Both
notes contain a row called "Bên thứ ba", so no amount of keyword matching, reranking or
auditing could separate them — and none did: the dense retriever, the wider shortlist, the
narrower shortlist and two different auditors all left it wrong.

The document settles it in one hop. Mã số 131 → note 5.2. There is nothing to match.

## Navigation instead of matching

Three printed links replace three families of heuristic:

| instead of | navigate |
|---|---|
| ranking nine tables by word overlap | statement line → `Thuyết minh` pointer → that note |
| a scale scanner plus a document-level fallback | the column's scale that satisfies the printed identities |
| a regex deciding "cuối năm" versus "đầu năm" | the column that equals next year's prior column |

Each link is also its own check, which is what makes the pipeline self-validating rather
than self-confirming:

* if the note pointer leads somewhere whose total does not equal the statement line, the
  parse or the navigation is wrong — drop the question rather than answer it;
* if a candidate column does not satisfy `270 = 100 + 200`, that interpretation of the
  table is wrong, whatever the OCR did to it;
* an OCR run that dropped thousands separators makes a cell 1000× too large, and it fails
  the sum with its siblings. No rule about separators is needed to catch it.

Note what this does to the merged-cell problem, which is where a patch would normally go.
When code and value are fused into one string, the correct split is the one that satisfies
the identity. Nobody has to guess how many digits the code has.

## Where the model is allowed to act

Exactly twice, and both times on meaning rather than structure:

1. **Question → indicator.** Map "chi phí quản lý doanh nghiệp" to the statement line it
   names, and emit several ways that line might be printed. A rule doing this missed the row
   on 18% of questions; generating phrasings is what a language model is for.
2. **Indicator → row inside one note.** After navigation there are five or ten rows, not
   nine tables. This is the setting the organisers measured at 87% rather than 64%.

Nowhere else. Not parsing, not units, not arithmetic, not choosing among tables. Their own
error budget puts arithmetic at 0.9% and formula application at 0.5%: there is no upside
there and `vote7` is what the downside costs.

## Honest coverage

This does not cover the exam. Banks and securities firms use different statement forms —
about 130 and 156 questions respectively — where the Mã số scheme does not apply the same
way. Documents whose `Thuyết minh` column was lost to OCR have no pointer. Multi-hop
questions still need the outer loop described above.

So the design is **additive by construction**: it answers what it can verify and leaves the
rest to the three-source vote that scores 0.415 today. The failure mode is coverage falling,
which is visible before submitting, rather than a score dropping a day later.

What it is worth is an open question and should stay open until the first number: **of the
1012 questions, how many resolve to a Mã số or to a note a Mã số points at, and how many of
those satisfy their own checks.** That count is computable with no model calls and no
submissions, and it decides whether any of this is worth building.

---

# First measurements of the design (no model, no submission)

Both numbers below come from `scripts/fresh/measure_pointers.py` and
`scripts/fresh/measure_reach.py`. Neither uses a language model, neither costs a
leaderboard slot, and neither shares code with the retriever or the reader — so unlike
every offline figure in this project's history, they are not confirming themselves.

## The navigation works

25 documents, 883 statement rows carrying a note pointer:

| | |
|---|---|
| pointer resolves to a table with that printed heading | 813 / 883 = **92%** |
| that table's total ties back to the statement line | 761 / 813 = **94%** |
| **same test, deliberately pointed at the WRONG note** | 88 / 813 = **11%** |

The 8.5× separation is what makes the 94% mean something. A note table holds dozens of
cells and the tie is allowed against any of them, so without the permuted control the
figure would have been worthless — which is exactly how this project produced a "66%
document reachability" and an "89% retrieval recall" that both evaporated.

The 6% that resolve but do not tie are the useful part of the design as much as the 94%:
they are questions to drop rather than answer.

## Reach: at least 55% of the exam

60 questions, matched with the crude token matcher against the first document per
company-year:

| | |
|---|---|
| matches a row inside a note that a statement line points at | 32% |
| matches a statement row that has a pointer | 12% |
| matches a statement row with no pointer | 12% |
| **reachable by navigation** | **55%** |
| document has no table with a Mã số column | 30% |
| no match by this route | 15% |

Two things hold that number down and both are fixable rather than intrinsic. The
measurement loads only one document per company-year, so a question about the separate
statements whose Mã số table sits in the consolidated file counts as "no Mã số table". And
the matcher tries one phrasing, which is the failure the design hands to a model on purpose
— generating label variants was measured earlier to be worth 18% of questions.

## What the arithmetic says, marked as optimistic

If 55% of questions reach a uniquely navigated cell, and navigation puts the reader in the
organisers' oracle condition rather than their retrieved one (87% against 64%), and the
remaining 45% keep today's vote at roughly 0.40:

    0.55 × 0.87  +  0.45 × 0.40  ≈  0.66

This is arithmetic, not a prediction, and every term in it is soft. But it is the first
route in this project whose arithmetic reaches the leaders at all, and it gets there without
a single rule about brackets, periods or averages.

## The order to build it in

1. Reconcile statements per document by constraint satisfaction, and build the
   `Mã số ↔ note number` index from the printed `Thuyết minh` column. No model. Output is
   checkable by the identities and by the 94% tie above.
2. Raise reach: load every document for a company-year, not the first.
3. Add the one model call — question to indicator plus label variants — and re-measure
   reach. This is where the 55% should move.
4. Only then answer anything.

Step 1 and 2 cost nothing but code and can be verified before step 3 is written.

## Third measurement: the reach figure was taken with half the corpus invisible

The 58% above was measured while a one-line assumption threw away most statement tables.
Across 150 documents, asked where the `Mã số` column header actually sits:

| | |
|---|---|
| in row 0 — what the code looks at | 52% |
| in rows 1–3 — **missed entirely** | 23% |
| no header at all, but a column of 2–3 digit integers is plainly there | 25% |

So **48% of documents violate "the header is `grid[0]`"**, and for those the balance sheet,
income statement and cash-flow statement are invisible to everything built here —
`parse_statements`, the Mã số address book, the note-pointer index, all of it. That has been
true for three weeks and no measurement caught it, because every measurement was taken on
the questions the pipeline already answered.

Attribution matters here: the residue in the reach table was 23% ordinary companies against
7% banks. The sector story was wrong. It is a bug.

The fix is not a rule per layout, which is what "vá" would look like. It is the same
principle as the rest of the design: **the Mã số column is whichever column is mostly 2–3
digit integers, and the interpretation is accepted only if the printed identities hold on
it.** One mechanism, self-checking, no special case for headers in row 2 versus row 3
versus absent.

That moves this to the top of the build order, ahead of everything else — it costs nothing,
needs no model, and every number measured so far was taken with it broken.

## Steps 1 and 2, measured

Two one-line structural defects, in different places, each with a large blast radius. The
diagnosis took three attempts and the first two were wrong, which is worth recording as
much as the answer.

**Wrong guess 1: the header row.** Across 150 documents the `Mã số` header sits in row 0
only 52% of the time. That looked like the cause, so a constraint-based detector was built —
find the column that is mostly 2–3 digit codes, accept it only if the printed identities
hold. It recovered **2 of 28** previously-empty documents. The example document it failed on
(AAA 2018 consolidated) has `Mã số` in row 0, plainly.

**Wrong guess 2: the sector.** Banks and securities firms file on different forms, so the
residue was attributed to them. Measured: the residue is 23% ordinary companies against 7%
banks.

**The actual cause, for the statement index:** `parse_statements.parse_statement` ends with

    if scale is None:
        return None

A balance sheet that parsed perfectly is discarded in full because no "Đơn vị tính" line
could be found. On 250 documents: 59% have a statement, **18% are dropped for the unit
alone**, 23% genuinely have none. The printed identities confirm the discarded tables were
read correctly — 48 balance sheets satisfy two identities, 52 satisfy one, and among income
statements six satisfy four and one satisfies five.

The fix is not a rule. The scale is an attribute, not a precondition: keep the statement
with `scale = None` and resolve it afterwards from the document-wide scan, the neighbouring
year's column, or the note tie. Document coverage 59% → 77%.

**The actual cause, for the reach and pointer measurements:** those read the CSVs directly
and took the header to be `grid[0]`, which is a different code path from the statement index
and a different defect with the same shape. Locating the code and pointer columns by content
instead:

| | before | after |
|---|---|---|
| statement rows carrying a note pointer (25 docs) | 883 | **943** |
| pointer resolves and the note total ties back | 94% | **93%** |
| same test pointed at the wrong note (control) | 11% | **11%** |
| questions reachable by navigation (120 questions) | 58% | **78%** |
| residue "document has no Mã số table" | 23% | **2%** |

## What 78% does and does not mean

It means the question's wording matches a row that navigation can reach: a statement line,
or a line inside the note that a statement line points at. It does **not** mean the right
row, the right column or the right unit — those are still decisions, and the organisers'
87% figure describes a model reading a table already known to be the right one.

So 78% is a ceiling on this path's coverage. The previous 58% left no room above 0.6 once
multiplied by any realistic accuracy; 78% does. That is the whole change in status, and it
cost no model calls and no submissions.

---

# Correction: Stage C verifies the read, it does not make the choice

The design above says Stage C "decides the cell" by external evidence. Building the
measurement showed that is wrong, and the distinction matters enough to state plainly.

If a question matches both `131 Phải thu ngắn hạn của khách hàng` and `136 Phải thu ngắn
hạn khác`, **both rows read correctly**. Every identity holds on both. The cross-year
comparison holds on both. The note tie holds on both. The arithmetic has nothing to say
about which one the question meant, because nothing is wrong with either reading.

So the checks are a **gate after the choice**, not a judge of it. They eliminate misparses,
mis-scaled columns and OCR damage — which is what they were measured at 93% doing. They do
not eliminate the wrong indicator.

What actually disambiguates is match specificity and the printed pointer. Measured on 120
questions with model-generated label variants, ranking candidates by how exactly the label
matches:

| top match tier | unique Mã số | still ambiguous |
|---|---|---|
| label equals the variant | 10 | 0 |
| variant contained in label | 4 | 3 |
| label contained in variant | 20 | **27** |
| token overlap only | 5 | 1 |
| no statement row matched at all | — | 50 |

**32% land on exactly one Mã số**, against 20% with plain matching. The ambiguity is
concentrated in one tier: a long variant swallows a short label like `Tiền` or `Cộng`. The
principled fix is to rank by the length of the matched label rather than by tier alone — not
a stop-list of short words, which would be the patch.

## The corrected shape

    A  spec        model, question only: indicator + label variants + operation + period
    B  locate      code: metadata filter, navigate the Thuyết minh pointer,
                   rank candidates by match specificity        → unique 32% of the time
    C  choose      model, ONLY when 2–4 candidates remain: pick among those rows
                   — not among nine tables, which is the difference between the
                     organisers' 64% setting and their 87% one
    D  verify      code: identities, cross-year column, note tie. A gate: it drops a
                   wrong read, it does not pick between two right ones
    E  compute     code: the locked operation, one unit conversion, one rounding
    F  fallback    anything unresolved keeps its answer from the 0.415 vote

## Where the numbers stand, all free and none self-confirming

| | |
|---|---|
| note pointer resolves to the table it names | 92% |
| that note's total ties back to the statement line | 93% (control, wrong note: 11%) |
| documents with a parsed statement, once the scale gate is removed | 59% → 77% |
| questions where a reachable row matches the wording | 91% |
| questions landing on exactly one candidate | 32% |

The last two are the pair that matters: reach is not the constraint, disambiguation is. That
is the opposite of what this project assumed for three weeks, and it is why the retrieval
work — dense embeddings, rerankers, wider and narrower shortlists — bought nothing.

Total spend on establishing all of it: about two cents of API and no submissions.
