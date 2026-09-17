---
name: VIFin
description: Dark evidence workspace for a Vietnamese financial-QA pipeline
colors:
  bg: "#121212"
  surface: "#1C1C1C"
  surface-raised: "#262626"
  surface-hover: "#303030"
  hairline: "#2E2E2E"
  ink: "#F7F7F7"
  ink-dim: "#EBEBEB"
  muted: "#A8A8A8"
  accent: "#FF611A"
typography:
  answer:
    fontFamily: "'Doto', 'Fragment Mono', ui-monospace, monospace"
    fontSize: "clamp(40px, 6vw, 68px)"
    fontWeight: 700
    letterSpacing: "0.01em"
    lineHeight: 1
  heading:
    fontFamily: "'Manrope', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "26px"
    fontWeight: 500
    letterSpacing: "-0.02em"
    lineHeight: 1.2
  body:
    fontFamily: "'Manrope', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "'Manrope', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "12px"
    fontWeight: 500
    letterSpacing: "0.08em"
  data:
    fontFamily: "'Fragment Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
  pill: "9999px"
spacing:
  s-1: "4px"
  s-2: "8px"
  s-3: "12px"
  s-4: "16px"
  s-5: "20px"
  s-6: "24px"
  s-7: "32px"
  s-9: "44px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.bg}"
    rounded: "{rounded.sm}"
  evidence-card:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
  evidence-card-hover:
    backgroundColor: "{colors.surface-hover}"
  composer:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "12px"
---

# Design System: VIFin

## Overview

**Creative North Star: "The Workbench"**

A dark, quiet workspace where the answer and the evidence that produced it are
on screen at the same time, permanently. The ground is a near-black `#121212`
with three lifted greys above it, separated by hairlines rather than shadows.
One hot orange carries every interactive signal and nothing else.

The structural idea is that this is **not a chat transcript**. A transcript
stacks a question, a trace, an answer and fifty candidate tables vertically,
which forces a presenter to scroll past the answer to reach the proof. Here
the workspace splits: the answer column on the left, the evidence column on
the right, both always visible. Opening a table swaps the evidence column's
contents in place, so the answer beside it never moves.

Three faces divide the work strictly by what they can spell. Manrope carries
every Vietnamese string. Fragment Mono and Doto are Latin-only, so they only
ever receive digits and ASCII identifiers — Doto, a dot-matrix face, is
reserved for the single largest object on screen: the computed figure.

**Key Characteristics:**
- Two permanent panes; evidence is never behind a drawer or below the fold.
- Hairlines and flat greys carry structure. No shadows anywhere.
- One orange, used only for interaction, state and the cited cell.
- Type is assigned by script coverage, not by taste.
- The question is a command line, not a bubble.

## Colors

### Primary
- **Accent** (`#FF611A`): the only colour in the system. Focus rings, the
  composer border on focus, hovered card borders, source references, the
  running stage, the cited cell, the brand mark, the send button. Never
  decorative, and never a large fill outside the cited cell and send button.
- **Accent Quiet** (`rgba(255, 97, 26, .14)`): the accent's only wash — the
  hovered-reference background and the fixture banner ground.

### Neutral
- **BG** (`#121212`): the page and the answer pane.
- **Surface** (`#1C1C1C`): the evidence pane and the composer.
- **Surface Raised** (`#262626`) / **Surface Hover** (`#303030`): evidence
  cards at rest and under the pointer.
- **Hairline** (`#2E2E2E`): every division in the interface.
- **Ink** (`#F7F7F7`) / **Ink Dim** (`#EBEBEB`): primary text and answers.
- **Muted** (`#A8A8A8`): labels, metadata, stage status, table cells. 8.0:1
  on the ground.

### Named Rules
**The One-Orange Rule.** If an element wants a colour, it wants attention, and
attention here is orange or nothing. There is no success green, no warning
amber, no danger red. A failed stage is orange and says the word.

**The Hairline Rule.** Structure is a 1px `#2E2E2E` division. Shadows do not
exist in this system.

## Typography

**Manrope** (self-hosted, 400/500/700, Latin + Vietnamese) carries all prose,
headings, labels and any string containing Vietnamese. This is not a
preference: the corpus is Vietnamese on every screen, and Manrope is the only
face here with the diacritic coverage.

**Fragment Mono** (self-hosted, 400, Latin) carries identifiers, scores, source
numbers, line numbers and numeric table columns.

**Doto** (self-hosted, 600/700, Latin) is a dot-matrix face used once per
screen: the computed answer. It reads as an instrument readout, which is what
the figure is.

### Hierarchy
- **Answer** (Doto 700, `clamp(40px, 6vw, 68px)`): the computed figure.
- **Heading** (Manrope 500, 26px, `-0.02em`): the welcome line.
- **Body** (Manrope 400, 14px, 1.6, max 62ch): questions, prose, card titles.
- **Label** (Manrope 500, 12px, `0.08em`, uppercase): pane heads, section keys.
- **Data** (Fragment Mono 400, 12px): scores, identifiers, numeric cells.

### Named Rules
**The Script Rule.** A face is chosen by what it can spell. Fragment Mono and
Doto never receive Vietnamese text, because they cannot render its diacritics —
any string that might carry a tone mark is Manrope's.

## Layout

Three regions: a 208px rail, an answer pane, and an evidence pane at
`minmax(320px, 34%)`. Both panes scroll independently and are always present.
The composer is pinned to the bottom of the answer pane, so the vertical space
between the answer and the input never grows empty.

The question renders as a command line at the top of the answer pane, the
answer beneath it, and the execution trace beneath that — the trace is
deliberately last, since it explains a result the reader already has.

Below 1100px the panes stack, evidence taking 42% of the height. Below 860px
the rail becomes an overlay.

## Elevation & Depth

No shadows. Depth is four flat grounds — `bg`, `surface`, `surface-raised`,
`surface-hover` — divided by hairlines. The only Z movement is a 1px lift on a
hovered evidence card.

## Shapes

Four radii: 6px on small controls and chips, 10px on evidence cards, 14px on
the composer, and a full pill on nothing but the skip link. Tables and panes
are square.

## Components

### Evidence pane
The product's centre of gravity and a permanent column. It holds a responsive
card grid at rest; opening a card replaces the grid with that table, its
metadata and its calculation, with a back control in the pane head. The answer
never reflows when this happens.

### Evidence card
A raised panel with a hairline: head carrying the source number and the
reranker's real score, a live preview of the actual table's first rows, then
title and metadata. Selected tables sort ahead of candidates, and a streaming
update merges by table id so cards already on screen do not reshuffle.

### Answer
A muted context line, the figure in Doto at display scale with its unit
beside it, then source references and a verification tag. The tag appears only
when the query actually executed.

### Execution trace
A hairline, a heading row, then numbered stages. The running stage is orange
and carries a three-frame dot spinner. Only the current stage is expanded.

### Composer
A `surface` field with a hairline that turns orange on focus, an orange prompt
mark, and an orange send button. Pinned, never floating.

## Do's and Don'ts

### Do:
- **Do** keep both panes visible. Evidence behind a drawer defeats the product.
- **Do** send every Vietnamese string to Manrope. See **The Script Rule**.
- **Do** right-align numeric columns and set them in Fragment Mono.
- **Do** merge streaming evidence by table id so visible cards do not jump.

### Don't:
- **Don't** add a second accent colour. See **The One-Orange Rule**.
- **Don't** add a shadow; structure is hairlines.
- **Don't** put Vietnamese text in Doto or Fragment Mono — the glyphs are not
  in the subsets and it will silently fall back mid-string.
- **Don't** return the layout to a chat transcript.

---

*This world replaced the fonts.ninja light system on a pinned brief pointing at
midlife.engineering, whose palette (`#FF611A` on `#121212`/`#262626`) and three
faces (Doto, Fragment Mono, Manrope) this system adopts. The layout is not
from that source: it answers a separate request to find a better arrangement
for a chatbot, and the transcript was replaced with a two-pane workspace
because this product's turns are few and their payloads are large. Doto and
Fragment Mono ship Latin-only subsets, so their use is restricted to digits and
identifiers rather than being applied by role.*
