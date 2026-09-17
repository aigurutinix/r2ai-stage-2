"""Dump the organisers' 70 Hard Cube question templates to plain text for review.

`template_intents.py` calls a `_intent(...)` wrapper 70 times, so the specs are
positional arguments rather than a literal dict; the wrapper's own signature is
read here to map positions to names instead of assuming an order. Metrics come
through a second wrapper, `_m(role, key, formula)`.

Read with `ast`, never imported and never transcribed: the point of the file is
to be checkable against the source, which a hand-copied version would not be.

Usage:  python scripts/dump_official_templates.py
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "vifinqa-official" / "src" / "vifinqa" / "generation"
          / "hard" / "template_intents.py")
OUT = ROOT / "TEMPLATE-BTC.txt"

# `_intent` defaults, so an omitted argument reports its real value rather
# than a dash that reads as "unknown".
DEFAULTS = {"state": "NOT_IMPLEMENTED (mặc định)"}


def name_of(node: ast.AST, aliases: dict[str, str] | None = None) -> str | None:
    """`UniverseKind.INDUSTRY` -> 'INDUSTRY'; plain names and strings pass through.

    The registry defines one-letter aliases (`I = UniverseKind.INDUSTRY`) for the
    table to stay narrow, so a bare Name has to be resolved through them or the
    dump reports "I" and "NS" — which is exactly the sort of unreadable output
    that makes a generated file useless for checking.
    """

    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return (aliases or {}).get(node.id, node.id)
    if isinstance(node, ast.Constant):
        return str(node.value)
    return None


def module_aliases(tree: ast.AST) -> dict[str, str]:
    """Module-level `NAME = SomeEnum.MEMBER` bindings, as NAME -> MEMBER."""

    found: dict[str, str] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Attribute)):
            found[node.targets[0].id] = node.value.attr
    return found


def literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def sequence(node: ast.AST) -> list:
    """A tuple/list literal whose members may be enum attributes, not constants."""

    if isinstance(node, (ast.Tuple, ast.List)):
        return [literal(e) if literal(e) is not None else name_of(e) for e in node.elts]
    value = literal(node)
    return list(value) if isinstance(value, (tuple, list)) else []


def metrics_of(node: ast.AST) -> list[tuple]:
    calls = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_m"
    ]
    return [tuple(literal(a) for a in call.args) for call in calls]


def parameter_names(tree: ast.AST, function: str) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            args = node.args
            return [a.arg for a in args.posonlyargs + args.args]
    return []


def wrap(text: str, width: int = 88, indent: str = " " * 25) -> str:
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return f"\n{indent}".join(lines)


def main() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    aliases = module_aliases(tree)
    positional = parameter_names(tree, "_intent")
    intents = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_intent"
    ]

    rows = []
    for call in intents:
        spec: dict = {}
        for index, arg in enumerate(call.args):
            if index < len(positional):
                spec[positional[index]] = arg
        for keyword in call.keywords:
            if keyword.arg:
                spec[keyword.arg] = keyword.value
        rows.append(spec)

    out: list[str] = []
    add = out.append
    add("70 TEMPLATE CÂU HỎI TẦNG HARD — TRÍCH TỪ MÃ NGUỒN CỦA BAN TỔ CHỨC")
    add("=" * 100)
    add("")
    add(f"Nguồn: {SOURCE.relative_to(ROOT).as_posix()}")
    add("Trích tự động bằng ast, không chép tay. Sinh lại: python scripts/dump_official_templates.py")
    add("")
    add("Docstring gốc của file:")
    doc = ast.get_docstring(tree) or ""
    for line in doc.strip().splitlines():
        add(f"    {line}")
    add("")
    add("Cách đọc: `operator_sequence` là chuỗi phép toán từ dữ liệu thô tới đáp án. `terminal` là")
    add("phép cuối cùng quyết định kiểu đáp án. `offsets` là các năm tương đối cần lấy — (0,) nghĩa")
    add("là một năm, (-1, 0) nghĩa là cần cả năm trước. `min_entities` là số công ty tối thiểu.")
    add("")

    # ---- summary tables --------------------------------------------------
    add("")
    add("TỔNG HỢP")
    add("-" * 100)

    for field, title in (
        ("terminal", "Phép cuối (terminal)"),
        ("unit", "Đơn vị đáp án"),
        ("universe", "Phạm vi lựa chọn (universe)"),
        ("state", "Trạng thái cài đặt"),
    ):
        counter: Counter = Counter()
        for spec in rows:
            counter[name_of(spec.get(field), aliases) or DEFAULTS.get(field, "-")] += 1
        add("")
        add(f"  {title}:")
        for key, count in counter.most_common():
            add(f"    {count:3d}  {key}")

    operations: Counter = Counter()
    for spec in rows:
        for op in sequence(spec.get("operations")):
            operations[op] += 1
    add("")
    add("  Phép toán, theo số template dùng tới:")
    for key, count in operations.most_common():
        add(f"    {count:3d}  {key}")

    lengths: Counter = Counter(len(sequence(spec.get("operations"))) for spec in rows)
    add("")
    add("  Độ dài chuỗi phép toán:")
    for key in sorted(lengths):
        add(f"    {lengths[key]:3d} template có {key} bước")

    # ---- one block per template -----------------------------------------
    add("")
    add("")
    add("CHI TIẾT 70 TEMPLATE")
    add("=" * 100)
    for spec in rows:
        add("")
        add(f"[{literal(spec.get('template_id'))}]")
        add(f"    universe      : {name_of(spec.get('universe'), aliases)}")
        ops = sequence(spec.get("operations"))
        add(f"    chuỗi phép    : {' -> '.join(str(o) for o in ops)}")
        add(f"    terminal      : {name_of(spec.get('terminal'), aliases)}"
            f"   (đơn vị: {name_of(spec.get('unit'), aliases)})")
        offsets = sequence(spec.get("offsets")) or [0]
        add(f"    offsets năm   : {tuple(offsets)}"
            f"   min_entities: {literal(spec.get('minimum_entities')) or 3}")
        add("    chỉ tiêu:")
        for role, key, formula in metrics_of(spec.get("metrics")) or []:
            add(f"      - {str(role):18s} key={key!r} formula={formula!r}")
        for field, title in (("limits", "giới hạn diễn giải"),
                             ("gates", "cổng kiểm tra riêng"),
                             ("special_sources", "nguồn đặc biệt")):
            for item in sequence(spec.get(field)):
                add(f"    {title:20s}: {wrap(item)}")
        state = name_of(spec.get("state"), aliases) or DEFAULTS["state"]
        if state:
            add(f"    trạng thái    : {state}")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"đã ghi {OUT}  ({len(rows)} template, {len(out)} dòng)")


if __name__ == "__main__":
    main()
