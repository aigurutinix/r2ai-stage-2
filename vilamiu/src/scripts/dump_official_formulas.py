"""Dump the organisers' formula definitions to a plain-text file for review.

Everything is read out of `vifinqa-official/` with `ast`, never by hand and never
by importing (the package's `__init__` pulls in the CLI and needs `bm25s`).
Transcribing these by hand would be the obvious way to introduce an error into
exactly the numbers we are trying to get right, so the file is generated.

The reason this matters: our `ratio.py` defines ROA and ROE against the closing
balance, while the organisers divide by the *average* of the opening and closing
balance. Answer tolerance is 0.02% relative, so that is not a rounding difference
— every ROA and ROE answer we ship is wrong.

Usage:  python scripts/dump_official_formulas.py
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "vifinqa-official" / "src" / "vifinqa"
OUT = ROOT / "CONG-THUC-BTC.txt"


def literal(node: ast.AST):
    """Best-effort constant folding; returns None for anything computed."""

    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def keywords(call: ast.Call) -> dict:
    return {k.arg: k.value for k in call.keywords if k.arg}


def calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


def parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def wrap(text: str, width: int = 92, indent: str = "      ") -> str:
    words, lines, line = text.split(), [], ""
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
    out: list[str] = []
    add = out.append

    add("CÔNG THỨC VÀ TỪ VỰNG CHỈ TIÊU — TRÍCH TỪ MÃ NGUỒN CỦA BAN TỔ CHỨC")
    add("=" * 92)
    add("")
    add("Nguồn: vifinqa-official/src/vifinqa/generation/")
    add("Trích tự động bằng ast, không chép tay. Sinh lại: python scripts/dump_official_formulas.py")
    add("")
    add("Đây là dữ liệu BTC phát hành kèm benchmark. Nếu dùng thì PHẢI trích dẫn trong working notes.")
    add("")

    # ---- intermediate_formulas/registry.py -------------------------------
    add("")
    add("1. CÔNG THỨC TỶ SỐ  (intermediate_formulas/registry.py)")
    add("-" * 92)
    tree = parse(OFFICIAL / "generation" / "intermediate_formulas" / "registry.py")
    for call in calls_named(tree, "FormulaDefinition"):
        kw = keywords(call)
        add("")
        add(f"[{literal(kw.get('formula_id'))}]  {literal(kw.get('name_vi'))}")
        add(f"    Công thức : {literal(kw.get('formula_text'))}")
        add(f"    Đơn vị    : {literal(kw.get('unit'))}   "
            f"(answer_type = {literal(kw.get('answer_type'))})")
        add("    Thành phần cần lấy:")
        roles = kw.get("required_roles")
        if roles is not None:
            for role in calls_named(roles, "FormulaRole"):
                rkw = keywords(role)
                names = literal(rkw.get("concept_names")) or ()
                add(f"      - {literal(rkw.get('role_id')):22s} nhãn: {literal(rkw.get('label_vi'))}")
                for name in names:
                    add(f"        {'':22s} khớp nhãn bảng: {name!r}")
        for field, title in (
            ("time_basis", "Quy tắc kỳ  "),
            ("denominator_policy", "Quy tắc mẫu "),
            ("sign_policy", "Quy tắc dấu "),
            ("industry_policy", "Loại trừ    "),
        ):
            value = literal(kw.get(field))
            if value:
                add(f"    {title}: {wrap(str(value))}")
        urls = literal(kw.get("source_urls")) or ()
        for url in urls:
            add(f"    Nguồn     : {url}")
        add(f"    enabled   : {literal(kw.get('enabled'))}")

    # ---- intermediate_formulas/longitudinal.py ---------------------------
    add("")
    add("")
    add("2. CHỈ TIÊU ĐẦU VÀO CHO CÔNG THỨC THEO THỜI GIAN  (intermediate_formulas/longitudinal.py)")
    add("-" * 92)
    tree = parse(OFFICIAL / "generation" / "intermediate_formulas" / "longitudinal.py")
    for call in calls_named(tree, "LongitudinalInputMetric"):
        kw = keywords(call)
        names = literal(kw.get("concept_names")) or ()
        add("")
        add(f"[{literal(kw.get('metric_id'))}]  {literal(kw.get('label_vi'))}"
            f"   (kind = {literal(kw.get('kind'))}, enabled = {literal(kw.get('enabled'))})")
        for name in names:
            add(f"      khớp nhãn bảng: {name!r}")

    add("")
    add("")
    add("3. PHÉP BIẾN ĐỔI THEO THỜI GIAN  (cùng file)")
    add("-" * 92)
    for call in calls_named(tree, "TransformDefinition"):
        kw = keywords(call)
        add("")
        add(f"[{literal(kw.get('transform_id'))}]  {literal(kw.get('name_vi'))}")
        for field in ("formula_text", "unit", "answer_type", "period_count", "notes"):
            value = literal(kw.get(field))
            if value not in (None, ""):
                add(f"    {field:14s}: {wrap(str(value))}")

    # ---- archetypes ------------------------------------------------------
    add("")
    add("")
    add("4. ARCHETYPE CÂU HỎI TẦNG HARD  (generation/hard/recipe/archetype.py)")
    add("-" * 92)
    source = (OFFICIAL / "generation" / "hard" / "recipe" / "archetype.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and node.value.value not in found:
                found.append(node.value.value)
    add("")
    for name in found:
        add(f"    - {name}")
    add("")
    add("    Nhóm phép gộp (_REDUCERS) và biến đổi (_TRANSFORMS) khai báo ngay đầu file:")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("_TRANSFORMS", "_REDUCERS"):
                value = literal(node.value.args[0]) if isinstance(node.value, ast.Call) else None
                add(f"      {node.targets[0].id} = {sorted(value) if value else '?'}")

    # ---- operation classes ----------------------------------------------
    add("")
    add("")
    add("5. LỚP PHÉP TOÁN CÀI ĐẶT SẴN  (generation/hard/recipe/operations/*.py)")
    add("-" * 92)
    ops_dir = OFFICIAL / "generation" / "hard" / "recipe" / "operations"
    for path in sorted(ops_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        names = [
            n.name for n in ast.walk(parse(path))
            if isinstance(n, ast.ClassDef) and n.name.endswith("Operation")
        ]
        if names:
            add("")
            add(f"    {path.name}")
            for name in names:
                add(f"      - {name}")

    # ---- cohort constraints ---------------------------------------------
    add("")
    add("")
    add("6. RÀNG BUỘC NHÓM SO SÁNH  (generation/hard/recipe/grounded/domains.py)")
    add("-" * 92)
    add("")
    tree = parse(OFFICIAL / "generation" / "hard" / "recipe" / "grounded" / "domains.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                add(f"    {name:38s} = {literal(node.value)}")

    # ---- account codes ---------------------------------------------------
    add("")
    add("")
    add("7. MÃ SỐ CHỈ TIÊU THEO THÔNG TƯ 200  (generation/panel/catalog.py)")
    add("-" * 92)
    add("")
    add("    Đây là thứ đáng giá nhất trong cả gói: BTC định danh chỉ tiêu bằng MÃ SỐ, không phải")
    add("    bằng nhãn. Báo cáo tài chính Việt Nam có cột 'Mã số' và mã đó là chuẩn pháp lý, nên")
    add("    khớp theo mã là khớp CHÍNH XÁC — không phải khớp nhãn mờ vốn đang chặn ta ở 42%.")
    add("")
    catalog = parse(OFFICIAL / "generation" / "panel" / "catalog.py")
    kinds = {
        "_KQKD_METRIC_NAMES": "kqkd — Báo cáo kết quả hoạt động kinh doanh",
        "_CDKT_METRIC_NAMES": "cdkt — Bảng cân đối kế toán",
        "_LCTT_METRIC_NAMES": "lctt — Báo cáo lưu chuyển tiền tệ",
    }
    codes: dict[str, str] = {}
    for node in catalog.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            title = kinds.get(node.target.id)
            if title and node.value is not None:
                table = literal(node.value) or {}
                add(f"    {title}")
                prefix = node.target.id.split("_")[1].lower()
                for code, name in table.items():
                    add(f"      {prefix}:{code:<5s} {name}")
                    codes[f"{prefix}:{code}"] = name
                add("")

    # ---- ratio catalogue -------------------------------------------------
    add("")
    add("8. 23 TỶ SỐ ĐỊNH NGHĨA BẰNG MÃ SỐ  (cùng file)")
    add("-" * 92)
    add("")
    add("    Tử và mẫu là tổ hợp tuyến tính các mã số: ((hệ số, mã), ...).")

    def terms(node: ast.AST) -> str:
        parts = []
        for coefficient, key in (literal(node) or ()):
            sign = "" if coefficient == 1.0 else f"{coefficient:g}·"
            parts.append(f"{sign}{key} [{codes.get(key, '?')}]")
        return " + ".join(parts) if parts else "?"

    for call in calls_named(catalog, "RatioDefinition"):
        kw = keywords(call)
        add("")
        add(f"    [{literal(kw.get('key'))}]  {literal(kw.get('name'))}"
            f"   ({literal(kw.get('value_kind'))})")
        add(f"        tử  : {terms(kw.get('numerator'))}")
        add(f"        mẫu : {terms(kw.get('denominator'))}")

    # ---- measurements ----------------------------------------------------
    add("")
    add("")
    add("9. CHỈ SỐ ĐO PHỨC TẠP  (generation/panel/measurements/catalog.py)")
    add("-" * 92)
    add("")
    add("    Mỗi lớp là một chỉ số cần nhiều hơn một phép chia. `period_basis` cho biết cần")
    add("    một năm hay hai năm liền kề.")
    measurements = parse(OFFICIAL / "generation" / "panel" / "measurements" / "catalog.py")
    for node in measurements.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = {
            t.targets[0].id: literal(t.value)
            for t in node.body
            if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
        }
        fields.update({
            t.target.id: literal(t.value)
            for t in node.body
            if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name) and t.value
        })
        if "measurement_id" not in fields:
            continue
        used = sorted({
            n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and ":" in n.value
            and n.value.split(":")[0] in ("kqkd", "cdkt", "lctt")
        })
        add("")
        add(f"    [{fields.get('measurement_id')}]  {fields.get('name')}")
        add(f"        đơn vị: {fields.get('unit')}   kỳ: {fields.get('period_basis')}")
        for key in used:
            add(f"        dùng  : {key:<10s} {codes.get(key, '?')}")

    # ---- our side --------------------------------------------------------
    add("")
    add("")
    add("10. ĐỐI CHIẾU VỚI src/vifin/answering/ratio.py CỦA TA")
    add("-" * 92)
    add("")
    add("    ROA   ta : Lợi nhuận sau thuế / Tổng cộng tài sản  (SỐ CUỐI NĂM)")
    add("          BTC: Lợi nhuận sau thuế / BÌNH QUÂN tổng tài sản (đầu năm + cuối năm)/2")
    add("          -> LỆCH. Dung sai 0,02% nên đây là SAI, không phải sai số.")
    add("")
    add("    ROE   ta : Lợi nhuận sau thuế / Vốn chủ sở hữu  (SỐ CUỐI NĂM)")
    add("          BTC: Lợi nhuận sau thuế / BÌNH QUÂN vốn chủ sở hữu (đầu năm + cuối năm)/2")
    add("          -> LỆCH, cùng lý do.")
    add("")
    add("    Hệ số thanh toán nhanh")
    add("          ta : KHÔNG CÓ")
    add("          BTC: (Tài sản ngắn hạn - Hàng tồn kho) / Nợ ngắn hạn")
    add("          -> THIẾU. 15 câu hỏi nhắc tới nó, đang rơi vào fallback/plan/none.")
    add("")
    add("    Ta có thêm mà BTC không định nghĩa ở file này (tự suy, chưa đối chiếu được):")
    add("          biên lợi nhuận gộp, biên lợi nhuận ròng/ROS, thanh toán hiện hành, hệ số nợ")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"đã ghi {OUT}  ({len(out)} dòng)")


if __name__ == "__main__":
    main()
