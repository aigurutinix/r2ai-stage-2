"""Build multi-fact formula candidates for ViFinQA questions q0656-q0732."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shutil
import sqlite3
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .l1_fact_layer import (
    FactCandidate,
    FactRetriever,
    ManualOverride,
    ParsedQuestion,
    load_companies,
    normalize,
    pandas_query_for,
    replay_generated_query,
    resolve_company,
    write_source_table_csv,
)


FIRST_ID = 656
LAST_ID = 732


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    metric_text: str
    target_unit: str
    year: int
    scope: str = "consolidated"
    period_kind: str = "end_or_flow"

    @property
    def metric_tokens(self) -> tuple[str, ...]:
        return tuple(normalize(self.metric_text).split())


@dataclass(frozen=True)
class FormulaSpec:
    question_id: int
    formula: str
    components: tuple[ComponentSpec, ...]


@dataclass(frozen=True)
class FormulaOverride:
    question_id: int
    component: str
    year: int
    document_id: str
    source_line_1: int
    row_index: int
    column_index: int
    raw_value: str
    review_note: str


def C(name: str, metric: str, year: int, scope: str = "consolidated",
      period: str = "end_or_flow", unit: str = "VND_1") -> ComponentSpec:
    return ComponentSpec(name, metric, unit, year, scope, period)


def F(question_id: int, formula: str, *components: ComponentSpec) -> FormulaSpec:
    return FormulaSpec(question_id, formula, tuple(components))


# The formula is deliberately explicit rather than inferred from loose words.
# Symbols map one-to-one to the component specs below.
FORMULA_SPECS = {
    656: F(656, "a / b * 100.0", C("a", "cac khoan tuong duong tien", 2024), C("b", "tong tai san", 2024)),
    657: F(657, "a / b / 1000.0", C("a", "loi nhuan sau thue", 2020, "separate"), C("b", "so luong co phieu dang luu hanh", 2020, "separate", unit="shares")),
    658: F(658, "a / b * 100.0", C("a", "phan lai trong cong ty lien ket lien doanh", 2024), C("b", "loi nhuan sau thue thu nhap doanh nghiep", 2024)),
    659: F(659, "-a / b * 100.0", C("a", "co tuc loi nhuan da tra cho chu so huu", 2024), C("b", "loi nhuan sau thue", 2024)),
    660: F(660, "(a + b) / c * 100.0", C("a", "nghia vu tiem an", 2022, "separate"), C("b", "cac cam ket dua ra", 2022, "separate"), C("c", "tong tai san", 2022, "separate")),
    661: F(661, "-a / b * 100.0", C("a", "tien chi mua sam xay dung tai san co dinh", 2021, "separate"), C("b", "tong nguyen gia tai san co dinh huu hinh cuoi nam", 2021, "separate")),
    662: F(662, "a / b * 100.0", C("a", "loi nhuan sau thue", 2016), C("b", "tong tai san", 2016)),
    663: F(663, "(a + b) / c * 100.0", C("a", "vay va no thue tai chinh ngan han", 2022, "separate"), C("b", "vay va no thue tai chinh dai han", 2022, "separate"), C("c", "tong tai san", 2022, "separate")),
    664: F(664, "-a / b * 100.0", C("a", "tong gia tri hao mon luy ke tai san co dinh huu hinh", 2024), C("b", "tong nguyen gia tai san co dinh huu hinh", 2024)),
    665: F(665, "a / b * 100.0", C("a", "tai san co dinh vo hinh", 2016), C("b", "tong tai san", 2016)),
    666: F(666, "a - b", C("a", "doanh thu hoat dong tai chinh", 2015, "separate", unit="VND_1e9"), C("b", "chi phi tai chinh", 2015, "separate", unit="VND_1e9")),
    667: F(667, "a / b * 100.0", C("a", "tong nguyen gia tai san co dinh huu hinh", 2023, "separate"), C("b", "tong tai san", 2023, "separate")),
    668: F(668, "a / b * 100.0", C("a", "co tuc loi nhuan duoc chia", 2023, "separate"), C("b", "tong gia tri dau tu", 2023, "separate")),
    669: F(669, "a / b * 100.0", C("a", "phai tra nguoi ban ngan han", 2019), C("b", "tong no ngan han", 2019)),
    670: F(670, "a / b * 100.0", C("a", "vay ngan hang ngan han", 2020, "separate", "start"), C("b", "tong no phai tra tai chinh", 2020, "separate", "start")),
    671: F(671, "-a / b * 100.0", C("a", "chi phi du phong rui ro tin dung", 2020), C("b", "loi nhuan truoc thue", 2020)),
    672: F(672, "a / b", C("a", "chi phi tra truoc ngan han", 2022), C("b", "chi phi tra truoc dai han", 2022)),
    673: F(673, "a / b * 100.0", C("a", "tong tai san ngan han", 2025, "separate"), C("b", "tong nguon von", 2025, "separate")),
    674: F(674, "a / b * 100.0", C("a", "loi nhuan sau thue", 2016), C("b", "doanh thu thuan", 2016)),
    675: F(675, "a / b * 100.0", C("a", "dau tu tai chinh dai han", 2021, "separate"), C("b", "von chu so huu", 2021, "separate")),
    676: F(676, "(a + b) / c * 100.0", C("a", "trich lap du phong chung cho vay khach hang", 2017, "separate"), C("b", "trich lap du phong cu the cho vay khach hang", 2017, "separate"), C("c", "loi nhuan thuan sau thue", 2017, "separate")),
    677: F(677, "a - b", C("a", "doanh thu hoat dong tai chinh", 2017, "separate", unit="VND_1e11"), C("b", "chi phi tai chinh", 2017, "separate", unit="VND_1e11")),
    678: F(678, "a / b", C("a", "tong no ngan han", 2025, "separate"), C("b", "von chu so huu", 2025, "separate")),
    679: F(679, "a - b", C("a", "doanh thu thuan", 2016, unit="VND_1e9"), C("b", "gia von hang ban", 2016, unit="VND_1e9")),
    680: F(680, "a / b * 100.0", C("a", "lai tien gui", 2024, "separate"), C("b", "tong doanh thu hoat dong tai chinh", 2024, "separate")),
    681: F(681, "a + b", C("a", "doanh thu thuan", 2019, "separate", unit="VND_1e9"), C("b", "gia von hang ban", 2019, "separate", unit="VND_1e9")),
    682: F(682, "a - b", C("a", "thu nhap khac", 2021, unit="VND_1e6"), C("b", "chi phi khac", 2021, unit="VND_1e6")),
    683: F(683, "(a - b) / 1000000.0", C("a", "thu nhap tu hoat dong dich vu", 2015, unit="VND_1e9"), C("b", "chi phi hoat dong dich vu", 2015, unit="VND_1e9")),
    684: F(684, "a / b * 100.0", C("a", "du phong chung cho vay khach hang", 2016, "separate"), C("b", "tong du phong rui ro cho vay khach hang", 2016, "separate")),
    685: F(685, "a / (b + c + d) * 100.0", C("a", "tong du phong rui ro cho vay khach hang", 2025, "separate"), C("b", "no duoi tieu chuan", 2025, "separate"), C("c", "no nghi ngo", 2025, "separate"), C("d", "no co kha nang mat von", 2025, "separate")),
    686: F(686, "a / b * 100.0", C("a", "loi nhuan truoc thue", 2023), C("b", "doanh thu thuan", 2023)),
    687: F(687, "a / b * 100.0", C("a", "tai san co dinh huu hinh", 2018), C("b", "tong tai san co dinh", 2018)),
    688: F(688, "a / (a + b) * 100.0", C("a", "phai thu ngan han khac", 2018), C("b", "phai thu dai han khac", 2018)),
    689: F(689, "(b - a) / a * 100.0", C("a", "tai san phai sinh", 2021, "separate"), C("b", "tai san phai sinh", 2022, "separate")),
    690: F(690, "a / b", C("a", "tong no vay", 2020, "separate"), C("b", "tong tien mat va tien gui ngan hang", 2020, "separate")),
    691: F(691, "a / b * 100.0", C("a", "doanh thu ure phu my", 2018), C("b", "tong doanh thu thuan hang hoa san xuat trong nuoc", 2018)),
    692: F(692, "-a / b * 100.0", C("a", "tong du phong rui ro cho vay khach hang", 2018, "separate"), C("b", "du no cho vay khach hang", 2018, "separate")),
    693: F(693, "a / (a + b) * 100.0", C("a", "tai san co dinh vo hinh", 2025, "separate"), C("b", "tai san co dinh huu hinh", 2025, "separate")),
    694: F(694, "a / b * 100.0", C("a", "tong no phai tra", 2015, "separate"), C("b", "tong tai san", 2015, "separate")),
    695: F(695, "a * 2.0 / (b + c) * 100.0", C("a", "loi nhuan sau thue", 2017), C("b", "von chu so huu cuoi nam", 2017), C("c", "von chu so huu dau nam", 2017, period="start")),
    696: F(696, "a * 2.0 / (b + c)", C("a", "doanh thu thuan", 2019), C("b", "von chu so huu cuoi nam", 2019), C("c", "von chu so huu dau nam", 2019, period="start")),
    697: F(697, "a / b * 100.0", C("a", "dau tu vao techcombank", 2016, "separate"), C("b", "tong dau tu vao cong ty lien ket", 2016, "separate")),
    698: F(698, "a / b * 100.0", C("a", "chi phi quan ly doanh nghiep", 2020), C("b", "doanh thu thuan", 2020)),
    699: F(699, "a / b * 100.0", C("a", "tai san co dinh huu hinh", 2025), C("b", "tong tai san", 2025)),
    700: F(700, "a / b * 100.0", C("a", "lai vay phai tra", 2015), C("b", "du no vay dai han", 2015)),
    701: F(701, "a / b * 100.0", C("a", "chi phi ban hang", 2020, "separate"), C("b", "doanh thu thuan", 2020, "separate")),
    702: F(702, "a - b", C("a", "thu nhap khac", 2018, "separate", unit="VND_1e9"), C("b", "chi phi khac", 2018, "separate", unit="VND_1e9")),
    703: F(703, "a + b + c - d - e - f", C("a", "phai thu khach hang ngan han ben lien quan", 2024, "separate", unit="VND_1e9"), C("b", "phai thu cho vay ngan han ben lien quan", 2024, "separate", unit="VND_1e9"), C("c", "phai thu ngan han khac ben lien quan", 2024, "separate", unit="VND_1e9"), C("d", "phai tra nguoi ban ngan han ben lien quan", 2024, "separate", unit="VND_1e9"), C("e", "phai tra ngan han khac ben lien quan", 2024, "separate", unit="VND_1e9"), C("f", "vay ngan han ben lien quan", 2024, "separate", unit="VND_1e9")),
    704: F(704, "a - b", C("a", "doanh thu hoat dong tai chinh", 2015, "separate", unit="VND_1e9"), C("b", "chi phi tai chinh", 2015, "separate", unit="VND_1e9")),
    705: F(705, "a / b * 100.0", C("a", "vay va no thue tai chinh ngan han", 2022), C("b", "von chu so huu", 2022)),
    706: F(706, "-a * 1000000.0 / b * 100.0", C("a", "du phong giam gia chung khoan dau tu san sang de ban", 2025), C("b", "tong chung khoan dau tu san sang de ban", 2025)),
    707: F(707, "-a / b * 100.0", C("a", "du phong no phai thu khach hang kho doi", 2024), C("b", "tong phai thu khach hang", 2024)),
    708: F(708, "a / b * 100.0", C("a", "loi nhuan sau thue", 2017), C("b", "doanh thu thuan", 2017)),
    709: F(709, "a / b * 100.0", C("a", "chi phi lai vay", 2022), C("b", "no vay ngan han", 2022)),
    710: F(710, "a / b * 100.0", C("a", "tong no ngan han", 2022, "separate"), C("b", "von chu so huu", 2022, "separate")),
    711: F(711, "a / b * 100.0", C("a", "doanh thu ban hang cho cac ben lien quan", 2017, "separate"), C("b", "tong doanh thu ban hang", 2017, "separate")),
    712: F(712, "-a / b * 100.0", C("a", "chi phi tai chinh", 2025), C("b", "doanh thu thuan", 2025)),
    713: F(713, "a / b * 100.0", C("a", "doanh thu hoat dong tai chinh", 2020, "separate"), C("b", "chi phi tai chinh", 2020, "separate")),
    714: F(714, "a - b", C("a", "doanh thu hoat dong tai chinh", 2024, unit="VND_1e9"), C("b", "chi phi tai chinh", 2024, unit="VND_1e9")),
    715: F(715, "a / b * 100.0", C("a", "dau tu ngan han", 2023, "separate"), C("b", "tien va cac khoan tuong duong tien", 2023, "separate")),
    716: F(716, "a / b * 100.0", C("a", "chi phi lai vay", 2025, "separate"), C("b", "no vay dai han", 2025, "separate")),
    717: F(717, "a / b * 100.0", C("a", "tong cam ket ngoai bang", 2019, "separate"), C("b", "tong tai san", 2019, "separate")),
    718: F(718, "-a / b * 100.0", C("a", "gia von hang ban", 2017), C("b", "doanh thu thuan", 2017)),
    719: F(719, "a / b * 100.0", C("a", "dong tien thuan tu hoat dong kinh doanh", 2016, "separate"), C("b", "loi nhuan truoc thue", 2016, "separate")),
    720: F(720, "(b - a) / a * 100.0", C("a", "gia tri con lai tai san co dinh vo hinh", 2017, "separate"), C("b", "gia tri con lai tai san co dinh vo hinh", 2018, "separate")),
    721: F(721, "a / b * 100.0", C("a", "gia tri dau tu vao co phieu", 2020), C("b", "dau tu vao cong ty lien doanh lien ket", 2020)),
    722: F(722, "a - b", C("a", "thu nhap khac", 2022, unit="VND_1e9"), C("b", "chi phi khac", 2022, unit="VND_1e9")),
    723: F(723, "a / (a + b) * 100.0", C("a", "cho vay dai han ben lien quan", 2018), C("b", "phai thu dai han khac ben lien quan", 2018)),
    724: F(724, "a / b * 100.0", C("a", "chi phi tai chinh", 2015, "separate"), C("b", "tong doanh thu", 2015, "separate")),
    725: F(725, "a + b", C("a", "doanh thu hoat dong tai chinh", 2017, "separate", unit="VND_1e9"), C("b", "chi phi tai chinh", 2017, "separate", unit="VND_1e9")),
    726: F(726, "a - b", C("a", "doanh thu hoat dong tai chinh", 2024, unit="VND_1e9"), C("b", "chi phi tai chinh", 2024, unit="VND_1e9")),
    727: F(727, "(b - a) / a * 100.0", C("a", "gia tri rui ro var mot ngay danh muc co phieu niem yet", 2018, "separate"), C("b", "gia tri rui ro var mot ngay danh muc co phieu niem yet", 2019, "separate")),
    728: F(728, "(a + b) / c * 100.0", C("a", "chi phi ban hang", 2018, "separate"), C("b", "chi phi quan ly doanh nghiep", 2018, "separate"), C("c", "doanh thu thuan", 2018, "separate")),
    729: F(729, "a / b * 100.0", C("a", "von gop co dong nguyen khai hoan", 2024), C("b", "tong von gop chu so huu", 2024)),
    730: F(730, "a - b", C("a", "doanh thu thuan", 2017, "separate", unit="VND_1e12"), C("b", "gia von hang ban", 2017, "separate", unit="VND_1e12")),
    731: F(731, "a / b * 100.0", C("a", "cho vay khach hang gop", 2025), C("b", "tong tien gui khach hang", 2025)),
    732: F(732, "a / b", C("a", "tong vay ngan han", 2023, "separate"), C("b", "tong tien mat va tien gui ngan hang", 2023, "separate")),
}


def load_questions(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if FIRST_ID <= int(row["id"]) <= LAST_ID:
                rows[int(row["id"])] = row
    if sorted(rows) != list(range(FIRST_ID, LAST_ID + 1)):
        raise ValueError("Formula question range is incomplete")
    if sorted(FORMULA_SPECS) != sorted(rows):
        raise ValueError("Formula registry does not cover every question")
    return rows


def component_question(question: dict, ticker: str, spec: ComponentSpec) -> ParsedQuestion:
    return ParsedQuestion(
        id=int(question["id"]),
        question=question["question"],
        ticker=ticker,
        matched_company_alias=ticker.lower(),
        year=spec.year,
        scope=spec.scope,
        period_kind=spec.period_kind,
        target_unit=spec.target_unit,
        metric_text=normalize(spec.metric_text),
        metric_tokens=spec.metric_tokens,
    )


def load_overrides(path: Optional[Path]) -> dict[tuple[int, str], FormulaOverride]:
    if path is None or not path.is_file():
        return {}
    required = {
        "question_id", "component", "year", "document_id", "source_line_1",
        "row_index", "column_index", "raw_value", "review_note",
    }
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Formula override columns must be {sorted(required)}")
        for row in reader:
            override = FormulaOverride(
                question_id=int(row["question_id"]), component=row["component"],
                year=int(row["year"]), document_id=row["document_id"],
                source_line_1=int(row["source_line_1"]), row_index=int(row["row_index"]),
                column_index=int(row["column_index"]), raw_value=row["raw_value"],
                review_note=row["review_note"],
            )
            key = (override.question_id, override.component)
            if key in result:
                raise ValueError(f"Duplicate formula override {key}")
            result[key] = override
    return result


def as_l1_override(override: FormulaOverride) -> ManualOverride:
    return ManualOverride(
        override.question_id, override.document_id, override.source_line_1,
        override.row_index, override.column_index, override.raw_value,
        override.review_note,
    )


def fact_record(candidate: FactCandidate, component: str, rank: int) -> dict:
    row = asdict(candidate)
    row["component"] = component
    row["candidate_rank"] = rank
    return row


def validate_formula(expression: str, names: set[str]) -> None:
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
               ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
               ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"Unsupported formula node {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"Unknown formula symbol {node.id}")


def evaluate_formula(expression: str, values: dict[str, float]) -> float:
    validate_formula(expression, set(values))
    return float(eval(compile(ast.parse(expression, mode="eval"), "<formula>", "eval"),
                      {"__builtins__": {}}, values))


def cell_query(fact: dict, variable: str) -> str:
    query = pandas_query_for(
        fact["raw_value"], int(fact["row_index"]), int(fact["column_index"]),
        float(fact["source_scale"]), float(fact["target_scale"]),
    )
    return query.replace("df1", variable, 1)


def formula_query_and_answer(plan: dict) -> tuple[str, float]:
    expression = plan["formula"]
    values = {}
    for index, (name, fact) in enumerate(plan["components"].items(), 1):
        expression = re.sub(
            rf"\b{re.escape(name)}\b", f"({cell_query(fact, f'df{index}')})",
            expression,
        )
        values[name] = float(fact["answer"])
    return expression, evaluate_formula(plan["formula"], values)


def load_plans(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        plans = [json.loads(line) for line in handle if line.strip()]
    if [int(plan["question"]["id"]) for plan in plans] != list(
        range(FIRST_ID, LAST_ID + 1)
    ):
        raise ValueError("A full formula build requires q0656-q0732 in order")
    for plan in plans:
        # The audited registry is authoritative; candidate files only persist
        # selected facts and may predate a formula-only scale correction.
        plan["formula"] = FORMULA_SPECS[int(plan["question"]["id"])].formula
        if not all(
            fact["selection_source"] == "manual"
            for fact in plan["components"].values()
        ):
            raise ValueError(
                f"q{plan['question']['id']}: every component must be manually reviewed"
            )
    return plans


def copy_base_submission(base_submission: Path, output_dir: Path) -> list[dict]:
    items = json.loads(base_submission.read_text(encoding="utf-8"))
    for item in items:
        for evidence in item["evidence"]:
            source = base_submission.parent / evidence["csv_path"]
            target = output_dir / evidence["csv_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return items


def build_submission(plans_path: Path, database: Path, base_submission: Path,
                     output_dir: Path, line_base: int,
                     temporal_pairs: Optional[Path] = None) -> Path:
    # Revalidate the previous public-score checkpoint before extending it.
    from .l2_temporal import validate_combined_submission as validate_temporal

    if temporal_pairs is not None:
        validate_temporal(base_submission, temporal_pairs)
    plans = load_plans(plans_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    submission = copy_base_submission(base_submission, output_dir)
    connection = sqlite3.connect(str(database))
    try:
        for plan in plans:
            question = plan["question"]
            question_id = int(question["id"])
            evidence, relevant_docs, relevant_tables = [], [], []
            for index, (name, fact) in enumerate(plan["components"].items(), 1):
                table = connection.execute(
                    "SELECT grid_json FROM tables WHERE table_id = ?",
                    (int(fact["table_id"]),),
                ).fetchone()
                if table is None:
                    raise ValueError(f"q{question_id}: missing table {fact['table_id']}")
                relative = f"data/q{question_id:04d}_{name}_evidence.csv"
                write_source_table_csv(output_dir / relative, json.loads(table[0]))
                evidence.append({"variable": f"df{index}", "csv_path": relative})
                if fact["document_id"] not in relevant_docs:
                    relevant_docs.append(fact["document_id"])
                source_line = int(
                    fact["source_line_0"] if line_base == 0 else fact["source_line_1"]
                )
                table_key = f"{fact['document_id']}|{source_line}"
                if table_key not in relevant_tables:
                    relevant_tables.append(table_key)
            query, answer = formula_query_and_answer(plan)
            submission.append({
                "id": question_id,
                "question": question["question"],
                "answer": float(answer),
                "relevant_docs": relevant_docs,
                "relevant_tables": relevant_tables,
                "evidence": evidence,
                "pandas_query": query,
            })
    finally:
        connection.close()
    submission.sort(key=lambda item: int(item["id"]))
    submission_path = output_dir / "submission.json"
    submission_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_formula_submission(submission_path, plans_path, base_submission)
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for path in sorted((output_dir / "data").glob("*.csv")):
            archive.write(path, path.relative_to(output_dir).as_posix())
    validation = validate_formula_submission(
        submission_path, plans_path, base_submission, zip_path
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "submission": str(submission_path), "zip": str(zip_path), **validation,
    }, ensure_ascii=False))
    return zip_path


def validate_formula_submission(submission_path: Path, plans_path: Path,
                                base_submission: Path,
                                zip_path: Optional[Path] = None) -> dict:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("pandas is required for validation") from error
    items = json.loads(submission_path.read_text(encoding="utf-8"))
    plans = {int(p["question"]["id"]): p for p in load_plans(plans_path)}
    base_items = {
        int(item["id"]): item
        for item in json.loads(base_submission.read_text(encoding="utf-8"))
    }
    errors, ids, paths = [], set(), set()
    formula_replayed = 0
    for item in items:
        question_id = int(item["id"])
        if question_id in ids:
            errors.append(f"q{question_id}: duplicate id")
        ids.add(question_id)
        if question_id in base_items and item != base_items[question_id]:
            errors.append(f"q{question_id}: base checkpoint item changed")
        frames = {}
        for evidence in item["evidence"]:
            relative = evidence["csv_path"]
            paths.add(relative)
            path = submission_path.parent / relative
            if not path.is_file():
                errors.append(f"q{question_id}: missing {relative}")
                continue
            frames[evidence["variable"]] = pd.read_csv(path)
        if question_id not in plans:
            continue
        plan = plans[question_id]
        try:
            expected_query, _ = formula_query_and_answer(plan)
            if item["pandas_query"] != expected_query:
                raise ValueError("formula query differs from audited canonical query")
            replay_values = {}
            for index, (name, fact) in enumerate(plan["components"].items(), 1):
                variable = f"df{index}"
                canonical_cell = cell_query(fact, variable)
                replay_values[name] = replay_generated_query(
                    canonical_cell.replace(variable, "df1", 1),
                    {"df1": frames[variable]},
                )
            actual = evaluate_formula(plan["formula"], replay_values)
            if not math.isclose(
                actual, float(item["answer"]), rel_tol=1e-12, abs_tol=1e-8
            ):
                raise ValueError(f"replay {actual} != answer {item['answer']}")
            formula_replayed += 1
        except (KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(f"q{question_id}: replay failed: {error}")
    expected_ids = {*base_items, *plans}
    if ids != expected_ids:
        errors.append("submission ids differ from base plus formula ids")
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as archive:
            members = set(archive.namelist())
        if members != {"submission.json", *paths}:
            errors.append("ZIP members differ from evidence references")
    result = {
        "valid": not errors,
        "items": len(items),
        "base_items_unchanged": sum(i in base_items for i in ids),
        "formula_items": sum(i in plans for i in ids),
        "formula_replayed": formula_replayed,
        "evidence_files": len(paths),
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def run_candidates(questions_path: Path, companies_path: Path, database: Path,
                   output_dir: Path, overrides_path: Optional[Path]) -> None:
    questions = load_questions(questions_path)
    companies = load_companies(companies_path)
    bank_tickers = {c.ticker for c in companies if "ngan hang" in normalize(c.name)}
    overrides = load_overrides(overrides_path)
    retriever = FactRetriever(database, bank_tickers=bank_tickers)
    candidates_output, plans = [], []
    try:
        for index, question_id in enumerate(sorted(questions), 1):
            question = questions[question_id]
            company, alias = resolve_company(question["question"], companies)
            spec = FORMULA_SPECS[question_id]
            validate_formula(spec.formula, {c.name for c in spec.components})
            plan = {
                "question": question,
                "ticker": company.ticker,
                "matched_company_alias": alias,
                "formula": spec.formula,
                "components": {},
            }
            for component in spec.components:
                parsed = component_question(question, company.ticker, component)
                candidates = retriever.retrieve(parsed, limit=5)
                override = overrides.get((question_id, component.name))
                if override is not None:
                    if override.year != component.year:
                        raise ValueError(f"q{question_id} {component.name}: wrong override year")
                    reviewed = retriever.retrieve_reviewed(parsed, as_l1_override(override))
                    candidates = [reviewed, *[
                        candidate for candidate in candidates
                        if (candidate.document_id, candidate.source_line_1,
                            candidate.row_index, candidate.column_index)
                        != (reviewed.document_id, reviewed.source_line_1,
                            reviewed.row_index, reviewed.column_index)
                    ][:4]]
                records = [fact_record(candidate, component.name, rank)
                           for rank, candidate in enumerate(candidates, 1)]
                candidates_output.extend({
                    **record, "question": question["question"],
                    "component_metric": component.metric_text,
                } for record in records)
                if not records:
                    raise ValueError(f"q{question_id} {component.name}: no candidates")
                plan["components"][component.name] = records[0]
            plans.append(plan)
            if index % 10 == 0:
                print(f"retrieved {index}/{len(questions)} formula questions", flush=True)
    finally:
        retriever.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates_output),
        encoding="utf-8",
    )
    (output_dir / "plans.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plans),
        encoding="utf-8",
    )
    summary = {
        "questions": len(plans),
        "components": sum(len(plan["components"]) for plan in plans),
        "candidate_facts": len(candidates_output),
        "manual_components": sum(
            fact["selection_source"] == "manual"
            for plan in plans for fact in plan["components"].values()
        ),
        "formula_shapes": dict(Counter(plan["formula"] for plan in plans)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--companies", type=Path, default=Path("ViFinQA/code_stock.csv"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l2-formula-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l2_formula_manual_overrides.csv"))
    submission = subparsers.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l2-formula-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-submission/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-formula-submission"))
    submission.add_argument("--line-base", choices=(0, 1), type=int, default=1)
    submission.add_argument(
        "--temporal-pairs", type=Path,
        help="Optional canonical q0578-q0655 plans used to validate the base submission",
    )
    validate = subparsers.add_parser("validate")
    validate.add_argument("--submission", type=Path, default=Path("outputs/l1-l2-formula-submission/submission.json"))
    validate.add_argument("--plans", type=Path, default=Path("outputs/l2-formula-facts/plans.jsonl"))
    validate.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-submission/submission.json"))
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        run_candidates(args.questions, args.companies, args.database,
                       args.output_dir, args.overrides)
    elif args.command == "submission":
        build_submission(args.plans, args.database, args.base_submission,
                         args.output_dir, args.line_base, args.temporal_pairs)
    else:
        print(json.dumps(validate_formula_submission(
            args.submission, args.plans, args.base_submission, args.zip_path
        ), ensure_ascii=False))


if __name__ == "__main__":
    main()
