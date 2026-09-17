"""Deterministic document retrieval and submission packaging for ViFinQA."""

import csv
from collections import defaultdict
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Callable
import unicodedata
from zipfile import ZIP_DEFLATED, ZipFile


LEGAL_WORDS = {
    "cong", "ty", "co", "phan", "ctcp", "tong", "tap", "doan", "ngan",
    "hang", "tmcp", "thuong", "mai", "trach", "nhiem", "huu", "han",
}
ALIAS_SPECS = (
    ("hang tieu dung masan", "MCH"), ("masan meatlife", "MML"),
    ("masan high tech materials", "MSR"), ("hoa phat", "HPG"),
    ("hoa sen", "HSG"), ("nam kim", "NKG"), ("masan", "MSN"),
    ("dai duong", "OGC"), ("vinamilk", "VNM"), ("dabaco", "DBC"),
    ("sao mai", "ASM"), ("thuy san minh phu", "MPC"),
    ("dam phu my", "DPM"), ("dam ca mau", "DCM"),
    ("do thi kinh bac", "KBC"), ("cong nghiep cao su viet nam", "GVR"),
    ("van phu", "VPI"), ("hai phat", "HPX"), ("tap doan dat xanh", "DXG"),
    ("dia oc sai gon thuong tin", "SCR"),
    ("ngan hang tmcp sai gon thuong tin", "STB"),
    ("sai gon thuong tin", "STB"), ("mbbank", "MBB"),
    ("eximbank", "EIB"), ("dau tu phat trien xay dung", "DIG"),
    ("tan binh", "PRT"), ("hoang huy", "HHS"),
    ("dien luc gelex", "GEE"), ("da nhim ham thuan da mi", "DNH"),
    ("nong nghiep quoc te hoang anh gia lai", "HNG"),
    ("go truong thanh", "TTF"), ("the gioi di dong", "MWG"),
    ("san bay viet nam", "ACV"), ("vicem ha tien", "HT1"),
    ("phu nhuan jewelry", "PNJ"), ("bac a", "BAB"), ("quoc dan", "NVB"),
    ("sai gon cong thuong", "SGB"), ("quan doi", "MBB"),
    ("saigonbank", "SGB"), ("bidv", "BID"),
    ("a chau", "ACB"), ("minh phu", "MPC"), ("dau khi ca mau", "DCM"),
    ("tap doan gelex", "GEX"), ("gelex", "GEX"),
    ("loc hoa dau binh son", "BSR"),
    ("binh son", "BSR"), ("pvtrans", "PVT"),
    ("nong nghiep quoc te hagl", "HNG"), ("vietjet", "VJC"),
    ("tkv", "DTK"),
    ("vingroup", "VIC"), ("vincom retail", "VRE"), ("song da", "SJG"),
    ("viglacera", "VGC"), ("kien long", "KLB"),
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
YEAR_DASH_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})\b")
RANGE_RE = re.compile(
    r"(?:giai doan(?: tu)?|trong cac nam(?: tai chinh)? tu|qua cac nam tu|tu nam)\s+"
    r"((?:19|20)\d{2})\s*(?:den|toi)\s*(?:nam\s+)?((?:19|20)\d{2})"
)
REQUIRED_KEYS = {"id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"}
PRIOR_YEAR_CUES = (
    "so voi nam truoc", "so voi nam lien truoc", "so voi nam truoc do",
    "so voi ky so sanh", "tai san binh quan", "von chu so huu binh quan",
    "hang ton kho binh quan", "trung binh hang ton kho",
    "trung binh tong tai san", "trung binh tai san co dinh",
    "binh quan tai san co dinh", "dau nam den cuoi nam",
    "dau ky va cuoi ky", "so ngay ton kho", "vong quay tong tai san",
    "vong quay tai san co dinh", " roe ", " roa ",
    "ty suat sinh loi tren von chu so huu",
)
NEXT_YEAR_CUES = (
    "vao nam sau nam", "cuoi nam ke tiep", "nam ngay sau nam dau tien",
    "nam lien sau nam dau tien",
)


@dataclass(frozen=True)
class Report:
    doc_id: str
    ticker: str
    year: int
    scope: str
    source_path: str
    duplicate_table_count: int | None = None


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    folded_name: str
    content_name: str


@dataclass
class ParsedQuestion:
    question: str
    tickers: list[str]
    years: list[int]
    scope: str
    scope_by_year: dict[int, str]
    entity_confidence: float
    entity_method: str
    candidate_tickers: list[str]


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("Đ", "D").replace("đ", "d"))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def content_tokens(text: str) -> list[str]:
    return [token for token in fold(text).split() if token not in LEGAL_WORDS]


def load_companies(csv_path: Path) -> dict[str, Company]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"Empty company registry: {csv_path}")
    companies = {}
    for row in rows[1:]:
        if len(row) >= 2:
            ticker, name = row[0].strip().upper(), row[1].strip()
            companies[ticker] = Company(ticker, name, fold(name), " ".join(content_tokens(name)))
    return companies


def scope_from_id(doc_id: str) -> str:
    lowered = doc_id.casefold()
    if "_separate" in lowered:
        return "separate"
    if "_consolidated" in lowered or "_aggregated" in lowered:
        return "consolidated"
    return "unknown"


def load_reports(statements_root: Path) -> dict[str, Report]:
    reports = {}
    for path in sorted(statements_root.glob("**/*_extracted.txt")):
        relative = path.relative_to(statements_root).as_posix()
        parts = relative.split("/")
        if len(parts) >= 3 and parts[1].isdigit():
            doc_id = path.parent.name
            reports[doc_id] = Report(doc_id, parts[0].upper(), int(parts[1]), scope_from_id(doc_id), relative)
    groups: dict[tuple[str, int, str], list[Report]] = defaultdict(list)
    for report in reports.values():
        groups[(report.ticker, report.year, report.scope)].append(report)
    for group in groups.values():
        if len(group) > 1:
            for report in group:
                path = statements_root / report.source_path
                count = path.read_text("utf-8", errors="replace").casefold().count("<table")
                reports[report.doc_id] = replace(report, duplicate_table_count=count)
    return reports


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def match_aliases(question: str) -> list[str]:
    normalized = fold(question)
    matches, occupied = [], []
    for alias, ticker in sorted(ALIAS_SPECS, key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            span = match.span()
            if not any(start <= span[0] and span[1] <= end for start, end in occupied):
                matches.append(ticker)
                occupied.append(span)
    return list(dict.fromkeys(matches))


def _explicit_tickers(question: str, known: set[str]) -> list[str]:
    parenthesized = re.findall(r"\(([A-Za-z0-9]{2,6})\)", question)
    upper_tokens = re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,5}(?![A-Za-z0-9])", question)
    explicit = [token.upper() for token in parenthesized + upper_tokens if token.upper() in known]
    if any(cue in fold(question) for cue in ("ma co phieu", "gom cac cong ty", "nhom")):
        tokens = re.findall(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]{1,5}(?![A-Za-z0-9])", question)
        explicit.extend(token.upper() for token in tokens if token.upper() in known)
    return list(dict.fromkeys(explicit))


def _company_score(question_folded: str, question_content: set[str], company: Company) -> float:
    if company.folded_name and company.folded_name in question_folded:
        return 1.0
    if company.content_name and company.content_name in question_folded:
        return 0.98
    company_tokens = set(content_tokens(company.name))
    if not company_tokens:
        return 0.0
    overlap = len(question_content & company_tokens) / len(company_tokens)
    return 0.82 * overlap + 0.18 * SequenceMatcher(None, company.content_name, question_folded).ratio()


def entity_candidates(question: str, companies: dict[str, Company]) -> list[tuple[str, float]]:
    question_folded = fold(question)
    question_tokens = set(content_tokens(question))
    scores = [(ticker, _company_score(question_folded, question_tokens, company)) for ticker, company in companies.items()]
    return sorted(scores, key=lambda item: (item[1], item[0]), reverse=True)


def infer_entities(question: str, companies: dict[str, Company]) -> tuple[list[str], float, str, list[str]]:
    explicit = _explicit_tickers(question, set(companies))
    scores = entity_candidates(question, companies)
    fuzzy_shortlist = [ticker for ticker, score in scores[:8] if score >= 0.32]
    question_folded = fold(question)
    question_content = " ".join(content_tokens(question))
    full_names = [ticker for ticker, company in companies.items() if company.folded_name and _contains_phrase(question_folded, company.folded_name)]
    content_names = [ticker for ticker, company in companies.items() if len(company.content_name.split()) >= 3 and _contains_phrase(question_content, company.content_name)]
    alias_names = match_aliases(question)

    def is_counterparty_only(ticker: str) -> bool:
        phrase, found = companies[ticker].content_name, False
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", question_content):
            found = True
            prefix = question_content[max(0, match.start() - 80):match.start()]
            relation = re.search(r"(?:phai thu tu|phai tra cho|dau tu vao|ban (?:hang )?(?:cho|voi)|giao dich voi)\s+((?:[a-z0-9]+\s+){0,7})$", prefix)
            if not relation or "cua me" in relation.group(1):
                return False
        return found

    full_names = [ticker for ticker in full_names if not is_counterparty_only(ticker)]
    content_names = [ticker for ticker in content_names if not is_counterparty_only(ticker)]
    alias_names = [ticker for ticker in alias_names if not is_counterparty_only(ticker)]
    specific_names = [ticker for ticker in content_names if not any(companies[ticker].content_name != companies[other].content_name and companies[ticker].content_name in companies[other].content_name for other in content_names)]
    exact_names = list(dict.fromkeys(full_names + specific_names + alias_names))
    explicit = [ticker for ticker in explicit if not any(ticker != other and fold(ticker) in companies[other].content_name.split() for other in exact_names)]
    entities = list(dict.fromkeys(explicit + exact_names))
    shortlist = list(dict.fromkeys(entities + fuzzy_shortlist))
    if entities:
        method = "explicit_ticker" if explicit and len(entities) == len(explicit) else "single_official_name" if len(entities) == 1 else "union_all_entity_detectors"
        return entities, 1.0 if explicit else 0.98, method, shortlist
    best_ticker, best_score = scores[0]
    if best_score >= 0.73 and best_score - scores[1][1] >= 0.08:
        return [best_ticker], best_score, "fuzzy_high_confidence", shortlist
    return [], best_score, "ambiguous", shortlist


def infer_years(question: str) -> list[int]:
    years = {int(value) for value in YEAR_RE.findall(question)}
    ranges = list(RANGE_RE.findall(fold(question))) + list(YEAR_DASH_RE.findall(question))
    for start, end in ranges:
        low, high = sorted((int(start), int(end)))
        if high - low <= 15:
            years.update(range(low, high + 1))
    return sorted(years)


def infer_scope(question: str, years: list[int]) -> tuple[str, dict[int, str]]:
    normalized = fold(question)
    separate_cues = (
        "cong ty me", "ngan hang me", "bao cao rieng",
        "bao cao tai chinh rieng", "bctc rieng", "co so rieng",
    )
    consolidated_cues = ("hop nhat", "bao cao hop nhat")
    has_separate = any(cue in normalized for cue in separate_cues)
    has_consolidated = any(cue in normalized for cue in consolidated_cues)
    if has_separate and not has_consolidated:
        return "separate", {year: "separate" for year in years}
    if has_consolidated and not has_separate:
        return "consolidated", {year: "consolidated" for year in years}
    if not (has_separate and has_consolidated):
        return "consolidated", {year: "consolidated" for year in years}
    cues = [(match.start(), scope) for scope, phrases in (("separate", separate_cues), ("consolidated", consolidated_cues)) for phrase in phrases for match in re.finditer(phrase, normalized)]
    cues.sort()
    scope_by_year = {}
    for year in years:
        positions = [match.start() for match in re.finditer(str(year), normalized)]
        position = positions[0] if positions else len(normalized)
        preceding = [cue for cue in cues if cue[0] <= position]
        scope_by_year[year] = preceding[-1][1] if preceding else cues[0][1]
    return "mixed", scope_by_year


def parse_question(question: str, companies: dict[str, Company]) -> ParsedQuestion:
    tickers, confidence, method, candidates = infer_entities(question, companies)
    years = infer_years(question)
    scope, scope_by_year = infer_scope(question, years)
    return ParsedQuestion(question, tickers, years, scope, scope_by_year, confidence, method, candidates)


def required_report_years(parsed: ParsedQuestion) -> list[int]:
    years = list(parsed.years)
    if not years:
        return []
    text = f" {fold(parsed.question)} "
    needs_prior = any(cue in text for cue in PRIOR_YEAR_CUES)
    if len(years) == 1 and any(cue in text for cue in ("tang truong", "toc do tang", "sut giam nam")):
        needs_prior = True
    average_cues = (
        "tai san binh quan", "von chu so huu binh quan",
        "hang ton kho binh quan", "trung binh hang ton kho",
        "trung binh tong tai san", "trung binh tai san co dinh",
        "binh quan tai san co dinh",
    )
    if needs_prior and any(cue in text for cue in average_cues):
        for match in re.finditer(r"(?:binh quan|trung binh)", text):
            if len(YEAR_RE.findall(text[match.start():match.start() + 130])) >= 2:
                needs_prior = False
                break
    if needs_prior and "dau nam den cuoi nam" in text:
        needs_prior = len(years) == 1
    if needs_prior and len(years) == 2 and years[1] - years[0] == 1:
        direct_span = re.search(rf"tu nam {years[0]} den (?:nam )?{years[1]}", text)
        comparison = any(cue in text for cue in ("so voi nam truoc", "so voi nam lien truoc", "so voi nam truoc do"))
        if direct_span and comparison:
            needs_prior = False
    if needs_prior and (" roe " in text or "ty suat sinh loi tren von chu so huu" in text):
        targets = [int(year) for year in re.findall(r"(?:ty so )?roe (?:tai )?nam ((?:19|20)\d{2})", text)]
        if targets and all(year - 1 in years for year in targets):
            needs_prior = False
    if needs_prior:
        years.append(min(years) - 1)
    if any(cue in text for cue in NEXT_YEAR_CUES):
        years.append(max(parsed.years) + 1)
    return list(dict.fromkeys(years))


def minimal_report_years(years: list[int]) -> list[int]:
    uncovered, selected = set(years), []
    while uncovered:
        year = max(uncovered)
        selected.append(year)
        uncovered.discard(year)
        uncovered.discard(year - 1)
    return sorted(selected)


def choose_report(candidates: list[Report], scope: str) -> Report | None:
    exact = [report for report in candidates if report.scope == scope]
    unknown = [report for report in candidates if report.scope == "unknown"]
    pool = exact or unknown
    if not pool:
        return None
    preferred = [report for report in pool if "_aggregated" not in report.doc_id] or pool
    most_tables = max(report.duplicate_table_count or 0 for report in preferred)
    complete = [report for report in preferred if (report.duplicate_table_count or 0) == most_tables]
    return sorted(complete, key=lambda report: report.doc_id)[0]


def retrieve_docs(parsed: ParsedQuestion, reports: dict[str, Report], use_comparative_cover: bool = False) -> tuple[list[str], list[dict]]:
    by_ticker_year: dict[tuple[str, int], list[Report]] = defaultdict(list)
    for report in reports.values():
        by_ticker_year[report.ticker, report.year].append(report)
    requested_years = required_report_years(parsed) or sorted({report.year for report in reports.values()})
    report_years = minimal_report_years(requested_years) if use_comparative_cover else requested_years
    selected, decisions = [], []
    for ticker in parsed.tickers:
        for year in report_years:
            scope = parsed.scope_by_year.get(year, parsed.scope if parsed.scope != "mixed" else "consolidated")
            report = choose_report(by_ticker_year.get((ticker, year), []), scope)
            decisions.append({
                "ticker": ticker,
                "requested_year": year,
                "year_source": "explicit" if year in parsed.years else "derived_dependency",
                "scope": scope,
                "selected_doc": report.doc_id if report else None,
                "selected_scope": report.scope if report else None,
            })
            if report:
                selected.append(report)
    return list(dict.fromkeys(report.doc_id for report in selected)), decisions


def make_row(source: dict, docs: list[str], tables: list[str] | None = None, evidence: list[dict] | None = None) -> dict:
    variables = [item["variable"] for item in evidence or []]
    return {
        "id": int(source["id"]), "question": source["question"], "answer": 0.0,
        "relevant_docs": docs, "relevant_tables": tables or [], "evidence": evidence or [],
        # Retrieval is the project boundary: bind the evidence without
        # attempting to execute an answer program.
        "pandas_query": f"result = ({', '.join(variables)})" if variables else "",
    }


def validate(rows: list[dict], questions: list[dict], reports: dict[str, Report]) -> list[str]:
    errors = []
    if len(rows) != len(questions):
        errors.append(f"row_count={len(rows)} expected={len(questions)}")
    if [row.get("id") for row in rows] != [int(question["id"]) for question in questions]:
        errors.append("IDs/order differ from official questions")
    valid_docs = set(reports)
    questions_by_id = {int(question["id"]): question for question in questions}
    for row in rows:
        missing = REQUIRED_KEYS - set(row)
        if missing:
            errors.append(f"id={row.get('id')} missing={sorted(missing)}")
        if row.get("question") != questions_by_id.get(int(row.get("id", 0)), {}).get("question"):
            errors.append(f"id={row.get('id')} question mismatch")
        invalid = [doc for doc in row.get("relevant_docs", []) if doc not in valid_docs]
        if invalid:
            errors.append(f"id={row.get('id')} invalid_docs={invalid}")
        if len(row.get("relevant_docs", [])) != len(set(row.get("relevant_docs", []))):
            errors.append(f"id={row.get('id')} duplicate docs")
        tables = row.get("relevant_tables", [])
        if len(tables) != len(set(tables)):
            errors.append(f"id={row.get('id')} duplicate tables")
        if any(table.partition("|")[0] not in row.get("relevant_docs", []) for table in tables):
            errors.append(f"id={row.get('id')} table outside relevant_docs")
        evidence = row.get("evidence", [])
        # Evidence binds the tables the answer reads, and `relevant_tables` is the
        # wider set the retrieval metrics score; expansion adds to the second only.
        if len(evidence) > len(tables):
            errors.append(f"id={row.get('id')} more evidence than tables")
    return errors


def write_package(
    output_dir: Path,
    rows: list[dict],
    validator: Callable[[Path], list[str]] | None = None,
) -> Path:
    data_dir = output_dir / "package" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    submission_path = data_dir.parent / "submission.json"
    submission_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    csv_paths = {
        output_dir / "package" / item["csv_path"]
        for row in rows
        for item in row["evidence"]
    }
    missing = sorted(path for path in csv_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing evidence CSV: {missing[0]}")
    if validator:
        errors = validator(data_dir.parent)
        if errors:
            raise ValueError("Strict submission validation failed:\n" + "\n".join(errors))
    zip_path = output_dir / "submission.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        evidence_paths = csv_paths | {path.with_suffix(".json") for path in csv_paths}
        for path in sorted(evidence_paths):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir / "package").as_posix())
    return zip_path
