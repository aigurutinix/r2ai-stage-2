"""Auditable L4 select-then-compute plans for ViFinQA q0495-q0538."""

from __future__ import annotations

import argparse
from pathlib import Path

from .set_reasoning import (
    ItemSpec,
    SetSpec,
    build_submission,
    fact,
    run_candidates,
    validate_specs,
    validate_submission,
)


FIRST_ID = 495
LAST_ID = 538


def Y(question_id: int, ticker: str, years: tuple[int, ...], selector: str,
      answer: str, unit: str = "VND_1", scope: str = "consolidated",
      operation: str = "select_argmax_answer", selector_expression: str = "abs(s)",
      answer_expression: str = "abs(a)", selector_unit: str = "VND_1",
      threshold: float | None = None) -> SetSpec:
    return SetSpec(question_id, operation, tuple(
        ItemSpec(str(year), answer_expression, (
            fact("s", ticker, year, selector, selector_unit, scope),
            fact("a", ticker, year, answer, unit, scope),
        ), selector_expression) for year in years
    ), threshold)


def E(question_id: int, tickers: tuple[str, ...], year: int, selector: str,
      answer: str, unit: str = "VND_1", scope: str = "consolidated",
      operation: str = "select_argmax_answer", selector_expression: str = "abs(s)",
      answer_expression: str = "abs(a)", selector_unit: str = "VND_1",
      threshold: float | None = None) -> SetSpec:
    return SetSpec(question_id, operation, tuple(
        ItemSpec(ticker, answer_expression, (
            fact("s", ticker, year, selector, selector_unit, scope),
            fact("a", ticker, year, answer, unit, scope),
        ), selector_expression) for ticker in tickers
    ), threshold)


L4_SPECS = {
    495: Y(495, "VGT", (2018, 2020, 2021, 2022),
           "phai thu ngan han khac cac ben lien quan",
           "tong tien thue toi thieu phai tra theo hop dong thue hoat dong khong huy ngang",
           "VND_1e9"),
    496: SetSpec(496, "select_argmax_answer", tuple(
        ItemSpec(str(year), "abs(a) / abs(b) * 100.0", (
            fact("s", "MWG", year, "chi phi khau hao va hao mon"),
            fact("a", "MWG", year, "gia tri thuan hang ton kho cuoi nam"),
            fact("b", "MWG", year, "tong no vay cuoi nam"),
        ), "abs(s)") for year in (2017, 2018, 2020, 2022)
    )),
    497: SetSpec(497, "select_argmax_answer", tuple(
        ItemSpec(ticker, "abs(a) / abs(b)", (
            fact("s", ticker, 2017, "chi phi thue thu nhap doanh nghiep hien hanh"),
            fact("a", ticker, 2017, "tong gia von hang ban"),
            fact("b", ticker, 2017, "tong gia goc hang ton kho cuoi nam"),
        ), "abs(s)") for ticker in ("VIC", "DXG", "SCR", "CEO")
    )),
    498: Y(498, "ACB", (2018, 2020, 2022), "tong chi phi hoat dong",
           "tien gui cua ca nhan", "VND_1e6", "separate",
           "select_filter_gt_answer", "abs(s)", "abs(a)", "VND_1e9", 10000.0),
    499: SetSpec(499, "select_argmax_answer", tuple(
        ItemSpec(str(year), "(a - b) / abs(b) * 100.0", (
            fact("s", "FTS", year, "vay ngan han"),
            fact("a", "FTS", year, "tien va cac khoan tuong duong tien cuoi nam"),
            fact("b", "FTS", year, "tien va cac khoan tuong duong tien dau nam", period="start"),
        ), "abs(s)") for year in (2018, 2019, 2022, 2023, 2024)
    )),
    500: SetSpec(500, "select_argmax_answer", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("s1", "PNJ", year, "chi phi xay dung co ban do dang cuoi nam"),
            fact("s0", "PNJ", year, "chi phi xay dung co ban do dang dau nam", period="start"),
            fact("a", "PNJ", year, "vay ngan hang tai ngay 31 thang 12", "VND_1e9"),
        ), "s1 - s0") for year in (2018, 2020, 2021, 2022)
    )),
    501: SetSpec(501, "select_max_then_argmax_key", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("s", "QNS", year, "gia goc dau tu vao cong ty tnhh mtv thuong mai thanh phat", scope="separate"),
            fact("a", "QNS", year, "tong gia goc no phai thu qua han", scope="separate"),
        ), "abs(s)") for year in (2015, 2020, 2021, 2023)
    )),
    502: Y(502, "PLX", (2015, 2016, 2018, 2019, 2020, 2021),
           "tien gui quy binh on gia xang dau tai ngan hang", "lai du thu cuoi nam",
           "VND_1e9", "separate"),
    503: Y(503, "VGT", (2015, 2017, 2019, 2021, 2022, 2023), "von chu so huu cuoi nam",
           "mua hang hoa va dich vu tu cong ty tnhh coats phong phu", "VND_1e9"),
    504: SetSpec(504, "select_argmax_answer", tuple(
        ItemSpec(str(year), "a / b * 100.0", (
            fact("s", "ASM", year, "vay ngan han cuoi nam"),
            fact("a", "ASM", year, "luu chuyen tien thuan tu hoat dong kinh doanh"),
            fact("b", "ASM", year, "doanh thu"),
        ), "abs(s)") for year in (2022, 2024, 2025)
    )),
    505: Y(505, "IJC", (2017, 2018, 2019, 2022, 2024), "vay ngan han ngan hang cuoi nam",
           "vay ngan han phai tra cac ben lien quan cuoi nam", "VND_1e9", "separate"),
    506: E(506, ("IJC", "DXG", "NVL", "NLG", "KBC"), 2024,
           "gia tri con lai tai san co dinh huu hinh cuoi nam",
           "doanh thu thuan ban hang va cung cap dich vu", "VND_1e12"),
    507: SetSpec(507, "select_argmax_answer", tuple(
        ItemSpec(str(year), "abs(a) / abs(b) * 100.0", (
            fact("s", "DIG", year, "tra truoc cho nguoi ban ngan han cuoi nam", scope="separate"),
            fact("a", "DIG", year, "chi phi lai vay", scope="separate"),
            fact("b", "DIG", year, "loi nhuan truoc thue", scope="separate"),
        ), "abs(s)") for year in (2015, 2016, 2017, 2021, 2024)
    )),
    508: E(508, ("OCB", "ACB", "STB"), 2021, "chi phi cho phan bo cuoi nam",
           "lai thuan tu hoat dong khac", "VND_1e6", "separate"),
    509: Y(509, "GAS", (2015, 2016, 2017, 2019),
           "thue va cac khoan khac phai nop nha nuoc cuoi nam",
           "ban hang cho tong cong ty dien luc dau khi viet nam", "VND_1e12"),
    510: SetSpec(510, "select_argmax_answer", tuple(
        ItemSpec(ticker, "abs(a) / (abs(b) + abs(c)) * 100.0", (
            fact("s", ticker, 2024, "chi phi cho nhan vien"),
            fact("a", ticker, 2024, "cho vay cac to chuc tin dung khac bang vnd"),
            fact("b", ticker, 2024, "tien gui khong ky han cua cac to chuc tin dung khac bang vnd"),
            fact("c", ticker, 2024, "tien gui co ky han cua cac to chuc tin dung khac bang vnd"),
        ), "abs(s)") for ticker in ("BAB", "SSB", "NAB", "VIB")
    )),
    511: SetSpec(511, "select_argmax_answer", tuple(
        ItemSpec(ticker, "abs(a) / abs(b) * 100.0", (
            fact("s", ticker, 2018, "lai co ban tren co phieu", "VND_per_share"),
            fact("a", ticker, 2018, "loi nhuan thuan sau thue hop nhat"),
            fact("b", ticker, 2018, "von chu so huu cuoi nam"),
        ), "abs(s)") for ticker in ("DPM", "HT1", "HPG")
    )),
    512: Y(512, "HSG", (2015, 2018, 2019, 2021, 2022, 2023), "tong von chu so huu",
           "vay dai han tai ngay 30 thang 9", "VND_1e9"),
    513: Y(513, "HHS", (2015, 2016, 2017, 2020, 2021), "tong gia goc hang ton kho cuoi nam",
           "gia goc nguyen lieu vat lieu cuoi nam", "VND_1e9"),
    514: Y(514, "HDG", (2015, 2016, 2017, 2018, 2019),
           "luu chuyen tien thuan tu hoat dong tai chinh",
           "khach hang mua can ho tra tien truoc cuoi nam", "VND_1e9", "separate",
           "select_argmin_answer", "s"),
    515: Y(515, "VAB", (2020, 2024, 2025), "tong no phai tra cuoi nam",
           "vat lieu va cong cu cuoi nam", "VND_1e9"),
    516: SetSpec(516, "select_argmax_answer", tuple(
        ItemSpec(str(year), "(a - b) / abs(b) * 100.0", (
            fact("s", "ACB", year, "quy khen thuong phuc loi cuoi nam", scope="separate"),
            fact("a", "ACB", year, "tong gia tri ghi nhan cong cu phai sinh cuoi nam", scope="separate"),
            fact("b", "ACB", year, "tong gia tri ghi nhan cong cu phai sinh dau nam", scope="separate", period="start"),
        ), "abs(s)") for year in (2015, 2019, 2022)
    )),
    517: SetSpec(517, "select_argmax_answer", tuple(
        ItemSpec(ticker, "(a - b) / abs(b) * 100.0", (
            fact("s", ticker, 2017, "thue thu nhap doanh nghiep cuoi nam"),
            fact("a", ticker, 2017, "doanh thu thuan nam 2017"),
            fact("b", ticker, 2017, "doanh thu thuan nam 2016", period="start"),
        ), "abs(s)") for ticker in ("PVT", "BSR", "PLX")
    )),
    518: E(518, ("CTG", "NAB", "ABB", "KLB"), 2023, "thu nhap tu hoat dong khac",
           "thu nhap lai thuan", "VND_1e6", operation="select_argmin_answer",
           selector_expression="s"),
    519: SetSpec(519, "select_argmax_answer", tuple(
        ItemSpec(str(year), "abs(a) / abs(b) * 100.0", (
            fact("s", "HPG", year, "phan bo chi phi sua chua van phong cong cu dung cu va chi phi tra truoc dai han khac", scope="separate"),
            fact("a", "HPG", year, "lai tien gui va cho vay", scope="separate"),
            fact("b", "HPG", year, "doanh thu hoat dong tai chinh", scope="separate"),
        ), "abs(s)") for year in (2015, 2018, 2022, 2024)
    )),
    520: E(520, ("MCH", "MML", "VNM", "ASM"), 2022, "gia tri ghi so vay ngan han cuoi nam",
           "tong doanh thu bo phan thuan hop nhat", "VND_1e12"),
    521: Y(521, "HDG", (2021, 2023, 2025), "tong tien va cac khoan tuong duong tien cuoi nam",
           "chi phi lai vay", "VND_1e9"),
    522: Y(522, "BVH", (2017, 2018, 2023, 2025), "no kho doi da xu ly",
           "anh huong loi nhuan truoc thue khi gia thi truong danh muc co phieu niem yet giam 10 phan tram",
           "VND_1e6"),
    523: Y(523, "DCM", (2019, 2022, 2023, 2024, 2025),
           "lai du thu tien gui co ky han cuoi nam", "tien mat cuoi nam", "VND_1e9", "separate"),
    524: Y(524, "OCB", (2017, 2018, 2021, 2022),
           "cho vay cac to chuc kinh te va ca nhan trong nuoc",
           "quy khen thuong va phuc loi cuoi nam", "VND_1e9"),
    525: Y(525, "OCB", (2017, 2019, 2022), "thue thu nhap doanh nghiep phai nop trong nam",
           "tong du phong rui ro cho vay khach hang cuoi nam", "VND_1e12", "separate"),
    526: Y(526, "MWG", (2023, 2024, 2025), "lai co ban va suy giam tren moi co phieu",
           "chi phi khac", "VND_1e9", selector_unit="VND_per_share"),
    527: SetSpec(527, "select_argmax_answer", tuple(
        ItemSpec(str(year), "a / b * 100.0", (
            fact("s", "ACV", year, "phai thu khac ve co tuc loi nhuan duoc chia", scope="separate"),
            fact("a", "ACV", year, "loi nhuan truoc thue", scope="separate"),
            fact("b", "ACV", year, "doanh thu cung cap dich vu", scope="separate"),
        ), "abs(s)") for year in (2016, 2020, 2021, 2023)
    )),
    528: SetSpec(528, "select_argmax_answer", tuple(
        ItemSpec(str(year), "a / b * 100.0", (
            fact("s", "ABB", year, "trich lap hoan nhap du phong chung khoan dau tu san sang de ban", scope="separate"),
            fact("a", "ABB", year, "loi nhuan thuan tu hoat dong kinh doanh truoc chi phi du phong rui ro tin dung", scope="separate"),
            fact("b", "ABB", year, "tong tai san", scope="separate"),
        ), "abs(s)") for year in (2020, 2022, 2023)
    )),
    529: Y(529, "QNS", (2017, 2020, 2023, 2024), "tong chi phi ban hang",
           "tong phai tra nguoi ban ngan han cuoi nam", "VND_1e9"),
    530: Y(530, "NAB", (2021, 2022, 2023, 2024, 2025),
           "chi phi xay dung co ban do dang cuoi ky", "no du tieu chuan cuoi nam", "VND_1e12"),
    531: Y(531, "MPC", (2016, 2018, 2020, 2022, 2023),
           "xay dung co ban do dang cuoi nam", "chi phi van chuyen va chi phi dich vu mua ngoai",
           "VND_1e9"),
    532: E(532, ("SAB", "MPC", "MSN", "MCH", "HAG"), 2017,
           "so luong co phieu pho thong cuoi nam", "tong tai san thue thu nhap hoan lai",
           "VND_1e9", selector_unit="shares"),
    533: Y(533, "HND", (2016, 2018, 2019, 2020), "thue tinh theo thue suat cua cong ty",
           "tien thue toi thieu phai tra trong vong mot nam theo hop dong thue hoat dong khong huy ngang",
           "VND_1e9"),
    534: E(534, ("GEX", "HBC", "PC1"), 2024,
           "ty le quyen bieu quyet tai don vi lien doanh lien ket",
           "tong phai tra sau 12 thang", "VND_1e12",
           operation="select_filter_ge_answer_sum", selector_expression="s",
           selector_unit="percent", threshold=50.0),
    535: Y(535, "TTF", (2017, 2019, 2020, 2021, 2022, 2024),
           "tong phai tra nguoi ban ngan han cuoi nam", "thu nhap khac tu thanh ly tai san",
           "VND_1e6", "separate"),
    536: E(536, ("GVR", "DPM", "HT1", "NKG"), 2023, "tong von chu so huu cuoi nam",
           "tong chi phi thue thu nhap doanh nghiep hien hanh", "VND_1e9"),
    537: Y(537, "VGC", (2019, 2020, 2022, 2023, 2024),
           "trich quy phat trien khoa hoc va cong nghe",
           "tong dau tu gop von vao don vi khac cuoi nam", "VND_1e9"),
    538: SetSpec(538, "select_filter_positive_argmax_key", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("s", ticker, 2024, "luu chuyen tien thuan tu hoat dong kinh doanh", scope="separate"),
            fact("a", ticker, year, "chi phi lai vay", scope="separate"),
        ), "s") for ticker in ("AAA", "VIF", "NKG") for year in (2023, 2024, 2025)
    )),
}


validate_specs(L4_SPECS, FIRST_ID, LAST_ID)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l4-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l4_manual_overrides.csv"))
    submission = sub.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l4-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-submission-final/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-l3-l4-submission"))
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--plans", type=Path, default=Path("outputs/l4-facts/plans.jsonl"))
    validate.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-submission-final/submission.json"))
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        run_candidates(L4_SPECS, args.questions, args.database, args.output_dir, args.overrides)
    elif args.command == "submission":
        build_submission(args.plans, L4_SPECS, args.database, args.base_submission, args.output_dir)
    else:
        print(validate_submission(args.submission, args.plans, L4_SPECS,
                                  args.base_submission, args.zip_path))


if __name__ == "__main__":
    main()
