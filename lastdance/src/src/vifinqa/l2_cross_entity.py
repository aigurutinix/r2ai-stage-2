"""Auditable L2 cross-entity plans for ViFinQA q0733-q0812."""

from __future__ import annotations

import argparse
from pathlib import Path

from .set_reasoning import (
    ItemSpec,
    SetSpec,
    build_submission,
    direct_item,
    fact,
    ratio_item,
    run_candidates,
    validate_specs,
    validate_submission,
)


FIRST_ID = 733
LAST_ID = 812


def D(question_id: int, left: str, right: str, year: int, metric: str,
      unit: str, left_scope: str = "consolidated",
      right_scope: str = "consolidated", operation: str = "difference",
      period: str = "end_or_flow") -> SetSpec:
    return SetSpec(question_id, operation, (
        direct_item(left, left, year, metric, unit, left_scope, period),
        direct_item(right, right, year, metric, unit, right_scope, period),
    ))


def DA(question_id: int, left: str, right: str, year: int, metric: str,
       unit: str, left_scope: str = "consolidated",
       right_scope: str = "consolidated") -> SetSpec:
    return SetSpec(question_id, "difference", (
        ItemSpec(left, "abs(a)", (fact("a", left, year, metric, unit, left_scope),)),
        ItemSpec(right, "abs(a)", (fact("a", right, year, metric, unit, right_scope),)),
    ))


def R(question_id: int, left: str, right: str, year: int, numerator: str,
      denominator: str, left_scope: str = "consolidated",
      right_scope: str = "consolidated") -> SetSpec:
    return SetSpec(question_id, "difference", (
        ratio_item(left, left, year, numerator, denominator, left_scope),
        ratio_item(right, right, year, numerator, denominator, right_scope),
    ))


def shares_million_item(ticker: str, year: int, metric: str,
                        scope: str = "consolidated") -> ItemSpec:
    return ItemSpec(ticker, "a / 1000000.0", (
        fact("a", ticker, year, metric, "shares", scope),
    ))


CROSS_SPECS = {
    733: D(733, "SSB", "VPB", 2020, "thu nhap binh quan thang tren moi nhan vien", "VND_1e6", "separate", "separate"),
    734: D(734, "DIG", "KBC", 2025, "luu chuyen tien thuan tu hoat dong kinh doanh", "VND_1e12", "separate", "separate"),
    735: R(735, "OGC", "ASM", 2017, "vay va no thue tai chinh ngan han", "vay va no"),
    736: D(736, "CEO", "VPI", 2024, "chi phi xay dung co ban do dang", "VND_1e9"),
    737: DA(737, "ASM", "BAF", 2023, "gia von hang ban", "VND_1e9"),
    738: D(738, "CTG", "MBB", 2024, "gia tri con lai tai san co dinh huu hinh", "VND_1e6", "separate", "separate"),
    739: D(739, "IJC", "SCR", 2017, "no trai phieu thuong", "VND_1e9", "separate", "separate"),
    740: D(740, "GEG", "DNH", 2023, "gia goc dau tu vao cong ty lien ket", "VND_1e9", "separate", "separate"),
    741: D(741, "VSF", "MPC", 2019, "loi nhuan tren moi co phieu co ban", "VND_1e3"),
    742: D(742, "MSB", "VCB", 2019, "du phong cu the", "VND_1e6"),
    743: D(743, "GEE", "SAM", 2023, "gia tri con lai tai san co dinh huu hinh", "VND_1e12"),
    744: SetSpec(744, "difference", (
        direct_item("KLB", "KLB", 2024, "trai phieu", "VND_1e6", "separate"),
        direct_item("EIB", "EIB", 2024, "phat hanh giay to co gia", "VND_1e6", "separate"),
    )),
    745: D(745, "NVB", "VIB", 2016, "cac khoan phai thu ben ngoai", "VND_1e6", "separate", "separate"),
    746: D(746, "DXS", "KHG", 2024, "chi phi thue thu nhap doanh nghiep hien hanh", "VND_1e6", "separate", "separate"),
    747: D(747, "VPI", "VRE", 2023, "chi phi xay dung va phat trien bat dong san", "VND_1e12"),
    748: SetSpec(748, "difference", (
        shares_million_item("MBB", 2018, "so luong co phieu pho thong dang luu hanh"),
        shares_million_item("ACB", 2018, "so luong co phieu pho thong dang luu hanh"),
    )),
    749: D(749, "EIB", "ACB", 2025, "thu nhap tu mua ban chung khoan dau tu", "VND_1e6", "separate", "separate"),
    750: SetSpec(750, "difference", (
        direct_item("SAB", "SAB", 2024, "loi nhuan thuan trong nam", "VND_1e9", "separate"),
        direct_item("DBC", "DBC", 2024, "loi nhuan sau thue thu nhap doanh nghiep", "VND_1e9", "separate"),
    )),
    751: D(751, "OCB", "NAB", 2023, "du no cho vay ca nhan", "VND_1e6"),
    752: D(752, "ABB", "MBB", 2022, "lai thuan tu hoat dong dich vu", "VND_1e6"),
    753: D(753, "VIB", "MSB", 2017, "loi nhuan thuan", "VND_1e6", "separate", "separate"),
    754: D(754, "HDB", "ABB", 2021, "cho vay khach hang trong nuoc", "VND_1e6", "separate", "separate"),
    755: D(755, "BID", "STB", 2025, "chenh nhay cam voi lai suat noi bang", "VND_1e6"),
    756: D(756, "ACB", "HDB", 2022, "cho vay ngan han khach hang", "VND_1e6"),
    757: D(757, "SHB", "NAB", 2020, "gia tri con lai tai san co dinh huu hinh", "VND_1e6"),
    758: D(758, "KBC", "VIC", 2025, "vay ngan han tu cac ben lien quan", "VND_1e9", "separate", "separate"),
    759: D(759, "SCR", "NVL", 2025, "phai thu ngan han cua khach hang gia tri ghi so", "VND_1e12"),
    760: D(760, "BAB", "NVB", 2024, "chi phi cho nhan vien", "VND_1e6"),
    761: D(761, "KHG", "IJC", 2024, "tong thu lao thanh vien quan ly chu chot", "VND_1e6"),
    762: D(762, "GEX", "SAM", 2019, "dau tu tai chinh dai han", "VND_1e9"),
    763: D(763, "CTG", "VPB", 2019, "thu nhap lai tien gui", "VND_1e6", "separate", "separate"),
    764: D(764, "GVR", "DPM", 2015, "quy dau tu phat trien", "VND_1e12"),
    765: D(765, "VPB", "ACB", 2018, "thu nhap lai va cac khoan thu nhap tuong tu", "VND_1e6"),
    766: D(766, "PVT", "BSR", 2017, "phai thu khach hang tu cac ben lien quan", "VND_1e9"),
    767: D(767, "DLG", "ACV", 2015, "tien va cac khoan tuong duong tien", "VND_1e12", "separate", "separate"),
    768: D(768, "MBB", "MSB", 2025, "lai thuan tu hoat dong khac", "VND_1e6"),
    769: D(769, "VSC", "ACV", 2015, "gia tri con lai quyen su dung dat", "VND_1e9"),
    770: D(770, "GAS", "POW", 2023, "von chu so huu", "VND_1e9", "separate", "separate"),
    771: D(771, "MBB", "CTG", 2023, "chi phi trich lap du phong rui ro cho vay khach hang", "VND_1e6", "separate", "separate"),
    772: D(772, "HAG", "HNG", 2021, "tra truoc cho nha cung cap hang hoa va dich vu", "VND_1e9"),
    773: D(773, "VRE", "KBC", 2020, "chi phi dich vu mua ngoai", "VND_1e6", "separate", "separate"),
    774: D(774, "VIB", "SHB", 2022, "tien mat va vang", "VND_1e6", "separate", "separate"),
    775: D(775, "SNZ", "VPI", 2022, "chi phi dich vu mua ngoai", "VND_1e9"),
    776: D(776, "MSN", "MML", 2021, "loi nhuan sau thue", "VND_1e12", "separate", "separate"),
    777: D(777, "MCH", "VNM", 2018, "chi phi thue thu nhap doanh nghiep hien hanh", "VND_1e9", "separate", "separate"),
    778: D(778, "HSG", "DPM", 2018, "chi phi lai vay", "VND_1e9"),
    779: D(779, "MCH", "MML", 2018, "lai co ban tren mot co phieu", "VND_1e3"),
    780: D(780, "BAB", "SGB", 2024, "du phong rui ro cho vay khach hang", "VND_1e6", "separate", "separate"),
    781: D(781, "PC1", "GEX", 2018, "no vay ngan han", "VND_1e9", "separate", "separate"),
    782: D(782, "VNM", "HNG", 2016, "von co phan", "VND_1e9"),
    783: D(783, "MBB", "EIB", 2023, "tong tai san", "VND_1e6", "separate", "separate"),
    784: SetSpec(784, "difference", (
        direct_item("VGT", "VGT", 2023, "tien thue trong vong 1 nam", "VND_1e9", "separate"),
        direct_item("TTF", "TTF", 2023, "tien thue khong qua 1 nam", "VND_1e9", "separate"),
    )),
    785: D(785, "HHS", "HUT", 2020, "so luong co phieu dang luu hanh", "shares", operation="difference"),
    786: D(786, "DTK", "HND", 2023, "loi nhuan truoc thue", "VND_1e9", "separate", "consolidated"),
    787: D(787, "NVB", "KLB", 2024, "trang thai tien te noi bang tong", "VND_1e6", "separate", "separate"),
    788: D(788, "BAF", "ASM", 2020, "chi phi nguyen vat lieu", "VND_1e12"),
    789: D(789, "VIF", "AAA", 2022, "tong chi phi dich vu mua ngoai", "VND_1e9", "separate", "separate"),
    790: D(790, "VIB", "BID", 2023, "cong nghiep che bien che tao", "percent", "separate", "separate"),
    791: D(791, "GAS", "GEG", 2017, "gia tri thuan nguyen vat lieu", "VND_1e9"),
    792: D(792, "EIB", "MBB", 2022, "du phong tai san co khac", "VND_1e6"),
    793: SetSpec(793, "difference", (
        shares_million_item("GEE", 2025, "so luong co phieu pho thong dang luu hanh"),
        shares_million_item("GEX", 2025, "so luong co phieu pho thong dang luu hanh"),
    )),
    794: D(794, "NLG", "SCR", 2023, "doanh thu dich vu xay dung", "VND_1e9"),
    795: D(795, "GAS", "POW", 2019, "chi phi tra truoc ngan han", "VND_1e9", "separate", "separate"),
    796: R(796, "SGB", "NVB", 2021, "tra lai tien gui", "chi phi lai va cac chi phi tuong tu"),
    797: D(797, "ACV", "VJC", 2019, "chi phi nhan vien", "VND_1e9"),
    798: D(798, "HHV", "VSC", 2022, "gia tri con lai tai san co dinh huu hinh", "VND_1e9", "separate", "separate"),
    799: D(799, "HBC", "SAM", 2021, "von chu so huu", "VND_1e9", "separate", "separate"),
    800: D(800, "BID", "MSB", 2022, "tien gui va vay cac to chuc tin dung khac", "VND_1e6", "separate", "separate"),
    801: D(801, "VJC", "VSC", 2016, "phai tra nguoi ban ngan han", "VND_1e9"),
    802: D(802, "HDB", "EIB", 2024, "lai cho vay chua thu duoc", "VND_1e6", operation="absolute_difference"),
    803: D(803, "MBB", "SHB", 2015, "tien gui tai ngan hang nha nuoc viet nam", "VND_1e6", "separate", "separate"),
    804: D(804, "CRE", "SCR", 2023, "tien gui co ky han tai ngan hang", "VND_1e9"),
    805: D(805, "SCR", "DIG", 2018, "tong phai thu ngan han tu cac ben lien quan", "VND_1e9", "separate", "separate"),
    806: D(806, "NLG", "SCR", 2020, "ty le bieu quyet trung binh cua cac cong ty con", "percent"),
    807: D(807, "SHB", "SSB", 2021, "du phong rui ro chung khoan san sang de ban", "VND_1e6", "separate", "separate"),
    808: D(808, "DCM", "DPM", 2025, "tong tai san", "VND_1e9"),
    809: D(809, "DNH", "HND", 2025, "thue thu nhap doanh nghiep phai nop trong nam", "VND_1e9", "separate", "consolidated"),
    810: SetSpec(810, "difference", (
        direct_item("VCB", "VCB", 2025, "chi phi lai tien gui", "VND_1e6", "separate"),
        direct_item("VPB", "VPB", 2025, "tra lai tien gui", "VND_1e6", "separate"),
    )),
    811: D(811, "HAG", "BAF", 2024, "loi nhuan thuan sau thue", "VND_1e9"),
    812: D(812, "VNM", "SAB", 2016, "gia tri con lai tai san co dinh huu hinh", "VND_1e12"),
}


validate_specs(CROSS_SPECS, FIRST_ID, LAST_ID)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l2-cross-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l2_cross_manual_overrides.csv"))
    submission = sub.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l2-cross-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-formula-submission/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-complete-submission"))
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--plans", type=Path, default=Path("outputs/l2-cross-facts/plans.jsonl"))
    validate.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-formula-submission/submission.json"))
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        run_candidates(CROSS_SPECS, args.questions, args.database, args.output_dir, args.overrides)
    elif args.command == "submission":
        build_submission(args.plans, CROSS_SPECS, args.database, args.base_submission, args.output_dir)
    else:
        print(validate_submission(args.submission, args.plans, CROSS_SPECS,
                                  args.base_submission, args.zip_path))


if __name__ == "__main__":
    main()
