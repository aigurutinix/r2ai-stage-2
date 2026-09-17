"""Auditable L3 set aggregation plans for ViFinQA q0813-q1012."""

from __future__ import annotations

import argparse
from pathlib import Path

from .question_taxonomy import block_for, operation_for
from .set_reasoning import (
    ItemSpec,
    SetSpec,
    absolute_ratio_item,
    build_submission,
    direct_item,
    fact,
    ratio_item,
    run_candidates,
    validate_specs,
    validate_submission,
)


FIRST_ID = 813
LAST_ID = 1012

OPERATION_MAP = {
    "set_argmax_key": "argmax_key",
    "set_max_value": "max_value",
    "set_sum": "sum",
    "set_mean": "mean",
    "set_aggregate_ratio": "aggregate_ratio",
}


def op(question_id: int) -> str:
    taxonomy_op = operation_for(question_id, block_for(question_id).solver_family)
    if taxonomy_op == "set_count":
        raise ValueError(f"q{question_id}: count operation must be explicit")
    return OPERATION_MAP[taxonomy_op]


def Y(question_id: int, ticker: str, years: tuple[int, ...], metric: str,
      unit: str = "VND_1", scope: str = "consolidated",
      period: str = "end_or_flow") -> SetSpec:
    return SetSpec(question_id, op(question_id), tuple(
        direct_item(year, ticker, year, metric, unit, scope, period) for year in years
    ))


def E(question_id: int, tickers: tuple[str, ...], year: int, metric: str,
      unit: str = "VND_1", scope: str = "consolidated",
      period: str = "end_or_flow") -> SetSpec:
    return SetSpec(question_id, op(question_id), tuple(
        direct_item(ticker, ticker, year, metric, unit, scope, period) for ticker in tickers
    ))


def YR(question_id: int, ticker: str, years: tuple[int, ...], numerator: str,
       denominator: str, scope: str = "consolidated",
       multiplier: float = 100.0, absolute: bool = True) -> SetSpec:
    factory = absolute_ratio_item if absolute else ratio_item
    return SetSpec(question_id, op(question_id), tuple(
        factory(year, ticker, year, numerator, denominator, scope,
                multiplier=multiplier) for year in years
    ))


def ER(question_id: int, tickers: tuple[str, ...], year: int, numerator: str,
       denominator: str, scope: str = "consolidated",
       multiplier: float = 100.0, absolute: bool = True) -> SetSpec:
    factory = absolute_ratio_item if absolute else ratio_item
    return SetSpec(question_id, op(question_id), tuple(
        factory(ticker, ticker, year, numerator, denominator, scope,
                multiplier=multiplier) for ticker in tickers
    ))


def count_years(question_id: int, ticker: str, years: tuple[int, ...], metric: str,
                unit: str, operation: str, threshold: float | None = None,
                scope: str = "consolidated") -> SetSpec:
    return SetSpec(question_id, operation, tuple(
        direct_item(year, ticker, year, metric, unit, scope) for year in years
    ), threshold)


def count_entities(question_id: int, tickers: tuple[str, ...], year: int,
                   metric: str, unit: str, operation: str,
                   threshold: float | None = None,
                   scope: str = "consolidated") -> SetSpec:
    return SetSpec(question_id, operation, tuple(
        direct_item(ticker, ticker, year, metric, unit, scope) for ticker in tickers
    ), threshold)


L3_SPECS = {
    813: Y(813, "ACV", (2015, 2017, 2019, 2023), "doanh thu thuan ban hang va cung cap dich vu", scope="separate"),
    814: Y(814, "GAS", (2020, 2022, 2024), "chi phi vo binh gas", "VND_1e9", "separate"),
    815: Y(815, "OCB", (2017, 2020, 2021, 2022), "lai thuan tu hoat dong kinh doanh ngoai hoi", "VND_1e9"),
    816: Y(816, "MSN", (2017, 2020, 2022), "du no vay ngan han", "VND_1e12"),
    817: Y(817, "HPG", (2015, 2019, 2022, 2023, 2024), "chi phi khau hao nha cua", "VND_1e9", "separate"),
    818: Y(818, "FTS", (2018, 2020, 2021, 2023, 2024), "tong du phong phai thu kho doi", "VND_1e6"),
    819: E(819, ("MSB", "BID", "ABB"), 2024, "chi phi trich lap du phong rui ro tin dung cho vay khach hang", "VND_1e9", "separate"),
    820: Y(820, "SNZ", (2018, 2019, 2020, 2021, 2022), "tong doanh thu hoat dong tai chinh", "VND_1e11", "separate"),
    821: YR(821, "OCB", (2017, 2020, 2024, 2025), "du no cho vay nganh bat dong san", "tong du no cho vay khach hang", "separate"),
    822: Y(822, "ASM", (2016, 2021, 2022, 2024), "thu nhap khac", scope="separate"),
    823: SetSpec(823, "count_gt", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "HSG", year, "trich lap quy khen thuong phuc loi", "VND_1e9"),
        )) for year in (2019, 2021, 2024, 2025)
    ), 40.0),
    824: SetSpec(824, "mean", tuple(
        ItemSpec(str(year), "a / 1000000.0", (
            fact("a", "VJC", year, "so luong co phieu pho thong dang luu hanh", "shares"),
        )) for year in (2016, 2019, 2021)
    )),
    825: Y(825, "POW", (2017, 2018, 2021, 2024), "phai tra ngan han tap doan dien luc viet nam", "VND_1e9", "separate"),
    826: YR(826, "KBC", (2016, 2019, 2020, 2022), "gia von cho thue dai han dat va co so ha tang", "tong gia von hang ban va dich vu cung cap"),
    827: E(827, ("SNZ", "VIC", "DXS", "HPX"), 2019, "tong thue va cac khoan phai nop nha nuoc", "VND_1e9", "separate"),
    828: SetSpec(828, op(828), tuple(
        ItemSpec(str(year), "abs(a) / (abs(b) + abs(c) + abs(d)) * 100.0", (
            fact("a", "ACB", year, "tong du phong rui ro cho vay khach hang", scope="separate"),
            fact("b", "ACB", year, "no duoi tieu chuan", scope="separate"),
            fact("c", "ACB", year, "no nghi ngo", scope="separate"),
            fact("d", "ACB", year, "no co kha nang mat von", scope="separate"),
        )) for year in (2017, 2020, 2021, 2022, 2024)
    )),
    829: Y(829, "TTF", (2016, 2017, 2023, 2025), "loi nhuan khac"),
    830: Y(830, "STB", (2017, 2021, 2022), "tong tien mat va vang", "VND_1e6", "separate"),
    831: E(831, ("DLG", "VJC", "ACV", "VSC"), 2015, "doanh thu hoat dong tai chinh", "VND_1e9"),
    832: Y(832, "MCH", (2017, 2018, 2019, 2022, 2025), "tien va cac khoan tuong duong tien", scope="separate"),
    833: SetSpec(833, "sum", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "VGT", year, "co tuc", "VND_1e11", "separate"),
        )) for year in (2016, 2024, 2025)
    )),
    834: Y(834, "GAS", (2015, 2016, 2017, 2025), "doanh thu thuan", "VND_1e12"),
    835: Y(835, "FOX", (2016, 2017, 2018, 2019, 2020), "tong no vay dai han", "VND_1e9", "separate"),
    836: SetSpec(836, "max_value", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "TTF", year, "du phong phai thu kho doi trich lap trong nam", "VND_1e9", "separate"),
        )) for year in (2017, 2019, 2021, 2025)
    )),
    837: count_years(837, "VSF", (2018, 2019, 2021, 2022, 2023), "doanh thu chua thuc hien nha so 2 dien bien phu", "VND_1", "count_nonzero"),
    838: Y(838, "MSN", (2017, 2018, 2020, 2021, 2024), "dau tu tai chinh dai han", "VND_1e12"),
    839: SetSpec(839, op(839), tuple(
        ItemSpec(str(year), "abs(a) / abs(b) * 100.0", (
            fact("a", "BSR", year, "du phong giam gia hang ton kho"),
            fact("b", "BSR", year, "tong hang ton kho gia goc"),
        )) for year in (2017, 2019, 2021, 2024, 2025)
    )),
    840: Y(840, "HPG", (2016, 2017, 2018, 2020, 2024), "tong chi phi tra truoc dai han", "VND_1e9", "separate"),
    841: Y(841, "VGT", (2017, 2018, 2021, 2022, 2023), "tong chi phi tra truoc dai han", scope="separate"),
    842: Y(842, "PDR", (2015, 2018, 2022), "loi nhuan truoc thue", scope="separate"),
    843: SetSpec(843, "sum", tuple(
        ItemSpec(str(year), "a * 1000000.0" if year == 2024 else "a", (
            fact("a", "EVF", year, "co tuc nhan duoc tu cac khoan dau tu", "VND_1e6"),
        )) for year in (2020, 2022, 2024)
    )),
    844: YR(844, "NLG", (2015, 2018, 2024), "von chu so huu", "tong nguon von"),
    845: Y(845, "BVH", (2017, 2019, 2022, 2024), "lai co ban tren co phieu", "VND_1e3"),
    846: YR(846, "HDG", (2020, 2021, 2022, 2023, 2025), "hao mon luy ke tai san co dinh huu hinh", "nguyen gia tai san co dinh huu hinh", "separate"),
    847: Y(847, "QNS", (2019, 2020, 2021, 2023, 2024), "gia tri con lai tai san co dinh vo hinh", "VND_1e9"),
    848: E(848, ("DXG", "DXS", "NLG"), 2021, "chi phi xay dung co ban do dang", "VND_1e9"),
    849: SetSpec(849, op(849), tuple(
        ItemSpec(str(year), "abs(a) / (abs(a) + abs(b) + abs(c) + abs(d) + abs(e)) * 100.0", (
            fact("a", "SAB", year, "chi phi khau hao va phan bo"),
            fact("b", "SAB", year, "chi phi nguyen vat lieu trong chi phi san xuat"),
            fact("c", "SAB", year, "chi phi nhan cong va nhan vien"),
            fact("d", "SAB", year, "chi phi dich vu mua ngoai"),
            fact("e", "SAB", year, "chi phi khac"),
        )) for year in (2018, 2020, 2024, 2025)
    )),
    850: Y(850, "TTF", (2017, 2018, 2020, 2022, 2024), "tong gia von hang ban va dich vu cung cap"),
    851: Y(851, "GEX", (2015, 2018, 2022, 2025), "thue thu nhap doanh nghiep phai nop trong nam", "VND_1e9", "separate"),
    852: Y(852, "BID", (2020, 2022, 2024, 2025), "cac khoan phai thu ben ngoai"),
    853: ER(853, ("DNH", "GEG", "POW"), 2022, "von chu so huu", "tong nguon von", "separate"),
    # "Doanh thu chưa thực hiện" is split between current and non-current
    # liabilities on NLG's balance sheet.  The question asks for the complete
    # ending balance, so each yearly observation must add both rows.
    854: SetSpec(854, op(854), tuple(
        ItemSpec(str(year), "a + b", (
            fact("a", "NLG", year, "doanh thu chua thuc hien ngan han", "VND_1e11"),
            fact("b", "NLG", year, "doanh thu chua thuc hien dai han", "VND_1e11"),
        )) for year in (2020, 2021, 2023)
    )),
    855: count_years(855, "HNG", (2015, 2016, 2021, 2022), "chi phi khau hao tai san ngung su dung", "VND_1", "count_positive"),
    856: Y(856, "PC1", (2015, 2020, 2022, 2023, 2024), "lai co ban tren co phieu", "VND_per_share"),
    857: Y(857, "EIB", (2016, 2017, 2018), "quy khen thuong va phuc loi", "VND_1e9"),
    858: E(858, ("SAB", "DBC", "MCH"), 2017, "chi phi ban hang", "VND_1e12", "separate"),
    859: Y(859, "DCM", (2022, 2023, 2024), "quy khen thuong phuc loi", "VND_1e9"),
    860: Y(860, "OGC", (2017, 2022, 2023), "tong chi phi khac"),
    861: E(861, ("OGC", "VNM", "HNG"), 2015, "so luong co phieu pho thong dang luu hanh", "shares"),
    862: SetSpec(862, "sum", tuple(
        ItemSpec(ticker, "abs(a)", (
            fact("a", ticker, 2016, "tong chi phi tai chinh", "VND_1e9"),
        )) for ticker in ("SCR", "DIG", "DXG", "KBC", "IJC")
    )),
    863: count_years(863, "HAG", (2015, 2019, 2021), "phai tra ngan han khac cong ty tnhh mtv kinh doanh xuat nhap khau hoang anh gia lai", "VND_1", "count_positive", scope="separate"),
    864: Y(864, "EIB", (2017, 2019, 2021, 2025), "quy khen thuong va phuc loi", "VND_1e11"),
    865: YR(865, "VIF", (2017, 2020, 2021, 2022), "du phong phai thu kho doi ngan han", "tong phai thu ngan han tu khach hang", "separate"),
    866: Y(866, "SJG", (2018, 2019, 2020, 2021), "tong thue va cac khoan phai nop nha nuoc"),
    867: Y(867, "HPX", (2022, 2023, 2025), "tong no phai tra", "VND_1e12"),
    868: SetSpec(868, "max_value", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "VCB", year, "thue thu nhap doanh nghiep hien hanh phai nop trong nam", "VND_1e9"),
        )) for year in (2015, 2017, 2018, 2022, 2023)
    )),
    869: SetSpec(869, "sum", tuple(
        ItemSpec(str(year), "a", (
            fact("a", "NVB", year, "luu chuyen tien thuan tu hoat dong tai chinh", "VND_1e12", "separate"),
        )) for year in (2015, 2016, 2019, 2024, 2025)
    )),
    870: count_years(870, "SAB", (2022, 2023, 2024), "luu chuyen tien thuan tu hoat dong dau tu", "VND_1", "count_negative"),
    871: SetSpec(871, "max_value", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "MPC", year, "trich lap quy khen thuong phuc loi trong nam", "VND_1e9"),
        )) for year in (2019, 2022, 2023, 2024)
    )),
    872: Y(872, "EVF", (2021, 2022, 2023, 2025), "gia tri con lai tai san co dinh vo hinh", "VND_1e9"),
    873: Y(873, "DXG", (2017, 2018, 2021, 2023, 2025), "tong chi phi thue thu nhap doanh nghiep", "VND_1e9"),
    874: Y(874, "VNM", (2015, 2016, 2021, 2023, 2025), "gia tri con lai tai san co dinh huu hinh"),
    875: Y(875, "STB", (2016, 2017, 2022, 2025), "tong tai san", "VND_1e12", "separate"),
    876: Y(876, "ACV", (2017, 2019, 2021, 2022), "chi phi lai vay"),
    877: YR(877, "HDB", (2023, 2024, 2025), "chi phi lai tien gui", "tong chi phi lai"),
    878: Y(878, "KBC", (2015, 2017, 2019), "gia tri con lai bat dong san dau tu nha xuong"),
    879: SetSpec(879, "argmax_key", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "VNM", year, "thue thu nhap doanh nghiep da nop", scope="separate"),
        )) for year in (2019, 2022, 2023, 2024, 2025)
    )),
    880: ER(880, ("MBB", "MSB", "STB"), 2022, "trai phieu chinh phu", "tong chung khoan no", "separate"),
    881: YR(881, "HND", (2016, 2018, 2019, 2020, 2021), "hao mon luy ke tai san co dinh huu hinh", "nguyen gia tai san co dinh huu hinh"),
    882: ER(882, ("NLG", "VIC", "DIG", "SNZ"), 2017, "tien thue phai thu trong tuong lai den han duoi mot nam", "tong tien thue trong tuong lai phai thu"),
    883: Y(883, "MBB", (2022, 2023, 2025), "chung khoan kinh doanh no"),
    884: Y(884, "HND", (2016, 2017, 2018, 2021, 2022), "tong no phai tra"),
    885: SetSpec(885, op(885), tuple(
        ItemSpec(str(year), "abs(a) / (abs(a) + abs(b) + abs(c)) * 100.0", (
            fact("a", "STB", year, "chung chi tien gui duoi 12 thang"),
            fact("b", "STB", year, "chung chi tien gui tu 12 thang den duoi 5 nam"),
            fact("c", "STB", year, "chung chi tien gui tu 5 nam tro len"),
        )) for year in (2021, 2022, 2025)
    )),
    886: Y(886, "VIB", (2015, 2022, 2023), "tong du phong rui ro cho vay khach hang", "VND_1e6"),
    887: Y(887, "VIC", (2022, 2023, 2024), "tong thu lao hoi dong quan tri", "VND_1e6", "separate"),
    888: E(888, ("SHB", "SSB", "CTG", "STB", "EIB"), 2021, "trai phieu chinh phu", "VND_1e12", "separate"),
    889: Y(889, "FTS", (2019, 2020, 2023, 2024), "gia tri con lai tai san co dinh vo hinh"),
    890: SetSpec(890, "argmax_key", tuple(
        ItemSpec(str(year), "abs(a)", (fact("a", "NVL", year, "tra truoc cho nguoi ban ngan han", scope="separate"),))
        for year in (2017, 2021, 2023)
    )),
    891: E(891, ("HBC", "GEX", "VGC", "SJG"), 2022, "chi phi lai vay ngan han phai tra", "VND_1e9"),
    892: YR(892, "HAG", (2015, 2016, 2017, 2022), "doanh thu khu vuc lao", "tong doanh thu"),
    893: Y(893, "HDG", (2016, 2017, 2018, 2019), "doanh thu ban hang hoa va dich vu cho cac ben lien quan", "VND_1e9", "separate"),
    894: SetSpec(894, "mean", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "VCB", year, "thue thu nhap doanh nghiep da nop", "VND_1e6", "separate"),
        )) for year in (2018, 2020, 2021, 2022, 2025)
    )),
    895: Y(895, "VIC", (2015, 2019, 2022, 2025), "tong tien thue toi thieu phai nhan theo hop dong thue hoat dong", "VND_1e9"),
    896: SetSpec(896, "mean", tuple(
        ItemSpec(str(year), "a", (
            fact("a", "TTF", year, "vay ben lien quan ngan han", "VND_1e9", "separate"),
        )) for year in (2018, 2021, 2022, 2024)
    )),
    897: Y(897, "DXG", (2016, 2018, 2020), "tong tien va cac khoan tuong duong tien"),
    898: E(898, ("DIG", "VRE", "PDR"), 2024, "phai tra ngan han khac cac ben lien quan", "VND_1e6", "separate"),
    899: Y(899, "MPC", (2018, 2020, 2022, 2023), "dau tu vao cong ty con", "VND_1e12", "separate"),
    900: Y(900, "MSN", (2016, 2017, 2019, 2021), "von co phan da phat hanh co phieu pho thong", scope="separate"),
    901: Y(901, "KLB", (2016, 2017, 2018, 2019, 2020), "gia tri con lai tai san co dinh vo hinh", "VND_1e6", "separate"),
    902: Y(902, "VPI", (2016, 2017, 2018), "tong tien va cac khoan tuong duong tien", "VND_1e9"),
    903: Y(903, "MPC", (2017, 2018, 2020, 2021, 2022), "chi phi khac", "VND_1e11"),
    904: Y(904, "HDG", (2020, 2023, 2024), "tong doanh thu hoat dong tai chinh"),
    905: Y(905, "VIC", (2020, 2021, 2023, 2025), "tong doanh thu hoat dong tai chinh", "VND_1e12"),
    906: Y(906, "VGT", (2015, 2017, 2018, 2020, 2022), "tai san co dinh huu hinh", scope="separate"),
    907: Y(907, "NVB", (2016, 2019, 2020, 2021), "so luong co phieu pho thong dang luu hanh", "shares"),
    908: Y(908, "ACB", (2016, 2017, 2019, 2023, 2025), "xay dung co ban do dang", "VND_1e12"),
    909: Y(909, "GEE", (2022, 2023, 2024, 2025), "phai thu ve cho vay ngan han cong ty day dong viet nam cft", "VND_1e9", "separate"),
    910: SetSpec(910, "argmax_key", tuple(
        ItemSpec(str(year), "a", (
            fact("a", "DBC", year, "hang hoa ton kho", scope="separate"),
        )) for year in (2015, 2016, 2019, 2022, 2025)
    )),
    911: Y(911, "VIF", (2017, 2020, 2021, 2022, 2024), "gia tri con lai tai san co dinh huu hinh", "VND_1e9", "separate"),
    912: Y(912, "VIF", (2022, 2024, 2025), "tong thu lao hoi dong quan tri va ban tong giam doc", "VND_1e9", "separate"),
    913: count_entities(913, ("VPI", "PDR", "DXS"), 2024, "no ngan han voi ben lien quan", "VND_1e9", "count_gt", 1.0, "separate"),
    914: Y(914, "SJG", (2019, 2020, 2021, 2022, 2023), "lai co ban tren co phieu", "VND_1e3"),
    915: E(915, ("MPC", "SAB", "HAG"), 2016, "tong chi phi tai chinh", "VND_1e9", "separate"),
    916: Y(916, "FOX", (2016, 2018, 2019, 2020), "chi phi tra truoc ngan han", "VND_1e9", "separate"),
    917: E(917, ("VSC", "VJC", "ACV"), 2017, "tong phai tra nguoi ban ngan han", "VND_1e6", "separate"),
    918: E(918, ("ABB", "SSB", "BID", "MBB"), 2024, "cam ket lc", "VND_1e6", "separate"),
    919: E(919, ("HNG", "HAG", "MPC"), 2019, "phai tra nguoi ban ngan han", "VND_1e6"),
    920: E(920, ("VIF", "GVR", "DPM"), 2025, "dau tu vao cong ty lien ket", "VND_1e9"),
    921: Y(921, "SAB", (2019, 2021, 2025), "phai tra cong ty lien doanh tnhh crown sai gon", scope="separate"),
    922: E(922, ("KHG", "PDR", "HPX", "DXS"), 2025, "chi phi tai chinh", "VND_1e11"),
    923: Y(923, "OGC", (2016, 2017, 2018, 2019), "chi phi nhan cong", "VND_1e9"),
    924: Y(924, "DXG", (2021, 2022, 2023, 2024, 2025), "chi phi khau hao bat dong san dau tu", "VND_1e9", "separate"),
    925: Y(925, "VAB", (2020, 2024, 2025), "von dieu le"),
    926: ER(926, ("MSN", "MML", "MPC", "VNM"), 2020, "thanh pham", "tong hang ton kho gia goc"),
    927: E(927, ("VPI", "DIG", "VRE", "DXG", "PDR"), 2016, "thue va cac khoan phai nop nha nuoc", "VND_1e9", "separate"),
    928: Y(928, "OCB", (2017, 2018, 2019, 2025), "chi phi trich lap du phong cu the cho vay khach hang", scope="separate"),
    929: Y(929, "MBB", (2015, 2016, 2017, 2018, 2022), "dau tu trai phieu chinh phu"),
    930: Y(930, "PC1", (2022, 2023, 2024, 2025), "phai thu co tuc va loi nhuan duoc chia ngan han", "VND_1e9", "separate"),
    931: ER(931, ("GEE", "GEX", "VGC", "SAM", "PC1"), 2025, "hao mon khau hao luy ke tai san co dinh", "nguyen gia tai san co dinh", "separate"),
    932: count_entities(932, ("MBB", "HDB", "KLB", "NAB"), 2020, "cam ket thue hoat dong den han trong mot nam", "VND_1e9", "count_gt", 40.0, "separate"),
    933: Y(933, "VNM", (2015, 2016, 2018, 2021), "tien chi tra co tuc"),
    934: E(934, ("VGC", "PC1", "SJG"), 2018, "doanh thu hoat dong tai chinh", "VND_1e9"),
    935: ER(935, ("HDG", "GEG", "DNH"), 2023, "trich lap quy khen thuong phuc loi", "loi nhuan thuan", "separate"),
    936: Y(936, "CRE", (2020, 2021, 2022, 2025), "tong doanh thu chua thuc hien ngan han"),
    937: Y(937, "CTG", (2017, 2018, 2019, 2020), "chung khoan no", "VND_1e6"),
    938: E(938, ("MSB", "EIB", "STB"), 2018, "cho vay khach hang", "VND_1e6"),
    939: ER(939, ("PNJ", "MWG", "HHS", "HUT"), 2022, "hao mon luy ke tai san co dinh huu hinh", "nguyen gia tai san co dinh huu hinh"),
    940: Y(940, "SJG", (2019, 2020, 2021), "doanh thu hoat dong tai chinh", "VND_1e6"),
    941: Y(941, "NLG", (2017, 2018, 2021, 2025), "phai tra ngan han khac voi cong ty con", "VND_1e9", "separate"),
    942: E(942, ("VPB", "EIB", "HDB"), 2020, "lai thuan tu hoat dong kinh doanh ngoai hoi", "VND_1e6", "separate"),
    943: E(943, ("DCM", "VIF", "NKG"), 2018, "tien va cac khoan tuong duong tien", "VND_1e9"),
    944: Y(944, "DLG", (2020, 2021, 2022, 2023), "chi phi xay dung co ban do dang", "VND_1e9"),
    945: ER(945, ("MSR", "GVR", "AAA"), 2023, "ngoai te usd ngoai bang", "tong ngoai te ngoai bang"),
    946: Y(946, "NVL", (2018, 2020, 2022), "chi phi xay dung co ban do dang"),
    947: Y(947, "IJC", (2016, 2017, 2020, 2024), "tong gia von hang ban", "VND_1e9"),
    948: Y(948, "PLX", (2017, 2018, 2019, 2023, 2024), "tong xay dung co ban do dang"),
    949: count_entities(949, ("ACB", "MBB", "EIB", "BID"), 2025, "lai thuan tu hoat dong kinh doanh ngoai hoi", "VND_1e9", "count_gt", 1000.0),
    # The related-party note spans several tables and has no printed grand
    # total, so sum every numeric current-year cell directly from each cited
    # source CSV at execution time.
    950: SetSpec(950, op(950), (
        ItemSpec("2019", "sumcol(a) + sumcol(b) + sumcol(c)", (
            fact("a", "VRE", 2019, "gia tri giao dich ben lien quan phan 1", scope="separate"),
            fact("b", "VRE", 2019, "gia tri giao dich ben lien quan phan 2", scope="separate"),
            fact("c", "VRE", 2019, "gia tri giao dich ben lien quan phan 3", scope="separate"),
        )),
        ItemSpec("2020", "sumcol(a) + sumcol(b) + sumcol(c)", (
            fact("a", "VRE", 2020, "gia tri giao dich ben lien quan phan 1", scope="separate"),
            fact("b", "VRE", 2020, "gia tri giao dich ben lien quan phan 2", scope="separate"),
            fact("c", "VRE", 2020, "gia tri giao dich ben lien quan phan 3", scope="separate"),
        )),
        ItemSpec("2021", "sumcol(a) + sumcol(b)", (
            fact("a", "VRE", 2021, "gia tri giao dich ben lien quan phan 1", scope="separate"),
            fact("b", "VRE", 2021, "gia tri giao dich ben lien quan phan 2", scope="separate"),
        )),
        ItemSpec("2022", "sumcol(a) + sumcol(b) + sumcol(c)", (
            fact("a", "VRE", 2022, "gia tri giao dich ben lien quan phan 1", scope="separate"),
            fact("b", "VRE", 2022, "gia tri giao dich ben lien quan phan 2", scope="separate"),
            fact("c", "VRE", 2022, "gia tri giao dich ben lien quan phan 3", scope="separate"),
        )),
    )),
    951: Y(951, "VCB", (2018, 2019, 2025), "thu nhap binh quan thang nguoi", "VND_1e6"),
    952: SetSpec(952, "mean", tuple(
        direct_item(year, "DTK", year, "phai thu khac ngan han", "VND_1e9",
                    "separate" if year in (2022, 2023) else "consolidated")
        for year in (2022, 2023, 2024, 2025)
    )),
    953: Y(953, "CTG", (2016, 2021, 2022), "tong gia tri con lai tai san co dinh vo hinh", scope="separate"),
    954: E(954, ("DPM", "VIF", "HSG"), 2018, "chi phi lai vay", "VND_1e9"),
    955: ER(955, ("ACB", "MBB", "BID", "STB"), 2019, "cho vay ngan han", "tong du no cho vay khach hang", "separate"),
    956: Y(956, "SHB", (2016, 2020, 2022, 2024), "loi nhuan truoc thue khu vuc mien bac", "VND_1e6"),
    957: ER(957, ("EIB", "STB", "SSB"), 2020, "tai san co ky han dinh lai lai suat 1 3 thang", "tong tai san"),
    958: E(958, ("VIC", "DIG", "DXG"), 2018, "vay dai han tu ngan hang", "VND_1e12"),
    959: Y(959, "GVR", (2016, 2017, 2018, 2020), "von chu so huu"),
    960: Y(960, "QNS", (2017, 2019, 2020, 2021, 2023), "luu chuyen tien thuan tu hoat dong kinh doanh", scope="separate"),
    961: Y(961, "BAF", (2020, 2022, 2024, 2025), "tong hang ton kho", "VND_1e9", "separate"),
    962: YR(962, "DNH", (2016, 2017, 2018, 2021, 2022), "no phai tra", "von chu so huu", "separate", multiplier=1.0),
    963: count_entities(963, ("PC1", "VGC", "SAM"), 2016, "tien va tuong duong tien ngan han", "VND_1e9", "count_gt", 100.0, "separate"),
    964: E(964, ("KHG", "CRE", "KBC"), 2023, "chi phi lai vay ngan han phai tra", "VND_1e9"),
    965: count_entities(965, ("SSH", "DXS", "CEO"), 2022, "chi phi khong duoc tru khi tinh thue thu nhap doanh nghiep", "VND_1e9", "count_gt", 20.0, "separate"),
    966: count_entities(966, ("HDG", "GEG", "DNH"), 2025, "luu chuyen tien thuan tu hoat dong kinh doanh", "VND_1e12", "count_gt", 1.0),
    967: Y(967, "BID", (2017, 2021, 2023, 2024, 2025), "tong du phong rui ro cho vay khach hang", "VND_1e6"),
    968: YR(968, "POW", (2017, 2019, 2022, 2023, 2024), "vay dai han bang usd", "tong vay dai han", "separate"),
    969: Y(969, "DNH", (2017, 2018, 2019), "gia von ban dien", "VND_1e9"),
    970: YR(970, "ACB", (2015, 2021, 2023, 2025), "quy du phong tai chinh", "von chu so huu"),
    971: Y(971, "KHG", (2019, 2020, 2021, 2022, 2023), "tong chi phi hoa hong moi gioi bat dong san", scope="separate"),
    972: Y(972, "DIG", (2021, 2023, 2025), "thu nhap khac", "VND_1e9", "separate"),
    973: count_entities(973, ("HSG", "HPG", "MSR"), 2022, "luu chuyen tien thuan tu hoat dong kinh doanh", "VND_1", "count_positive", scope="separate"),
    974: Y(974, "CTG", (2022, 2023, 2024, 2025), "tong tai san va chung tu giu ho bao quan"),
    975: E(975, ("MCH", "MSN", "VNM", "ASM"), 2024, "chi phi thue thu nhap doanh nghiep hien hanh", "VND_1e9"),
    976: ER(976, ("MSN", "MPC", "VNM", "MML"), 2022, "chi phi khau hao", "tong chi phi quan ly doanh nghiep", "separate"),
    977: ER(977, ("DPM", "AAA", "MSR"), 2017, "no dai han", "von chu so huu", "separate"),
    978: Y(978, "NVL", (2020, 2022, 2025), "chi phi xay dung phai tra ngan han"),
    979: SetSpec(979, "aggregate_ratio", tuple(
        absolute_ratio_item(ticker, ticker, 2022, "lai tien gui", "tong doanh thu hoat dong tai chinh")
        for ticker in ("POW", "GAS", "DTK", "GEG")
    )),
    980: YR(980, "HDB", (2018, 2021, 2022, 2024), "trang thai tien te noi bang thuan", "tong tai san", absolute=False),
    981: Y(981, "BSR", (2017, 2019, 2021, 2022, 2025), "doanh thu thuan san pham khi lpg"),
    982: YR(982, "PVT", (2018, 2019, 2022, 2023, 2025), "tai san bo phan dich vu van tai", "tong tai san", "separate"),
    983: count_entities(983, ("MCH", "MPC", "VSF"), 2019, "thue thu nhap doanh nghiep phat sinh", "VND_1e9", "count_gt", 50.0),
    984: Y(984, "PLX", (2019, 2020, 2023), "vay dai han den han trong nam", "VND_1e9"),
    985: Y(985, "FIT", (2016, 2017, 2019, 2020, 2021), "vay va no thue tai chinh ngan han", scope="separate"),
    986: SetSpec(986, "argmax_key", tuple(
        ItemSpec(str(year), "abs(a)", (
            fact("a", "HAG", year, "chi phi thue thu nhap doanh nghiep hien hanh", scope="separate"),
        )) for year in (2016, 2017, 2024, 2025)
    )),
    987: E(987, ("VNM", "MCH", "SAB", "MSN"), 2020, "lai lo chenh lech ty gia hoi doai", "VND_1e6", "separate"),
    988: Y(988, "SJG", (2019, 2021, 2022, 2023), "lai co ban tren co phieu", "VND_per_share"),
    989: Y(989, "STB", (2017, 2022, 2024), "lai du thu tu cho vay khach hang", "VND_1e6"),
    990: SetSpec(990, "count_gt", tuple(
        ItemSpec(ticker, "a / 1000000.0", (fact("a", ticker, 2021, "so luong co phieu dang luu hanh", "shares", "separate"),))
        for ticker in ("VPI", "NLG", "DXG", "SNZ")
    ), 350.0),
    991: SetSpec(991, "sum", tuple(
        ItemSpec(ticker, "abs(a)", (
            fact("a", ticker, 2016, "du phong giam gia chung khoan dau tu san sang de ban", "VND_1e6"),
        )) for ticker in ("SHB", "VIB", "BID", "CTG")
    )),
    992: ER(992, ("VPB", "SHB", "MBB"), 2022, "hao mon luy ke tai san co dinh huu hinh", "nguyen gia tai san co dinh huu hinh"),
    # GEE's 2022 separate income statement prints current income tax as a
    # dash.  Represent that zero from two independently printed source cells
    # (PBT - PAT) so the submitted Pandas query remains fully data-derived;
    # the generic numeric retriever cannot bind a dash-valued cell.
    993: SetSpec(993, op(993), (
        ItemSpec("GEE", "abs(b - c) / abs(b) * 100.0", (
            fact("b", "GEE", 2022, "loi nhuan ke toan truoc thue", scope="separate"),
            fact("c", "GEE", 2022, "loi nhuan sau thue thu nhap doanh nghiep", scope="separate"),
        )),
        *tuple(
            absolute_ratio_item(
                ticker, ticker, 2022,
                "chi phi thue thu nhap doanh nghiep hien hanh",
                "loi nhuan ke toan truoc thue",
                "separate",
            )
            for ticker in ("VGC", "SJG")
        ),
    )),
    994: Y(994, "QNS", (2019, 2020, 2021, 2022, 2023), "tong chi phi ban hang", "VND_1e9"),
    995: SetSpec(995, "argmax_key", (
        ItemSpec("2017", " + ".join(f"abs({name})" for name in "abcdefghijk"), tuple(
            fact(name, "HAG", 2017, "tong so du phai thu phai tra ben lien quan")
            for name in "abcdefghijk"
        )),
        ItemSpec("2018", " + ".join(f"abs({name})" for name in "abcdefghij"), tuple(
            fact(name, "HAG", 2018, "tong so du phai thu phai tra ben lien quan")
            for name in "abcdefghij"
        )),
        ItemSpec("2019", " + ".join(f"abs({name})" for name in "abcdefghi"), tuple(
            fact(name, "HAG", 2019, "tong so du phai thu phai tra ben lien quan")
            for name in "abcdefghi"
        )),
    )),
    996: Y(996, "SHB", (2016, 2018, 2020, 2021), "tong gia tri con lai tai san co dinh vo hinh", "VND_1e6"),
    997: Y(997, "AAA", (2019, 2023, 2025), "tra truoc cho nguoi ban ngan han"),
    998: Y(998, "HHS", (2016, 2017, 2018, 2019, 2021), "tien va cac khoan tuong duong tien", "VND_1e9", "separate"),
    999: Y(999, "VCB", (2016, 2020, 2025), "tong phat hanh giay to co gia", scope="separate"),
    1000: Y(1000, "SAM", (2020, 2021, 2023, 2024), "xay dung co ban do dang"),
    1001: ER(1001, ("BID", "NAB", "ABB"), 2023, "chi phi du phong rui ro tin dung", "loi nhuan truoc du phong"),
    1002: count_entities(1002, ("AAA", "NKG", "DCM", "DPM"), 2016, "chi phi lai vay", "VND_1e9", "count_gt", 100.0, "separate"),
    1003: E(1003, ("MSR", "HPG", "AAA", "DCM"), 2015, "luu chuyen tien thuan tu hoat dong kinh doanh", "VND_1e9", "separate"),
    1004: YR(1004, "IJC", (2016, 2017, 2018, 2023, 2024), "chi phi khau hao tai san co dinh", "tong chi phi quan ly doanh nghiep", "separate"),
    1005: SetSpec(1005, "count_gt", tuple(
        ItemSpec(ticker, "a / 1000000.0", (fact("a", ticker, 2020, "so luong co phieu pho thong dang luu hanh", "shares"),))
        for ticker in ("MWG", "HHS", "PNJ", "HUT")
    ), 400.0),
    1006: SetSpec(1006, op(1006), tuple(
        ItemSpec(ticker, "a" if ticker == "ACB" else "a * 12.0", (
            fact("a", ticker, 2025, "thu nhap binh quan cua nhan vien", "VND_1e6", "separate"),
        )) for ticker in ("NAB", "ABB", "ACB", "STB")
    )),
    1007: YR(1007, "HHV", (2021, 2022, 2024, 2025), "tai san bo phan bot", "tong tai san", "consolidated"),
    1008: Y(1008, "SSH", (2020, 2021, 2022, 2023), "tong doanh thu cung cap dich vu cho cac ben lien quan", scope="separate"),
    1009: ER(1009, ("OCB", "EIB", "MSB", "VPB"), 2018, "du phong chung", "tong du phong rui ro cho vay khach hang"),
    1010: count_entities(1010, ("NVB", "SGB", "VIB", "HDB", "MSB"), 2020, "chenh lech thanh khoan rong tong the", "VND_1", "count_positive"),
    1011: SetSpec(1011, "max_value", tuple(
        ItemSpec(str(year), "a", (
            fact("a", "SSB", year, "tien vay tai ngan hang", "VND_1e6", "separate"),
        )) for year in (2021, 2023, 2024)
    )),
    1012: E(1012, ("DTK", "GAS", "POW"), 2017, "phai thu ngan han tu cac ben lien quan", "VND_1e9"),
}


validate_specs(L3_SPECS, FIRST_ID, LAST_ID)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l3-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l3_manual_overrides.csv"))
    submission = sub.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l3-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-complete-submission/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-l3-submission"))
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--plans", type=Path, default=Path("outputs/l3-facts/plans.jsonl"))
    validate.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-complete-submission/submission.json"))
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        run_candidates(L3_SPECS, args.questions, args.database, args.output_dir, args.overrides)
    elif args.command == "submission":
        build_submission(args.plans, L3_SPECS, args.database, args.base_submission, args.output_dir)
    else:
        print(validate_submission(args.submission, args.plans, L3_SPECS,
                                  args.base_submission, args.zip_path))


if __name__ == "__main__":
    main()
