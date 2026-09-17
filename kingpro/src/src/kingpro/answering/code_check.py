"""code_intersect — chặn code pandas tham chiếu CỘT BỊA trước khi chạy (đòn nextgen).

Bóc mọi tên cột (chuỗi) mà code dùng làm khoá subscript rồi giao với schema thật.
Cột nào không có trong schema -> REJECT (đưa lỗi ngược cho model sửa, khỏi tốn 1 lần chạy).
Chỉ tính khoá dạng CHUỖI (df['x'], df.loc[..,'x']); .iloc[..] vị trí thì bỏ qua (luôn hợp lệ).
"""

from __future__ import annotations

import ast


def columns_used(code: str) -> set[str]:
    """Tập tên cột (chuỗi) code tham chiếu qua subscript."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    cols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            # dfs['table_ref'] là CHỌN BẢNG (khoá dict), không phải cột -> bỏ qua
            if isinstance(node.value, ast.Name) and node.value.id == "dfs":
                continue
            sl = node.slice
            # df['col']
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                cols.add(sl.value)
            # df[['a','b']]
            elif isinstance(sl, (ast.List, ast.Tuple)):
                for el in sl.elts:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        cols.add(el.value)
    return cols


def check_columns(code: str, schema_columns) -> tuple[bool, list[str]]:
    """(ok, cột_bịa). schema_columns = tập tên cột thật (mọi bảng đưa cho model)."""
    schema = {str(c) for c in schema_columns}
    used = columns_used(code)
    bad = sorted(c for c in used if c not in schema)
    return (len(bad) == 0, bad)


if __name__ == "__main__":
    schema = ["Chỉ tiêu", "Mã số", "Số cuối kỳ", "Số đầu năm"]
    ok1, bad1 = check_columns("result = df[df['Mã số']=='270']['Số cuối kỳ'].values[0]", schema)
    ok2, bad2 = check_columns("result = df[df['ma_so']=='270']['doanh_thu'].values[0]", schema)
    ok3, bad3 = check_columns("result = df.iloc[3, 2]", schema)  # vị trí -> luôn ok
    print("ca dung (chi cot that):", ok1, bad1)
    print("ca bia cot         :", ok2, bad2)
    print("ca iloc vi tri     :", ok3, bad3)
    assert ok1 and not bad1
    assert (not ok2) and bad2 == ["doanh_thu", "ma_so"]
    assert ok3 and not bad3
    print("code_check OK")
