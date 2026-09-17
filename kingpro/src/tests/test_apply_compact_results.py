from scripts.apply_approved_compact_results import parse_ids


def test_parse_approved_ids() -> None:
    assert parse_ids("508, 512") == [508, 512]
