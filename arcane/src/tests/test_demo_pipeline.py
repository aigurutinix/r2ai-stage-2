import asyncio

import pytest

from vifinqa.demo_pipeline import DemoConfig, DemoPipeline, Plan, PlanFact, _validate_plan


def test_plan_rejects_unknown_entity_and_future_reference():
    payload = {
        "facts": [{"id": "f1", "query": "Doanh thu", "entity": "ABC", "period": 2024}],
        "steps": [{"id": "s1", "op": "add", "inputs": ["s2"], "instruction": "bad"}],
    }
    with pytest.raises(ValueError):
        _validate_plan(payload, {"tickers": ["ABC"], "years": [2024]})

    payload["steps"][0]["inputs"] = ["f1"]
    with pytest.raises(ValueError):
        _validate_plan(payload, {"tickers": ["XYZ"], "years": [2024]})


def test_fixture_pipeline_emits_result_and_provenance():
    async def collect():
        pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
        return [event async for event in pipeline.run("Tiền và tương đương tiền năm 2024", "req-1")], pipeline

    events, pipeline = asyncio.run(collect())
    result = next(event for event in events if event["type"] == "result")
    citation = result["data"]["citations"][0]
    assert result["data"]["checks"]["executed"] is True
    assert citation["table_id"] == "fixture_2024|1"
    evidence = pipeline.evidence("req-1", "c1")
    assert evidence["raw_value"] == "42"
    assert evidence["headers"] == ["Chỉ tiêu", "2024"]
    assert evidence["rows"] == [["Tiền và tương đương tiền", "42"]]
    assert evidence["highlight"] == {"row": 0, "column": 1}


def test_live_mode_requires_model_endpoints():
    with pytest.raises(RuntimeError, match="PLANNER_URL and RERANK_URL"):
        DemoPipeline(DemoConfig(data_root=None, mode="live"))


import pandas as pd

from vifinqa.demo_pipeline import (
    _apply_unit_scale,
    _cells_used,
    _citations_for,
    _frame_context,
)
from vifinqa.generated_query import GeneratedQueryError


FRAMES = {
    "df0": pd.DataFrame(
        [["Lãi tiền gửi", "208.253.201.298", "69.917.578.051"]],
        columns=["Chỉ tiêu", "2018VND", "2017VND"],
    )
}
SOURCES = {
    "df0": {
        "report_id": "VJC_financial_statements_2018_separate",
        "table_id": "VJC_financial_statements_2018_separate|1179",
        "unit": "VND",
    }
}


def test_cells_used_reads_only_integer_iloc_pairs():
    query = "result = float(df0.iloc[0, 1]) - float(df0.iloc[0, 2]) + float(df0.iloc[0, 1])"
    assert _cells_used(query, FRAMES) == [("df0", 0, 1), ("df0", 0, 2)]
    assert _cells_used("result = df0.shape[0]", FRAMES) == []


def test_citations_cover_the_cited_cells_and_nothing_else():
    query = "result = float(str(df0.iloc[0, 1]).replace('.', ''))"
    citations, evidence = _citations_for(query, FRAMES, SOURCES)
    assert [citation["id"] for citation in citations] == ["c1"]
    assert citations[0]["raw_value"] == "208.253.201.298"
    assert citations[0]["row_label"] == "Lãi tiền gửi"
    assert citations[0]["column_label"] == "2018VND"
    assert evidence["c1"]["highlight"] == {"row": 0, "column": 1}
    assert evidence["c1"]["headers"] == ["Chỉ tiêu", "2018VND", "2017VND"]
    assert evidence["c1"]["rows"][0][1] == "208.253.201.298"


def test_citations_drop_a_cell_outside_the_frame():
    assert _citations_for("result = float(df0.iloc[9, 9])", FRAMES, SOURCES) == ([], {})


def test_single_cell_lookup_is_rescaled_to_the_requested_unit():
    question = "Lãi tiền gửi năm 2018 của VJC là bao nhiêu tỷ đồng?"
    rescaled = _apply_unit_scale("result = float(df0.iloc[0, 1])", question, FRAMES)
    assert "1e-09" in rescaled
    from vifinqa.generated_query import execute_generated_query

    assert execute_generated_query(rescaled, FRAMES) == 208.25


def test_multi_cell_expression_is_left_alone_but_still_validated():
    question = "Chênh lệch lãi tiền gửi 2018 và 2017 là bao nhiêu?"
    query = "result = float(df0.iloc[0, 1].replace('.', '')) - float(df0.iloc[0, 2].replace('.', ''))"
    assert _apply_unit_scale(query, question, FRAMES) == query
    with pytest.raises(GeneratedQueryError):
        _apply_unit_scale("result = __import__('os')", question, FRAMES)


def test_frame_context_masks_numbers_and_emits_real_newlines():
    context = _frame_context({"evidence": [{"variable": "df0"}]}, FRAMES)
    assert "\\n" not in context and context.count("\n") >= 2
    assert "208.253.201.298" not in context
    assert "Lãi tiền gửi" in context


def test_solve_repairs_once_then_gives_up():
    replies = [
        {"expression": "result = float(df0.iloc[99, 0])"},
        {"expression": "result = float(str(df0.iloc[0, 1]).replace('.', ''))"},
    ]

    async def collect():
        pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))

        async def fake_chat(url, prompt, model):
            return replies.pop(0)

        pipeline._chat = fake_chat
        return await pipeline._solve("prompt", "Lãi tiền gửi năm 2018?", FRAMES)

    expression, answer, repaired = asyncio.run(collect())
    assert repaired is True
    assert answer == 208253201298.0

    replies[:] = [{"expression": "result = float(df0.iloc[99, 0])"}] * 2

    async def collect_failure():
        pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))

        async def fake_chat(url, prompt, model):
            return replies.pop(0)

        pipeline._chat = fake_chat
        return await pipeline._solve("prompt", "Lãi tiền gửi năm 2018?", FRAMES)

    with pytest.raises(GeneratedQueryError, match="after one repair"):
        asyncio.run(collect_failure())


def test_same_triple_reports_keeps_the_gate_triple_and_drops_other_scopes():
    from types import SimpleNamespace

    from vifinqa.demo_pipeline import same_triple_reports

    def report(ticker, year, scope):
        return SimpleNamespace(identity=SimpleNamespace(ticker=ticker, year=year, scope=scope))

    reports = {
        "AAA_2024_consolidated": report("AAA", 2024, "consolidated"),
        "AAA_2024_aggregated": report("AAA", 2024, "aggregated"),
        "AAA_2024_separate": report("AAA", 2024, "separate"),
        "BBB_2024_consolidated": report("BBB", 2024, "consolidated"),
    }
    assert same_triple_reports(reports, ["AAA_2024_consolidated"]) == {
        "AAA_2024_consolidated",
        "AAA_2024_aggregated",
    }
    assert same_triple_reports(reports, []) == set()


def test_health_reports_a_down_endpoint_as_not_ready(tmp_path, monkeypatch):
    (tmp_path / "code_stock.csv").write_text("ticker,name\n")
    (tmp_path / "financial_statements").mkdir()
    pipeline = DemoPipeline(
        DemoConfig(data_root=tmp_path, mode="live", planner_url="http://p", rerank_url="http://r")
    )

    async def probe():
        return {"planner": "up", "reranker": "down"}

    pipeline._probe_endpoints = probe
    pipeline._load_data = lambda: None
    pipeline._tables = ["a table"]
    pipeline._documents = {}
    pipeline._report_meta = {}
    report = asyncio.run(pipeline.health())
    assert report["ready"] is False
    assert report["endpoints"]["reranker"] == "down"
    assert report["local_execution"] is True
    assert report["dense"] == {"enabled": False}


def test_salvage_keeps_valid_facts_when_steps_are_malformed():
    from vifinqa.demo_pipeline import _salvage_facts

    payload = {
        "facts": [
            {"query": "Lãi tiền gửi", "entity": "VJC", "period": 2018},
            {"query": "Doanh thu", "entity": "OTHER", "period": 2018},
            {"noquery": True},
        ],
        "steps": [{"step": "đọc bảng"}],
    }
    facts = _salvage_facts(payload, {"tickers": ["VJC"], "years": [2018]})
    # the OTHER entity is coerced to the gate's ticker, not dropped; only the
    # entry with no query at all is unusable
    assert [fact.query for fact in facts] == ["Lãi tiền gửi", "Doanh thu"]
    assert [fact.entity for fact in facts] == ["VJC", "VJC"]
    assert [fact.id for fact in facts] == ["f1", "f2"]
    assert _salvage_facts({"facts": []}, {"tickers": ["VJC"], "years": [2018]}) == []
    assert _salvage_facts(None, {}) == []


def test_answer_prompt_states_the_valid_cell_bounds():
    from vifinqa.demo_pipeline import _answer_prompt

    plan = Plan(facts=[PlanFact(id="f1", query="Lãi tiền gửi", entity="VJC", period=2018)], steps=[])
    prompt = _answer_prompt("Lãi tiền gửi 2018?", plan, FRAMES, "TABLE df0")
    assert "df0: rows 0-0, columns 0-2" in prompt


def test_plan_falls_back_to_facts_when_every_attempt_fails_validation():
    bad = {"facts": [{"query": "Lãi tiền gửi", "entity": "VJC", "period": 2018}],
           "steps": [{"step": "không hợp lệ"}]}

    async def collect():
        pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))

        async def fake_chat(url, prompt, model):
            pipeline._last_plan_payload = bad
            return bad

        pipeline._chat = fake_chat
        return await pipeline._plan_or_facts("Lãi tiền gửi 2018?", {"tickers": ["VJC"], "years": [2018]})

    plan, degraded = asyncio.run(collect())
    assert degraded is True
    assert [fact.query for fact in plan.facts] == ["Lãi tiền gửi"]
    assert plan.steps == []


def test_plan_still_raises_when_nothing_can_be_salvaged():
    async def collect():
        pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))

        async def fake_chat(url, prompt, model):
            return {"nonsense": True}

        pipeline._chat = fake_chat
        return await pipeline._plan_or_facts("Lãi tiền gửi 2018?", {"tickers": ["VJC"], "years": [2018]})

    with pytest.raises(Exception):
        asyncio.run(collect())


def test_bare_expression_is_normalised_but_an_assignment_is_untouched():
    from vifinqa.demo_pipeline import _as_assignment

    assert _as_assignment("(df0.iloc[0, 1]) * 10") == "result = (df0.iloc[0, 1]) * 10"
    assert _as_assignment("result = df0.iloc[0, 1]") == "result = df0.iloc[0, 1]"
    assert _as_assignment("```python\ndf0.iloc[0, 1]\n```") == "result = df0.iloc[0, 1]"
    assert _as_assignment("import os\nresult = 1") == "import os\nresult = 1"
    assert _as_assignment("") == ""
    assert _as_assignment(None) == ""


def test_out_of_bounds_cell_is_rejected_with_the_real_range():
    with pytest.raises(GeneratedQueryError, match=r"df0 has rows 0-0 and columns 0-2"):
        _apply_unit_scale("result = float(df0.iloc[3, 1])", "Lãi tiền gửi 2018?", FRAMES)


def test_pool_depth_is_flat_by_default_and_scales_only_when_asked():
    parity = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    assert [parity._pool_depth(n) for n in (0, 1, 10, 33)] == [50, 50, 50, 50]

    scaled = DemoPipeline(DemoConfig(data_root=None, mode="fixture", depth_per_report=8))
    assert scaled._pool_depth(1) == 50        # never below the flat floor
    assert scaled._pool_depth(10) == 80
    assert scaled._pool_depth(33) == 200      # capped so the rerank batch stays bounded


def test_line_item_lexicon_is_sorted_so_substrings_are_pruned(tmp_path):
    from vifinqa.slots import named_line_items

    path = tmp_path / "line_items.json"
    path.write_text('{"cho vay": 1, "cho vay khach hang": 1, "thuong mai": 1}', encoding="utf-8")
    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture", line_items_path=path))
    labels = pipeline._labels()
    assert labels[0] == "cho vay khach hang"          # longest first
    assert named_line_items("So du cho vay khach hang nganh Thuong mai", labels) == [
        "cho vay khach hang", "thuong mai",
    ]


def test_note_links_are_off_by_default_and_opt_in():
    off = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    assert off.config.note_max_links == 0
    assert off._note_links({"candidates": []}, ["a|1"], ["lai tien gui"]) == []

    on = DemoPipeline(DemoConfig(data_root=None, mode="fixture", note_max_links=3))
    assert on.config.note_max_links == 3


def test_retrieve_refuses_an_empty_gate_instead_of_scanning_the_corpus():
    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    pipeline._load_data = lambda: None
    pipeline._documents, pipeline._tables, pipeline._report_meta = {}, [], {}

    class Parsed:
        tickers, years, scope = [], [], "consolidated"

    plan = Plan(facts=[PlanFact(id="f1", query="lai tien gui", entity="", period=None)], steps=[])
    with pytest.raises(ValueError, match="No filing matches"):
        pipeline._retrieve("cau hoi khong co ma co phieu", plan, Parsed())


def test_salvage_coerces_a_wrong_entity_and_period_to_the_gate_values():
    from vifinqa.demo_pipeline import _salvage_facts

    payload = {"facts": [{"query": "Lãi tiền gửi", "entity": "WRONG", "period": 1999}]}
    facts = _salvage_facts(payload, {"tickers": ["VJC"], "years": [2018], "slot_years": [2018]})
    assert [(f.entity, f.period) for f in facts] == [("VJC", None)]


def test_error_event_never_has_an_empty_message():
    import httpx

    async def collect():
        pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
        pipeline.config = DemoConfig(data_root=None, mode="live", planner_url="x", rerank_url="x")

        def boom():
            raise httpx.ReadTimeout("")          # str() of this is ""

        pipeline._load_data = boom
        return [e async for e in pipeline.run("Lãi tiền gửi VJC 2018?", "r")]

    events = asyncio.run(collect())
    error = next(e for e in events if e["type"] == "error")
    assert error["message"] == "ReadTimeout"
    assert error["data"]["error_type"] == "ReadTimeout"


def test_table_preview_returns_headers_rows_and_source():
    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    table = {
        "table_id": "VJC_..._2018_separate|1179", "report_id": "VJC_..._2018_separate",
        "title": "Thuyết minh", "unit": "VND",
        "rows": [["Chỉ tiêu", "2018VND"], ["Lãi tiền gửi", "208.253.201.298"]],
    }
    pipeline._remember_tables("r1", [table])
    preview = pipeline.table_preview("r1", "VJC_..._2018_separate|1179")
    assert preview["headers"] == ["Chỉ tiêu", "2018VND"]
    assert preview["rows"] == [["Lãi tiền gửi", "208.253.201.298"]]
    assert preview["report_id"] == "VJC_..._2018_separate"
    assert preview["truncated"] is False


def test_table_preview_is_none_for_an_unknown_table():
    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    assert pipeline.table_preview("nope", "nope|1") is None


def test_table_card_is_a_small_id_report_title_triple():
    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    card = pipeline._table_card({"table_id": "a|1", "report_id": "a", "title": "T", "rows": [["x"]]})
    assert card == {"table_id": "a|1", "report_id": "a", "title": "T"}


def test_table_card_carries_a_real_score_only_when_one_exists():
    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    table = {"table_id": "a|1", "report_id": "a", "title": "T"}
    assert "score" not in pipeline._table_card(table)
    assert pipeline._table_card(table, 0.94123)["score"] == 0.9412


def test_probe_reports_busy_separately_from_down():
    import httpx
    import vifinqa.demo_pipeline as module

    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    pipeline.config = DemoConfig(data_root=None, mode="live", planner_url="http://p", rerank_url="http://r")

    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            if url.startswith("http://p"):
                raise httpx.ReadTimeout("still working")
            raise httpx.ConnectError("refused")

    original = httpx.AsyncClient
    httpx.AsyncClient = lambda **kw: Client()
    try:
        assert asyncio.run(pipeline._probe_endpoints()) == {"planner": "busy", "reranker": "down"}
    finally:
        httpx.AsyncClient = original


def test_a_busy_endpoint_still_counts_as_ready(tmp_path):
    pipeline = DemoPipeline(DemoConfig(data_root=None, mode="fixture"))
    pipeline.config = DemoConfig(data_root=None, mode="live", planner_url="http://p", rerank_url="http://r")

    async def probe():
        return {"planner": "up", "reranker": "busy"}

    pipeline._probe_endpoints = probe
    pipeline._load_data = lambda: None
    pipeline._tables, pipeline._documents, pipeline._report_meta = ["t"], {}, {}
    report = asyncio.run(pipeline.health())
    # A model server mid-inference is working, not broken.
    assert report["ready"] is True
    assert report["endpoints"]["reranker"] == "busy"
