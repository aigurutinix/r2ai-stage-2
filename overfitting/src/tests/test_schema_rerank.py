from vifinqa.retrieval.schema_rerank import rerank_codegen_tail


def _candidate(pos, *, hits=(), req_score=0.0, score=0.0):
    return {
        "report_id": "AAA_2024_consolidated", "table_pos": pos,
        "requirement_hits": list(hits),
        "requirement_scores": {hit: req_score for hit in hits},
        "score": score, "row_score": 0.0, "label_match": 0.0,
    }


def test_freezes_submission_head_and_promotes_missing_requirement():
    candidates = [_candidate(i, score=10 - i) for i in range(8)]
    candidates[7] = _candidate(7, hits=("req-1",), req_score=99.0)
    record = {
        "id": 1,
        "route": {"evidence_requirements": [{"requirement_id": "req-1"}]},
        "candidates": candidates,
    }

    result = rerank_codegen_tail(
        record, lambda _candidate: "", freeze_top=2, model_top=5)

    assert result["candidates"][:2] == candidates[:2]
    assert result["candidates"][2]["table_pos"] == 7


def test_schema_phrase_promotes_note_inside_model_window():
    candidates = [_candidate(i, score=10 - i) for i in range(8)]
    record = {
        "id": 1,
        "route": {
            "metric_norm": "lai va phi phai tra",
            "metric_variants": ["lai va phi phai tra"],
            "evidence_requirements": [],
        },
        "candidates": candidates,
    }
    blobs = {7: "Thuyet minh lai va phi phai tra cua cac khoan no"}

    result = rerank_codegen_tail(
        record,
        lambda candidate: blobs.get(candidate["table_pos"], "khong lien quan"),
        freeze_top=2,
        model_top=5,
    )

    assert result["candidates"][:2] == candidates[:2]
    assert result["candidates"][2]["table_pos"] == 7
