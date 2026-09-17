from scripts.adjudicate_v265_public_score import BASELINE, adjudicate


def scores(**changes):
    result = dict(BASELINE)
    result.update(changes)
    return result


def test_promotes_q24_when_answer_gains_one_and_tables_hold():
    report = adjudicate(scores(answer=361 / 506, execution=361 / 506))
    assert report["q24_inference"] == "new_answer_public_correct"
    assert report["verdict"] == "promote_candidate"


def test_splits_q24_when_answer_wins_but_table_ablation_regresses():
    report = adjudicate(
        scores(answer=361 / 506, execution=361 / 506, tables_f2=0.6090)
    )
    assert report["verdict"] == "split_q24_from_retrieval_ablation"


def test_rejects_when_answer_loses_one():
    report = adjudicate(scores(answer=359 / 506, execution=359 / 506))
    assert report["q24_inference"] == "new_answer_public_wrong"
    assert report["verdict"] == "reject_rollback_v217"


def test_keeps_v217_on_exact_tie():
    report = adjudicate(scores())
    assert report["verdict"] == "public_neutral_keep_v217_selected"
