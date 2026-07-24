import pytest

from src.evaluation.evaluator import Evaluator, calculate_nesting_depth
from src.evaluation.utils import average_dictionary


def write_data(path, examples):
    blocks = [f"{text}\n{entities}" if entities else text for text, entities in examples]
    path.write_text("\n\n".join(blocks) + "\n")


def test_entity_length_is_inclusive_and_predicted_only_groups_are_reported(tmp_path):
    gold = tmp_path / "gold.data"
    pred = tmp_path / "pred.data"
    write_data(gold, [("A B C", "0,2 OUTER")])
    write_data(pred, [("A B C", "0,2 OUTER|1,1 EXTRA")])
    evaluator = Evaluator()

    by_length = evaluator.calculate_metrics_by_length(gold, pred)
    by_label = evaluator.calculate_metrics_by_label(gold, pred)

    assert set(by_length) == {1, 3}
    assert by_length[3]["f1"] == 1.0
    assert by_length[1]["fp"] == 1
    assert by_label["EXTRA"]["n_gold"] == 0
    assert by_label["EXTRA"]["n_pred"] == 1


def test_sentence_count_mismatch_is_rejected(tmp_path):
    gold = tmp_path / "gold.data"
    pred = tmp_path / "pred.data"
    write_data(gold, [("A", ""), ("B", "")])
    write_data(pred, [("A", "")])

    with pytest.raises(ValueError, match="sentence counts"):
        Evaluator().calculate_metrics(gold, pred)


def test_depth_metrics_distinguish_gold_and_predicted_depth(tmp_path):
    gold = tmp_path / "gold.data"
    pred = tmp_path / "pred.data"
    write_data(gold, [("A B C", "0,2 OUTER|1,1 INNER")])
    write_data(pred, [("A B C", "1,1 INNER")])

    results = Evaluator().calculate_metrics_by_depth(gold, pred)

    assert results[1]["n_correct_pred_depth"] == 1
    assert results[1]["n_correct_gold_depth"] == 0
    assert results[2]["n_correct_gold_depth"] == 1
    assert results[2]["n_correct_pred_depth"] == 0


def test_same_span_different_labels_do_not_create_artificial_depth():
    entities = {("A", 0, 1), ("B", 0, 1)}

    assert calculate_nesting_depth(entities) == {("A", 0, 1): 1, ("B", 0, 1): 1}


def test_missing_subgroup_contributes_zero_when_averaging():
    runs = [
        {"overall": {"f1": 1.0}, "by_label": {"X": {"f1": 1.0}}},
        {"overall": {"f1": 0.0}, "by_label": {}},
    ]

    result = average_dictionary(runs)

    assert result["overall"]["f1"] == {"mean": 0.5, "std": 0.5}
    assert result["by_label"]["X"]["f1"] == {"mean": 0.5, "std": 0.5}
