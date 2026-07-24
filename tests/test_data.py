import json

import pytest
from nltk.tree import Tree

from src.data.utils import (
    _codelin_command,
    add_bos_eos,
    build_tree,
    data_to_jsonlines,
    extract_entities_from_tree,
    find_entities,
    nner_to_tree,
)


def test_single_entity_is_not_dropped():
    tree = nner_to_tree("New York shines", "0,1 GPE")

    assert tree == "(ROOT (GPE New York) shines)"


def test_nested_tree_round_trip():
    serialized = build_tree(
        "New York University",
        [(0, 2, "ORG"), (0, 1, "GPE")],
    )

    text, entities = extract_entities_from_tree(Tree.fromstring(serialized))

    assert text == "New York University"
    assert set(entities) == {(0, 2, "ORG"), (0, 1, "GPE")}


def test_crossing_entities_fail_loudly():
    with pytest.raises(ValueError, match="Crossing"):
        build_tree("a b c d", [(0, 2, "A"), (1, 3, "B")])


def test_find_entities_does_not_add_trailing_empty_sentence(tmp_path):
    path = tmp_path / "sample.data"
    path.write_text("A B\n0,0 X\n\nC\n\n", encoding="utf-8")

    result = find_entities(path)

    assert result == [{("X", 0, 0)}, set()]


def test_jsonlines_text_uses_inclusive_end_offset(tmp_path):
    source = tmp_path / "sample.data"
    target = tmp_path / "sample.jsonl"
    source.write_text("New York City\n0,1 GPE\n", encoding="utf-8")

    data_to_jsonlines(source, target)
    record = json.loads(target.read_text(encoding="utf-8"))

    assert record["entity_mentions"][0]["text"] == "New York"
    assert record["entity_mentions"][0]["end"] == 1


def test_boundary_markers_are_idempotent(tmp_path):
    labels = tmp_path / "output.labels"
    labels.write_text("A\tX\nB\tY\n", encoding="utf-8")

    add_bos_eos(labels)
    once = labels.read_text(encoding="utf-8")
    add_bos_eos(labels)

    assert labels.read_text(encoding="utf-8") == once
    assert once.count("-BOS-") == 2
    assert once.count("-EOS-") == 2


def test_codelin_command_uses_current_cli_flag(tmp_path):
    (tmp_path / "main.py").write_text("", encoding="utf-8")

    command = _codelin_command(
        "ENC",
        "REL",
        "input.trees",
        "output.labels",
        True,
        tmp_path,
    )

    assert "--ignore_postags" in command
    assert "--ignore_postag" not in command
    assert command[command.index("--n_label_cols") + 1] == "3"
