import json
import sys

from scripts.evaluate import count_data, timing_summary
from scripts.train import training_command
from src.machamp.configs import ConfigCreator


def test_config_creator_uses_encoder_basename_and_explicit_roots(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "bert.json").write_text(
        json.dumps({"training": {"num_epochs": 0}}),
        encoding="utf-8",
    )
    creator = ConfigCreator(
        "genia",
        "org/model-name",
        "REL",
        5,
        2,
        template_dir=templates,
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
    )

    params = json.loads(open(creator.create_parameters_config(), encoding="utf-8").read())
    dataset = json.loads(open(creator.create_dataset_config(), encoding="utf-8").read())

    assert creator.encoder_name == "model-name"
    assert params["transformer_model"] == "org/model-name"
    assert params["random_seed"] == 2
    assert params["training"]["num_epochs"] == 5
    assert dataset["genia"]["train_data_path"].endswith("genia/REL/train.labels")


def test_training_command_does_not_use_a_shell():
    command = training_command("train.py", "data.json", "params.json", "0", 3, "model")

    assert command[0] == sys.executable
    assert command[-2:] == ["--model_dir", "model"]


def test_count_data_counts_examples_not_annotation_lines(tmp_path):
    path = tmp_path / "sample.data"
    path.write_text("A B\n0,0 X\n\nC\n\n", encoding="utf-8")

    assert count_data(path) == (2, 3)


def test_empty_timing_summary_is_json_safe():
    assert timing_summary([]) == {}
    summary = timing_summary(
        [{"predict": 0.0, "decode": 0.0, "total": 0.0, "num_sentences": 0, "num_tokens": 0}]
    )
    assert summary["avg_time_per_sentence"] is None
