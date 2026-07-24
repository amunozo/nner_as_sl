"""Exact-match evaluation for flat and nested named entities."""

import subprocess
import sys
from pathlib import Path

from src.data.utils import find_entities


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_nesting_depth(entities):
    """Assign depth 1 to outer entities and increasing depth to nested ones."""
    sorted_entities = sorted(entities, key=lambda entity: (entity[1], -entity[2], entity[0]))
    depths = {}
    for entity in sorted_entities:
        parents = [
            other
            for other in sorted_entities
            if (other[1], other[2]) != (entity[1], entity[2])
            and other[1] <= entity[1]
            and entity[2] <= other[2]
        ]
        depths[entity] = 1 + max((depths.get(parent, 1) for parent in parents), default=0)
    return depths


def _metric_result(n_gold, n_pred, n_correct):
    precision = 0.0 if n_pred == 0 else n_correct / n_pred
    recall = 0.0 if n_gold == 0 else n_correct / n_gold
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "n_correct": n_correct,
        "fp": n_pred - n_correct,
        "fn": n_gold - n_correct,
    }


def _validate_alignment(gold_entities, predicted_entities):
    if len(gold_entities) != len(predicted_entities):
        raise ValueError(
            "Gold and prediction files contain different sentence counts: "
            f"{len(gold_entities)} != {len(predicted_entities)}"
        )


class Evaluator:
    def __init__(
        self,
        encoder=None,
        dataset=None,
        encoding=None,
        device=None,
        *,
        project_root=PROJECT_ROOT,
        data_dir="data",
        logs_dir="logs",
        machamp_dir="machamp",
    ):
        self.reset()
        self.encoder = encoder
        self.dataset = dataset
        self.encoding = encoding
        self.device = device
        self.project_root = Path(project_root).resolve()
        self.data_dir = self.project_root / data_dir
        self.logs_dir = self.project_root / logs_dir
        self.machamp_dir = self.project_root / machamp_dir
        self.encoder_name = Path(encoder).name if encoder else None
        self.model_dirs = None
        self.seeds = []
        if encoder and dataset and encoding:
            self.model_dirs = (
                self.logs_dir / "machamp" / dataset / self.encoder_name / encoding
            )
            if self.model_dirs.is_dir():
                self.seeds = sorted(
                    path.name.removeprefix("seed_")
                    for path in self.model_dirs.iterdir()
                    if path.is_dir() and path.name.startswith("seed_")
                )

    def reset(self):
        self.n_correct = 0
        self.n_gold = 0
        self.n_pred = 0
        self.decoder_timing = 0

    def __call__(self, gold, pred):
        self.n_correct += len(gold & pred)
        self.n_gold += len(gold)
        self.n_pred += len(pred)

    def precision(self):
        return 0.0 if self.n_pred == 0 else self.n_correct / self.n_pred

    def recall(self):
        return 0.0 if self.n_gold == 0 else self.n_correct / self.n_gold

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def write(self, ostream, dataset_name):
        ostream.write(
            f"Eval on {dataset_name}:\t"
            f"prec={100 * self.precision():.2f}\t"
            f"rec={100 * self.recall():.2f}\t"
            f"f1={100 * self.f1():.2f}\t"
            f"timing={self.decoder_timing}"
        )

    @staticmethod
    def _load_aligned(gold_data, predicted_data):
        gold_entities = find_entities(gold_data)
        predicted_entities = find_entities(predicted_data)
        _validate_alignment(gold_entities, predicted_entities)
        return gold_entities, predicted_entities

    def calculate_metrics(self, gold_data, predicted_data):
        gold_entities, predicted_entities = self._load_aligned(gold_data, predicted_data)
        n_gold = sum(len(sentence) for sentence in gold_entities)
        n_pred = sum(len(sentence) for sentence in predicted_entities)
        n_correct = sum(len(gold & pred) for gold, pred in zip(gold_entities, predicted_entities))
        return _metric_result(n_gold, n_pred, n_correct)

    def calculate_metrics_by_length(self, gold_data, predicted_data):
        gold_entities, predicted_entities = self._load_aligned(gold_data, predicted_data)
        lengths = sorted(
            {
                end - start + 1
                for sentences in (gold_entities, predicted_entities)
                for sentence in sentences
                for _label, start, end in sentence
            }
        )
        results = {}
        for length in lengths:
            gold_groups = [
                {entity for entity in sentence if entity[2] - entity[1] + 1 == length}
                for sentence in gold_entities
            ]
            pred_groups = [
                {entity for entity in sentence if entity[2] - entity[1] + 1 == length}
                for sentence in predicted_entities
            ]
            n_gold = sum(len(group) for group in gold_groups)
            n_pred = sum(len(group) for group in pred_groups)
            n_correct = sum(len(gold & pred) for gold, pred in zip(gold_groups, pred_groups))
            results[length] = _metric_result(n_gold, n_pred, n_correct)
        return results

    def calculate_metrics_by_depth(self, gold_data, predicted_data):
        """Evaluate recall by gold depth and precision by predicted depth.

        A correct entity can have different gold and predicted depths when a
        surrounding entity is missed. Separate correct counts make that case
        explicit instead of forcing inconsistent FP/FN values.
        """
        gold_entities, predicted_entities = self._load_aligned(gold_data, predicted_data)
        gold_depths = [calculate_nesting_depth(sentence) for sentence in gold_entities]
        pred_depths = [calculate_nesting_depth(sentence) for sentence in predicted_entities]
        depths = sorted(
            {
                depth
                for sentence_depths in (*gold_depths, *pred_depths)
                for depth in sentence_depths.values()
            }
        )
        results = {}
        for depth in depths:
            n_gold = n_pred = correct_gold_depth = correct_pred_depth = 0
            for gold, pred, gold_map, pred_map in zip(
                gold_entities,
                predicted_entities,
                gold_depths,
                pred_depths,
            ):
                gold_at_depth = {entity for entity in gold if gold_map[entity] == depth}
                pred_at_depth = {entity for entity in pred if pred_map[entity] == depth}
                n_gold += len(gold_at_depth)
                n_pred += len(pred_at_depth)
                correct_gold_depth += len(gold_at_depth & pred)
                correct_pred_depth += len(pred_at_depth & gold)

            precision = 0.0 if n_pred == 0 else correct_pred_depth / n_pred
            recall = 0.0 if n_gold == 0 else correct_gold_depth / n_gold
            f1 = (
                0.0
                if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
            results[depth] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_pred": n_pred,
                "n_gold": n_gold,
                "n_correct_pred_depth": correct_pred_depth,
                "n_correct_gold_depth": correct_gold_depth,
                "fp": n_pred - correct_pred_depth,
                "fn": n_gold - correct_gold_depth,
            }
        return results

    @staticmethod
    def calculate_nesting_depth(entities):
        return calculate_nesting_depth(entities)

    def calculate_metrics_by_label(self, gold_data, predicted_data):
        gold_entities, predicted_entities = self._load_aligned(gold_data, predicted_data)
        labels = sorted(
            {
                entity[0]
                for sentences in (gold_entities, predicted_entities)
                for sentence in sentences
                for entity in sentence
            }
        )
        results = {}
        for label in labels:
            gold_groups = [
                {entity for entity in sentence if entity[0] == label}
                for sentence in gold_entities
            ]
            pred_groups = [
                {entity for entity in sentence if entity[0] == label}
                for sentence in predicted_entities
            ]
            n_gold = sum(len(group) for group in gold_groups)
            n_pred = sum(len(group) for group in pred_groups)
            n_correct = sum(len(gold & pred) for gold, pred in zip(gold_groups, pred_groups))
            results[label] = _metric_result(n_gold, n_pred, n_correct)
        return results

    def predict(self, seed):
        if not all((self.encoder, self.dataset, self.encoding, self.model_dirs)):
            raise ValueError("Model parameters are not configured for prediction")
        script = self.machamp_dir / "predict.py"
        if not script.is_file():
            raise FileNotFoundError(
                f"MaChAmp entry point not found at {script}. "
                "Run 'git submodule update --init --recursive'."
            )
        model_dir = self.model_dirs / f"seed_{seed}"
        model_file = model_dir / "model.pt"
        input_file = self.data_dir / self.dataset / self.encoding / "test.labels"
        output_file = model_dir / "output.labels"
        for required in (model_file, input_file):
            if not required.is_file():
                raise FileNotFoundError(f"Required evaluation input not found: {required}")
        subprocess.run(
            [
                sys.executable,
                str(script),
                str(model_file),
                str(input_file),
                str(output_file),
                "--device",
                str(self.device),
                "--dataset",
                self.dataset,
            ],
            check=True,
            cwd=self.project_root,
        )
        return str(output_file)
