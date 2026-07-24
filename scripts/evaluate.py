#!/usr/bin/env python3
"""Predict, decode, and evaluate nested NER models across seeds."""

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import (  # noqa: E402
    add_bos_eos,
    decode,
    iter_data_examples,
    trees_to_data,
)
from src.evaluation.evaluator import Evaluator  # noqa: E402
from src.evaluation.utils import average_dictionary  # noqa: E402


def count_data(path):
    examples = list(iter_data_examples(Path(path).read_text(encoding="utf-8")))
    return len(examples), sum(len(tokens) for tokens, _entities in examples)


def timing_summary(timings):
    if not timings:
        return {}
    total_time = sum(item["total"] for item in timings)
    total_sentences = sum(item["num_sentences"] for item in timings)
    total_tokens = sum(item["num_tokens"] for item in timings)
    return {
        "total_predict": sum(item["predict"] for item in timings),
        "total_decode": sum(item["decode"] for item in timings),
        "total_time": total_time,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "avg_sentences_per_second": (
            total_sentences / total_time if total_time else 0.0
        ),
        "avg_tokens_per_second": total_tokens / total_time if total_time else 0.0,
        "avg_time_per_sentence": (
            total_time / total_sentences if total_sentences else None
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MaChAmp nested NER models."
    )
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--encoding", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--predict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run MaChAmp prediction before decoding",
    )
    parser.add_argument(
        "--by-label", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--by-depth", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--by-length", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", action="append", help="evaluate only this seed")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--machamp-dir", type=Path, default=PROJECT_ROOT / "machamp")
    parser.add_argument("--codelin-dir", type=Path, default=PROJECT_ROOT / "CoDeLin")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluator = Evaluator(
        args.encoder,
        args.dataset,
        args.encoding,
        args.device,
        project_root=PROJECT_ROOT,
        data_dir=args.data_dir,
        logs_dir=args.logs_dir,
        machamp_dir=args.machamp_dir,
    )
    seeds = args.seed or evaluator.seeds
    if not seeds:
        parser.error(f"No seed directories found under {evaluator.model_dirs}")

    gold_data = args.data_dir / args.dataset / "test.data"
    if not gold_data.is_file():
        parser.error(f"Gold data not found: {gold_data}")

    all_results = []
    all_times = []
    for seed in seeds:
        print(f"Evaluating seed {seed}")
        total_start = time.perf_counter()
        if args.predict:
            predict_start = time.perf_counter()
            predicted_labels = Path(evaluator.predict(seed))
            predict_time = time.perf_counter() - predict_start
        else:
            predicted_labels = evaluator.model_dirs / f"seed_{seed}" / "output.labels"
            if not predicted_labels.is_file():
                parser.error(f"Prediction file not found: {predicted_labels}")
            predict_time = 0.0

        add_bos_eos(predicted_labels)
        decode_start = time.perf_counter()
        predicted_trees = predicted_labels.with_suffix(".trees")
        predicted_data = predicted_labels.with_suffix(".data")
        decode(
            args.encoding,
            predicted_labels,
            predicted_trees,
            codelin_dir=args.codelin_dir,
        )
        trees_to_data(predicted_trees, predicted_data)
        decode_time = time.perf_counter() - decode_start
        num_sentences, num_tokens = count_data(predicted_data)
        timing = {
            "predict": predict_time,
            "decode": decode_time,
            "total": time.perf_counter() - total_start,
            "num_sentences": num_sentences,
            "num_tokens": num_tokens,
        }

        results = {"overall": evaluator.calculate_metrics(gold_data, predicted_data)}
        if args.by_depth:
            results["by_depth"] = evaluator.calculate_metrics_by_depth(
                gold_data, predicted_data
            )
        if args.by_length:
            results["by_length"] = evaluator.calculate_metrics_by_length(
                gold_data, predicted_data
            )
        if args.by_label:
            results["by_label"] = evaluator.calculate_metrics_by_label(
                gold_data, predicted_data
            )
        results["timing"] = timing
        all_results.append(results)
        all_times.append(timing)

        seed_dir = evaluator.model_dirs / f"seed_{seed}"
        (seed_dir / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    averaged = average_dictionary(all_results)
    averaged["timing"] = timing_summary(all_times)
    evaluator.model_dirs.mkdir(parents=True, exist_ok=True)
    (evaluator.model_dirs / "avg_results.json").write_text(
        json.dumps(averaged, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved per-seed and averaged results under {evaluator.model_dirs}")


if __name__ == "__main__":
    main()
