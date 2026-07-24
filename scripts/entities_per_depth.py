#!/usr/bin/env python3
"""Summarize entity types and nesting depths for NNER datasets."""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import find_entities  # noqa: E402
from src.evaluation.evaluator import calculate_nesting_depth  # noqa: E402


def entity_count(dataset_directory):
    dataset_directory = Path(dataset_directory)
    counts = {"all": 0}
    all_depths = set()
    entity_depths = []
    sentence_count = 0
    entity_types = Counter()
    for split in ("train", "dev", "test"):
        path = dataset_directory / f"{split}.data"
        if not path.is_file():
            raise FileNotFoundError(f"Required split not found: {path}")
        sentences = find_entities(path)
        sentence_count += len(sentences)
        counts["all"] += sum(len(sentence) for sentence in sentences)
        for sentence in sentences:
            depths = calculate_nesting_depth(sentence)
            entity_depths.extend(depths.values())
            for entity, depth in depths.items():
                all_depths.add(depth)
                counts[depth] = counts.get(depth, 0) + 1
                entity_types[entity[0]] += 1

    percentages = {
        depth: round(100 * counts[depth] / counts["all"], 2)
        for depth in all_depths
        if counts["all"]
    }
    return {
        "counts": counts,
        "percentages": percentages,
        "average_depth": (
            round(statistics.mean(entity_depths), 2) if entity_depths else 0
        ),
        "median_depth": (
            round(statistics.median(entity_depths), 2) if entity_depths else 0
        ),
        "sentence_count": sentence_count,
        "entity_types": dict(sorted(entity_types.items())),
        "all_depths": sorted(all_depths),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="dataset directory names")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, help="optional JSON output")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = {
            dataset: entity_count(args.data_dir / dataset)
            for dataset in args.datasets
        }
    except FileNotFoundError as error:
        parser.error(str(error))
    serialized = json.dumps(results, indent=2) + "\n"
    print(serialized, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
