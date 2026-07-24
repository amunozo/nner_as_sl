#!/usr/bin/env python3
"""Measure the maximum entity recall retained by each linearization."""

import argparse
import csv
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import (  # noqa: E402
    decode,
    encode,
    find_entities,
    to_parenthesized,
    trees_to_data,
)


def create_joint_file(dataset_dir, output_file):
    dataset_dir = Path(dataset_dir)
    examples = []
    for split in ("train", "dev", "test"):
        path = dataset_dir / f"{split}.data"
        if not path.is_file():
            raise FileNotFoundError(f"Required split not found: {path}")
        text = path.read_text(encoding="utf-8")
        examples.extend(block for block in text.strip().split("\n\n") if block.strip())
    Path(output_file).write_text("\n\n".join(examples) + "\n", encoding="utf-8")
    return str(output_file)


def max_possible_recall(filename, encoding, codelin_dir, verbose=False):
    """Encode and decode a dataset, then calculate exact entity retention."""
    with tempfile.TemporaryDirectory(prefix="nner-coverage-") as temp_directory:
        temp = Path(temp_directory)
        trees = Path(to_parenthesized(filename, temp / "gold.trees"))
        labels = Path(encode(encoding, trees, temp / "encoded.labels", codelin_dir=codelin_dir))
        decoded_trees = Path(
            decode(encoding, labels, temp / "decoded.trees", codelin_dir=codelin_dir)
        )
        decoded_data = Path(trees_to_data(decoded_trees, temp / "decoded.data"))
        gold_entities = find_entities(filename)
        predicted_entities = find_entities(decoded_data)
        if len(gold_entities) != len(predicted_entities):
            raise ValueError("Encoded round trip changed the number of sentences")

        n_correct = 0
        n_gold = 0
        for sentence_id, (gold, predicted) in enumerate(
            zip(gold_entities, predicted_entities),
            1,
        ):
            n_correct += len(gold & predicted)
            n_gold += len(gold)
            missed = gold - predicted
            if verbose and missed:
                print(f"Sentence {sentence_id}: missed {sorted(missed)}")
        return {
            "max_recall": 0.0 if n_gold == 0 else n_correct / n_gold,
            "correct_entities": n_correct,
            "gold_entities": n_gold,
            "predicted_entities": sum(len(pred) for pred in predicted_entities),
        }


def save_results_to_csv(results, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for result in results for key in result})
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="dataset directory names")
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=["ABS", "REL", "DYN", "4EC"],
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--codelin-dir", type=Path, default=PROJECT_ROOT / "CoDeLin")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "label_coverage.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    all_results = []
    with tempfile.TemporaryDirectory(prefix="nner-joint-") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        for dataset in args.datasets:
            try:
                joint_file = create_joint_file(
                    args.data_dir / dataset,
                    temporary_directory / f"{dataset}.data",
                )
                for encoding in args.encodings:
                    result = max_possible_recall(
                        joint_file,
                        encoding,
                        args.codelin_dir,
                        args.verbose,
                    )
                    result.update(
                        {
                            "dataset": dataset,
                            "encoding": encoding,
                            "max_recall_percentage": f"{result['max_recall']:.2%}",
                        }
                    )
                    all_results.append(result)
                    print(
                        f"{dataset}/{encoding}: {result['max_recall']:.2%} "
                        f"({result['correct_entities']}/{result['gold_entities']})"
                    )
            except (FileNotFoundError, ValueError) as error:
                parser.error(str(error))
    save_results_to_csv(all_results, args.output)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
