#!/usr/bin/env python3
"""Prepare encoded data and train MaChAmp models across random seeds."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import encode, remove_bos_eos, to_parenthesized  # noqa: E402
from src.machamp.configs import ConfigCreator  # noqa: E402


ENCODINGS = ("ABS", "REL", "JUX", "DYN", "4EC")


def prepare_labels(dataset_dir, encoding, codelin_dir, force=False):
    """Create CoDeLin label files for each available data split."""
    encoded_dir = dataset_dir / encoding
    encoded_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        data_file = dataset_dir / f"{split}.data"
        if not data_file.is_file():
            raise FileNotFoundError(f"Required data split not found: {data_file}")
        tree_file = dataset_dir / f"{split}.trees"
        label_file = encoded_dir / f"{split}.labels"
        if label_file.is_file() and not force:
            continue
        to_parenthesized(data_file, tree_file)
        encode(encoding, tree_file, label_file, codelin_dir=codelin_dir)
        remove_bos_eos(label_file)
        print(f"Prepared {label_file}")


def training_command(
    machamp_script,
    dataset_config,
    parameter_config,
    device,
    seed,
    model_dir,
):
    return [
        sys.executable,
        str(machamp_script),
        "--dataset_configs",
        str(dataset_config),
        "--device",
        str(device),
        "--parameters_config",
        str(parameter_config),
        "--seed",
        str(seed),
        "--model_dir",
        str(model_dir),
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train nested NER sequence-labeling models with MaChAmp."
    )
    parser.add_argument("--dataset", required=True, help="dataset directory name")
    parser.add_argument("--encoder", required=True, help="Hugging Face encoder ID")
    parser.add_argument("--encoding", choices=ENCODINGS, required=True)
    parser.add_argument("--device", default="0", help="MaChAmp device value")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--time", action="store_true", help="report time per seed")
    parser.add_argument("--force-encode", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=PROJECT_ROOT / "parameter_configs",
    )
    parser.add_argument("--machamp-dir", type=Path, default=PROJECT_ROOT / "machamp")
    parser.add_argument("--codelin-dir", type=Path, default=PROJECT_ROOT / "CoDeLin")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.n_seeds <= 0:
        parser.error("--n-seeds must be positive")

    machamp_script = args.machamp_dir / "train.py"
    if not machamp_script.is_file():
        parser.error(
            f"MaChAmp entry point not found at {machamp_script}. "
            "Run 'git submodule update --init --recursive'."
        )
    try:
        prepare_labels(
            args.data_dir / args.dataset,
            args.encoding,
            args.codelin_dir,
            args.force_encode,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    for seed in range(args.n_seeds):
        creator = ConfigCreator(
            args.dataset,
            args.encoder,
            args.encoding,
            args.num_epochs,
            seed,
            template_dir=args.template_dir,
            data_dir=args.data_dir,
            logs_dir=args.logs_dir,
        )
        model_file = creator.model_dir / "model.pt"
        if model_file.is_file():
            print(f"Seed {seed} already has a completed model; skipping")
            continue
        if creator.model_dir.exists() and any(creator.model_dir.iterdir()):
            print(f"Seed {seed} has incomplete output; starting MaChAmp in the same directory")

        dataset_config = creator.create_dataset_config()
        parameter_config = creator.create_parameters_config()
        command = training_command(
            machamp_script,
            dataset_config,
            parameter_config,
            args.device,
            seed,
            creator.model_dir,
        )
        start = time.perf_counter()
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        if args.time:
            elapsed = time.perf_counter() - start
            print(f"Training seed {seed} took {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
