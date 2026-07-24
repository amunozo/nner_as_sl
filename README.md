# Nested Named Entity Recognition as Single-Pass Sequence Labeling

[![CI](https://github.com/amunozo/nner_as_sl/actions/workflows/ci.yml/badge.svg)](https://github.com/amunozo/nner_as_sl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/ACL-Anthology-red.svg)](https://aclanthology.org/2025.findings-emnlp.530/)

Research code for **[Nested Named Entity Recognition as Single-Pass Sequence
Labeling](https://aclanthology.org/2025.findings-emnlp.530/)**, by Alberto
Muñoz-Ortiz, David Vilares, Caio Corro, and Carlos Gómez-Rodríguez, published
in Findings of EMNLP 2025.

The project represents nested entities as constituency-like trees, linearizes
those trees with CoDeLin, and trains a multitask sequence labeler with MaChAmp.
It includes the ABS, REL, JUX, DYN, and 4EC encodings used in the experiments.
This is a research artifact, not a general-purpose NER package.

## Repository structure

- `src/data/`: NNER data parsing and tree/label conversion.
- `src/evaluation/`: exact-match metrics overall and by label, span length, and
  nesting depth.
- `src/machamp/`: generation of per-seed MaChAmp configurations.
- `scripts/train.py`: data encoding and multi-seed training.
- `scripts/evaluate.py`: prediction, decoding, timing, and metric aggregation.
- `scripts/entities_per_depth.py`: dataset statistics.
- `scripts/label_coverage.py`: encode/decode round-trip coverage.
- `parameter_configs/`: MaChAmp parameter template.
- `CoDeLin/` and `machamp/`: pinned Git submodules.
- `tests/`: fast conversion, metric, and command-construction tests.

## Installation

Clone the submodules and create an isolated Python environment:

```bash
git clone --recurse-submodules https://github.com/amunozo/nner_as_sl.git
cd nner_as_sl
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the repository was cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

The MaChAmp revision is the one originally used by this project. The original
CoDeLin object recorded by the repository is no longer served by its upstream
remote; the submodule is therefore pinned to an available upstream revision
that contains the nested-NER and 4EC fixes and exposes the same command-line
interface used here.

## Data format

Datasets are not redistributed. Place each dataset under `data/<dataset>/` with
`train.data`, `dev.data`, and `test.data`. Each example contains a tokenized
sentence followed by zero or more pipe-separated entities. End offsets are
inclusive.

```text
IL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production .
0,1 G#DNA|4,5 G#protein|8,8 G#protein
```

Nested spans must be properly nested. Crossing spans cannot be represented by
the constituency-tree conversion and are rejected with an explicit error.

## Training

The training command creates tree and label files when necessary, writes one
configuration per seed, and invokes the pinned MaChAmp entry point without a
shell:

```bash
python scripts/train.py \
  --dataset genia \
  --encoder bert-base-uncased \
  --encoding REL \
  --n-seeds 3 \
  --num-epochs 30 \
  --device 0 \
  --time
```

Completed seeds with a `model.pt` are skipped. If a seed directory contains
only incomplete output, MaChAmp is launched again in that directory and the
situation is reported explicitly. Use `--force-encode` to regenerate label
files after changing the source data.

All important roots can be overridden through `--data-dir`, `--logs-dir`,
`--template-dir`, `--machamp-dir`, and `--codelin-dir`. Run `--help` for the
complete interface.

## Evaluation

```bash
python scripts/evaluate.py \
  --dataset genia \
  --encoder bert-base-uncased \
  --encoding REL \
  --device 0
```

Use `--no-predict` to decode and evaluate an existing `output.labels` file, or
repeat `--seed` to select specific seeds. Per-seed `results.json` files and an
`avg_results.json` file are stored next to the models.

Metrics use exact matches over `(label, start, inclusive_end)` tuples:

- span-length groups use `end - start + 1`;
- label and length reports include prediction-only groups, so false positives
  are not hidden;
- files with different sentence counts fail instead of being silently
  truncated;
- depth recall groups correct predictions by the entity's gold depth, while
  depth precision groups them by predicted depth. Separate correct counts are
  reported because a correctly recovered entity can change depth when a
  surrounding entity is missed.

Dataset and encoding diagnostics are available through explicit CLIs:

```bash
python scripts/entities_per_depth.py genia ace2005
python scripts/label_coverage.py genia --encodings ABS REL DYN 4EC
```

## Development checks

```bash
python -m pip install -r requirements-dev.txt nltk
python -m ruff check .
python -m pytest
```

CI validates deterministic conversion and evaluation code on Python 3.10 and
3.12. It does not run transformer training, which requires the datasets,
downloaded model weights, and suitable compute.

## Citation

```bibtex
@inproceedings{munoz-ortiz-etal-2025-nested,
    title = {Nested Named Entity Recognition as Single-Pass Sequence Labeling},
    author = {Mu\~noz-Ortiz, Alberto and Vilares, David and Corro, Caio and G\'omez-Rodr\'iguez, Carlos},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
    year = {2025},
    address = {Suzhou, China},
    publisher = {Association for Computational Linguistics},
    doi = {10.18653/v1/2025.findings-emnlp.530},
    pages = {9993--10002},
}
```

## Contact

Alberto Muñoz-Ortiz: [alberto.munoz.ortiz@udc.es](mailto:alberto.munoz.ortiz@udc.es)

## Acknowledgments

This work received support from SCANNER-UDC (PID2020-113230RB-C21), Xunta de
Galicia (ED431C 2024/02), GAP (PID2022-139308OA-I00), PRE2021-097001, LATCHING
(PID2023-147129OB-C21), TSI-100925-2023-1, CITIC, the Galician Supercomputing
Center, and the projects supporting Caio Corro. See the paper for the complete
funding statement.
