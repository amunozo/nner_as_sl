# Nested Named Entity Recognition as Single-Pass Sequence Labeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/ACL-Anthology-red.svg)](https://aclanthology.org/2025.findings-emnlp.530/)

This repository contains the official implementation for the paper:
**[Nested Named Entity Recognition as Single-Pass Sequence Labeling](https://aclanthology.org/2025.findings-emnlp.530/)**
*Alberto Muñoz-Ortiz, David Vilares, Caio Corro, and Carlos Gómez-Rodríguez*
Presented at **Findings of EMNLP 2025** in Suzhou, China.

---

## Overview

This project provides tools to linearize nested entity structures into sequence labeling formats and train models using various encoders (like BERT, RoBERTa, etc.). It supports multiple encoding strategies including REL, ABS, JUX, DYN, and 4EC.

## Project Structure

The repository is organized as follows:

- `src/`: Core logic, including data utilities and MaChAmp configurations.
- `scripts/`: Entry-point scripts for training, evaluation, and data analysis.
- `parameter_configs/`: Configuration templates for model training.
- `data/`: Directory for storing datasets (should contain `train.data`, `dev.data`, and `test.data`).
- `logs/`: Directory where training outputs and results are stored.

## Installation

1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/amunozo/nner_as_sl.git
   cd nner_as_sl
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train the NNER models:
```bash
python scripts/train.py \
  --dataset genia \
  --encoder bert-base-uncased \
  --encoding REL \
  --n_seeds 1
```

### Evaluation
To evaluate the trained models:
```bash
python scripts/evaluate.py \
  --dataset genia \
  --encoder bert-base-uncased \
  --encoding REL \
  --device 0
```

## Data Format

Input data should have a line with the tokens followed by a line with the annotated entities. Entities are defined by a triple `start, end, type` separated by `|`.

Example:
```text
IL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase .
0,1 G#DNA|4,5 G#protein|8,8 G#protein|14,14 G#protein
```

## Contact

For any questions or issues, please contact the author:
**Alberto Muñoz-Ortiz** - [alberto.munoz.ortiz@udc.es](mailto:alberto.munoz.ortiz@udc.es)

## Citation

If you use this code or our findings in your research, please cite:

```bibtex
@inproceedings{munoz-ortiz-etal-2025-nested,
    title = "Nested Named Entity Recognition as Single-Pass Sequence Labeling",
    author = "Mu{\~n}oz-Ortiz, Alberto and
      Vilares, David and
      Corro, Caio and
      G{\\'o}mez-Rodr{\\'i}guez, Carlos",
    editor = "Christodoulopoulos, Christos and
      Chakraborty, Tanmoy and
      Rose, Carolyn and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.530/",
    doi = "10.18653/v1/2025.findings-emnlp.530/",
    pages = "9993--10002",
}
```

## Acknowledgments

This work is based on the MaChAmp toolkit and uses various transformer models from HuggingFace.
