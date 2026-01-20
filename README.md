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

This work was funded by SCANNER-UDC (PID2020-113230RB-C21) funded by MICIU/AEI/10.13039/501100011033; Xunta de Galicia (ED431C 2024/02); GAP (PID2022-139308OA-I00) funded by MICIU/AEI/10.13039/501100011033/ and by ERDF, EU; Grant PRE2021-097001 funded by MICIU/AEI/10.13039/501100011033 and by ESF+ (predoctoral training grant associated to project PID2020-113230RB-C21); LATCHING (PID2023-147129OB-C21) funded by MICIU/AEI/10.13039/501100011033 and ERDF; TSI-100925-2023-1 funded
by Ministry for Digital Transformation and Civil
Service and “NextGenerationEU” PRTR; 
and Centro de Investigación de Galicia ‘‘CITIC’’, funded by the Xunta de Galicia through the collaboration agreement between the Consellería de Cultura, Educación, Formación Profesional e Universidades and the Galician universities for the reinforcement of the research centres of the Galician University System (CIGUSA). 

This research project was made possible through the access granted by the Galician Supercomputing Center (CESGA) to its supercomputing infrastructure. The supercomputer FinisTerrae III and its permanent data storage system have been funded by the NextGeneration EU 2021 Recovery, Transformation and Resilience Plan, ICT2021-006904, and also from the Pluriregional Operational Programme of Spain 2014-2020 of the European Regional Development Fund (ERDF), ICTS-2019-02-CESGA-3, and from the State Programme for the Promotion of Scientific and Technical Research of Excellence of the State Plan for Scientific and Technical Research and Innovation 2013-2016 State subprogramme for scientific and technical infrastructures and equipment of ERDF, CESG15-DE-3114.

Caio Corro has received funding from the French Agence Nationale pour la Recherche under grant agreement InExtenso ANR-23-IAS1-0004 and SEMIAMOR ANR-23-CE23-0005.