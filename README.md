# NNER: Neural Nested Named Entity Recognition

Code for training and evaluating nested NER models using transformers-based architectures through MaChAmp.

## Setup

1. Clone repository with submodule:
```bash
git clone --recurse-submodules https://github.com/your-username/nner_as_sl.git
cd nner_as_sl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format
Input data should have a line with the tokens and the next one with the annotated entities. The tokens are separated by spaces and the entities are separated by `|`. Entities are defined by a triple `start, end, type`.

```text
IL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase .
0,1 G#DNA|4,5 G#protein|8,8 G#protein|14,14 G#protein

Activation of the CD28 surface receptor provides a major costimulatory signal for T cell activation resulting in enhanced production of interleukin-2 ( IL-2 ) and cell proliferation .
3,3 G#protein|3,5 G#protein|20,20 G#protein|22,22 G#protein
```
The datasets must be stored in the `data` directory, and the dataset split must be named as `train.data`, `dev.data`, and `test.data`.

## Usage
### Training
```bash
python train_machamp.py \
  --dataset DATASET \
  --encoder bert-base-uncased \
  --encoding ENCODING \
  --device 0
```

### Evaluation 
```bash
python evaluate_machamp.py \
  --dataset DATASET \
  --encoder bert-base-uncased \
  --encoding ENCODING \
  --device 0
```
