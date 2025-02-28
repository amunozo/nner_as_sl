datasets=("genia" "ace2004" "ace2005" "nne")
encoders=("bert-base-uncased" "answerdotai/ModernBERT-base")
n_seeds=5

for dataset in "${datasets[@]}"
do
    python train.py --dataset "$dataset" --encoder bert-base-uncased --encoding ABS --device 0 --n_seeds $n_seeds &
    python train.py --dataset "$dataset" --encoder bert-base-uncased --encoding REL --device 1 --n_seeds $n_seeds &
    wait 
    python train.py --dataset "$dataset" --encoder bert-base-uncased --encoding JUX --device 0 --n_seeds $n_seeds &
    python train.py --dataset "$dataset" --encoder bert-base-uncased --encoding DYN --device 1 --n_seeds $n_seeds &
    wait
    python evaluate.py --dataset "$dataset" --encoder bert-base-uncased --encoding ABS --device 0 &
    python evaluate.py --dataset "$dataset" --encoder bert-base-uncased --encoding REL --device 1 &
    wait
    python evaluate.py --dataset "$dataset" --encoder bert-base-uncased --encoding JUX --device 0 &
    python evaluate.py --dataset "$dataset" --encoder bert-base-uncased --encoding DYN --device 1 &
    wait
done