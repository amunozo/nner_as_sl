datasets=("genia") # "ace2004" "ace2005" "nne")
encoders=("answerdotai/ModernBERT-base") # bert-base-uncased
n_seeds=1

for dataset in "${datasets[@]}"
do
    for encoder in "${encoders[@]}"
    do
        python train.py --dataset "$dataset" --encoder "$encoder" --encoding ABS --device 0 --n_seeds $n_seeds &
        #python train.py --dataset "$dataset" --encoder "$encoder" --encoding REL --device 1 --n_seeds $n_seeds &
        #wait 
        #python train.py --dataset "$dataset" --encoder "$encoder" --encoding JUX --device 0 --n_seeds $n_seeds &
        #python train.py --dataset "$dataset" --encoder "$encoder" --encoding DYN --device 1 --n_seeds $n_seeds &
        #wait
        python evaluate.py --dataset "$dataset" --encoder "$encoder" --encoding ABS --device 0 &
        #python evaluate.py --dataset "$dataset" --encoder "$encoder" --encoding REL --device 1 &
        #wait
        #python evaluate.py --dataset "$dataset" --encoder "$encoder" --encoding JUX --device 0 &
        #python evaluate.py --dataset "$dataset" --encoder "$encoder" --encoding DYN --device 1 &
        3wait
    done
done
