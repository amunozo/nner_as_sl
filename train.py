"""
Train a pretrained multi-task learning model using MaChAmp. Datasets need to be already converted to
linearized labels.
"""

import argparse
import torch
import os
from src.machamp.configs import ConfigCreator
import os
from src.data.utils import to_parenthesized, remove_bos_eos, encode, remove_features
from tqdm import tqdm


machamp_training_script = 'machamp/train.py'

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="NNER dataset", required=True)
parser.add_argument('--encoder', help="Encoder model from HuggingFace.", required=True)
parser.add_argument('--encoding', help="Sequence labeling encoding", 
                    choices=['ABS', 'REL', 'JUX', 'DYN'], required=True)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--n_seeds',
                    help="Number of random initializations for the experiment", 
                    default=1)

args = parser.parse_args()

if args.device == None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

encoder_name = args.encoder.split('/')[-1]
data_dir = f'data/{args.dataset}'

# create .label data if it does not exist
if not os.path.exists(f'data/{args.dataset}/{args.encoding}/train.labels'):
    for split in tqdm(['test', 'dev', 'train']):
        to_parenthesized(
            f'{data_dir}/{split}.data', 
            f'{data_dir}/{split}.trees'
            )
        print(f'{split}.trees file created.')
        input_file = f"{data_dir}/{split}.trees"

        encoding_dir = f'{data_dir}/{args.encoding}'
        if not os.path.exists(encoding_dir):
            os.makedirs(encoding_dir)
        
        output_file = f"{encoding_dir}/{split}.labels"
        encode(args.encoding, input_file, output_file)
        remove_bos_eos(output_file)
        print(f'{split}.labels file created.')

for seed in range(int(args.n_seeds)):
    config_creator = ConfigCreator(args.dataset, args.encoder,args.encoding, 
                                   seed, template_dir='parameter_configs')
    model_dir = f'logs/machamp/{args.dataset}/{encoder_name}/{args.encoding}/seed_{seed}'
    dataset_config = config_creator.create_dataset_config()
    parameter_config = config_creator.create_parameters_config()

    os.system(f'python machamp/train.py --dataset_configs {dataset_config} \
            --device {args.device} --parameters_config {parameter_config} \
            --seed {seed} --model_dir {model_dir}')