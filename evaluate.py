from src.evaluation.evaluator import Evaluator, average_dictionary
import argparse
from src.data.utils import trees_to_data, decode, add_bos_eos
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Machamp model")
    parser.add_argument('--encoder' , type=str, required=True, help="Name of the encoder used from HF")
    parser.add_argument('--encoding', type=str, required=True, help='Constituency encoding')
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset we want to evaluate the \
                        model on")
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'f1_micro', 'f1_macro'],
                    help='Metric using during training to select the best model')
    parser.add_argument('--device', type=str, default='0', help="Device to run the evaluation on")
    parser.add_argument('--evaluator', default='caio', choices=['caio', 'nervaluate'])


    args = parser.parse_args()


    machamp_eval = Evaluator(args.encoder, args.encoding, args.dataset, args.device)
    all_results = []
    all_results_per_tag = []
    model_dirs = f'logs/machamp/{machamp_eval.dataset}/{machamp_eval.encoder}/{machamp_eval.encoding}/{machamp_eval.metric}' 
    gold_data = f'clean_data/{machamp_eval.dataset}/test.data'

    for seed in machamp_eval.seeds:
        # predict label
        predicted_labels = machamp_eval.predict(seed)
        predicted_labels = add_bos_eos(predicted_labels)

        # labels to trees
        pred_trees = decode(machamp_eval.encoding, predicted_labels, predicted_labels.replace('labels', 'trees'))
        # trees to data
        pred_data = trees_to_data(pred_trees, pred_trees.replace('trees', 'data'))
        
        if args.evaluator == 'nervaluate':
            results, results_per_tag = machamp_eval.calculate_metrics(seed, 'nervaluate')
            all_results_per_tag.append(results_per_tag)
        
        elif args.evaluator == 'caio':
            results = machamp_eval.calculate_metrics(seed, 'caio')

        all_results.append(results)


    # average accross categories
    if args.evaluator == 'nervaluate': 
        average_results = average_dictionary(all_results)
        with open(f'{model_dirs}/results.json', 'w') as f:
                json.dump(average_results, f)

        average_results_per_tag = average_dictionary(all_results_per_tag)
        with open(f'{model_dirs}/results_per_tag.json', 'w') as f:
                json.dump(average_results_per_tag, f)
    
    elif args.evaluator == 'caio':
        print(all_results)
        with open(f'{model_dirs}/results_caio.json', 'w') as f:
                json.dump(all_results[0], f)