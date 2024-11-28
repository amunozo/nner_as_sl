from src.evaluation.evaluator import Evaluator
from src.evaluation.utils import average_dictionary
import argparse
from src.data.utils import trees_to_data, decode, add_bos_eos
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Machamp model")
    parser.add_argument('--encoder' , type=str, required=True, help="Name of the encoder used from HF")
    parser.add_argument('--encoding', type=str, required=True, help='Constituency encoding')
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset we want to evaluate the \
                        model on")
    parser.add_argument('--device', type=str, default='0', help="Device to run the evaluation on")
    parser.add_argument('--evaluator', default='caio', choices=['caio', 'nervaluate'])


    args = parser.parse_args()


    evaluator = Evaluator(args.encoder, args.dataset, args.encoding, args.device)
    all_results = []
    all_results_per_tag = []
    model_dirs = f'logs/machamp/{evaluator.dataset}/{evaluator.encoder}/{evaluator.encoding}/' 
    gold_data = f'data/{evaluator.dataset}/test.data'

    for seed in evaluator.seeds:
        # predict label
        predicted_labels = evaluator.predict(seed)
        predicted_labels = add_bos_eos(predicted_labels)

        # labels to trees
        pred_trees = decode(evaluator.encoding, predicted_labels, predicted_labels.replace('labels', 'trees'))
        # trees to data
        pred_data = trees_to_data(pred_trees, pred_trees.replace('trees', 'data'))       
        results = evaluator.calculate_metrics(gold_data, pred_data)
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