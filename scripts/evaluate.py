from src.evaluation.evaluator import Evaluator
from src.evaluation.utils import average_dictionary
import argparse
from src.data.utils import trees_to_data, decode, add_bos_eos
import json
import os
import time
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Machamp model (caio evaluator only)")
    parser.add_argument('--encoder', type=str, required=True, help="Name of the encoder used from HF")
    parser.add_argument('--encoding', type=str, required=True, help="Constituency encoding")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument('--device', type=str, default='0', help="Device to run the evaluation on")
    parser.add_argument('--predict', action='store_true', help="Whether to predict (default: False)")
    parser.add_argument('--no-predict', dest='predict', action='store_false', help="Whether to use existing predictions")
    parser.add_argument('--by-label', action='store_true', help="Evaluate metrics by label")
    parser.add_argument('--by-depth', action='store_true', help="Evaluate metrics by depth")
    parser.add_argument('--by-length', action='store_true', help="Evaluate metrics by entity length")
    parser.set_defaults(predict=True, by_depth=True, by_label=True, by_length=True)

    args = parser.parse_args()

    evaluator = Evaluator(args.encoder, args.dataset, args.encoding, args.device)
    all_results = []
    all_times = []

    # Base directory for model evaluations
    model_dirs = f'logs/machamp/{evaluator.dataset}/{evaluator.encoder}/{evaluator.encoding}/'
    gold_data = f'data/{evaluator.dataset}/test.data'

    for seed in evaluator.seeds:
        print(f"Evaluating seed: {seed}")
        time_dict = {}
        total_start = time.time()
        predict_end = predict_start = total_start # Initialize in case --no-predict and file missing

        # Predict
        if args.predict:
            predict_start = time.time()
            predicted_labels = evaluator.predict(seed)
            predict_end = time.time()
            predicted_labels = add_bos_eos(predicted_labels)
            time_dict['predict'] = predict_end - predict_start
        else:
            predicted_labels = f'{model_dirs}seed_{seed}/output.labels'
            if not os.path.exists(predicted_labels):
                print(f"Prediction file not found for seed {seed}. Skipping...")
                continue
            # If not predicting, set predict time to 0 or handle appropriately
            time_dict['predict'] = 0.0 # Or load from a previous run if available

        # Decode
        decode_start = time.time()
        pred_trees = decode(args.encoding, predicted_labels, predicted_labels.replace('labels', 'trees'))
        pred_data = trees_to_data(pred_trees, pred_trees.replace('trees', 'data'))
        decode_end = time.time()
        time_dict['decode'] = decode_end - decode_start

        # Count sentences and tokens
        num_sentences = 0
        num_tokens = 0
        try:
            with open(pred_data, 'r') as f:
                for line in f:
                    num_sentences += 1
                    num_tokens += len(line.strip().split())
        except FileNotFoundError:
             print(f"Decoded data file not found: {pred_data}. Skipping timing calculation for this seed.")
             # Handle error appropriately, maybe skip this seed or set counts to 0
             num_sentences = 0
             num_tokens = 0
             # Decide how to handle timing if counts are zero
             time_dict['total'] = time.time() - total_start # Still record total time spent so far
             time_dict['num_sentences'] = 0
             time_dict['num_tokens'] = 0
             # Add basic timing to all_times before continuing
             all_times.append({
                 'predict': time_dict.get('predict', 0.0),
                 'decode': time_dict.get('decode', 0.0),
                 'total': time_dict.get('total', 0.0),
                 'num_sentences': 0,
                 'num_tokens': 0,
             })
             continue # Skip metric calculation and saving for this seed


        time_dict['num_sentences'] = num_sentences
        time_dict['num_tokens'] = num_tokens
        time_dict['total'] = time.time() - total_start

        seed_results = {}
        # Calculate metrics
        seed_results["overall"] = evaluator.calculate_metrics(gold_data, pred_data)
        if args.by_depth:
            seed_results["by_depth"] = evaluator.calculate_metrics_by_depth(gold_data, pred_data)
        if args.by_length:
            seed_results["by_length"] = evaluator.calculate_metrics_by_length(gold_data, pred_data)
        if args.by_label:
            seed_results["by_label"] = evaluator.calculate_metrics_by_label(gold_data, pred_data)

        # Add simplified timing info to results and track for averaging
        simplified_timing = {
            'predict': time_dict.get('predict', 0.0), # Use .get for safety
            'decode': time_dict['decode'],
            'total': time_dict['total'],
            'num_sentences': time_dict['num_sentences'],
            'num_tokens': time_dict['num_tokens'],
        }
        seed_results['timing'] = simplified_timing
        all_results.append(seed_results)
        all_times.append(simplified_timing) # Append the simplified dict

        seed_dir = os.path.join(model_dirs, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, "results.json"), "w") as f:
            json.dump(seed_results, f, indent=2)

    avg_results = average_dictionary(all_results)

    # Compute global timing (simplified: only total sums)
    global_time = {}
    if all_times:
        global_time['total_predict'] = sum(t.get('predict', 0.0) for t in all_times) # Use .get for safety
        global_time['total_decode'] = sum(t['decode'] for t in all_times)
        global_time['total_time'] = sum(t['total'] for t in all_times)
        global_time['total_sentences'] = sum(t['num_sentences'] for t in all_times)
        global_time['total_tokens'] = sum(t['num_tokens'] for t in all_times)
        # Optionally calculate overall rates here if needed:
        if global_time['total_time'] > 0:
             global_time['avg_sentences_per_second'] = global_time['total_sentences'] / global_time['total_time']
             global_time['avg_tokens_per_second'] = global_time['total_tokens'] / global_time['total_time']
        else:
             global_time['avg_sentences_per_second'] = 0.0
             global_time['avg_tokens_per_second'] = 0.0
        if global_time['total_sentences'] > 0:
             global_time['avg_time_per_sentence'] = global_time['total_time'] / global_time['total_sentences']
        else:
             global_time['avg_time_per_sentence'] = float('inf')

    # Removed micro and macro average sections for rates

    avg_results['timing'] = global_time  # Ensure timing is in average results

    # Save averaged results in the base model directory
    os.makedirs(model_dirs, exist_ok=True)
    with open(os.path.join(model_dirs, "avg_results.json"), "w") as f:
        json.dump(avg_results, f, indent=2)

    print("Done! Per-seed results and averaged results saved.")
