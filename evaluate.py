from src.evaluation.evaluator import Evaluator
from src.evaluation.utils import average_dictionary
import argparse
from src.data.utils import trees_to_data, decode, add_bos_eos
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Machamp model (caio evaluator only)")
    parser.add_argument('--encoder', type=str, required=True, help="Name of the encoder used from HF")
    parser.add_argument('--encoding', type=str, required=True, help="Constituency encoding")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument('--device', type=str, default='0', help="Device to run the evaluation on")
    parser.add_argument('--predict', action='store_true', help="Whether to predict (default: False)")
    parser.add_argument('--no-predict', dest='predict', action='store_false', help="Whether to use existing predictions")
    parser.add_argument('--by-label', action='store_true', help="Evaluate metrics by label")
    parser.set_defaults(predict=True, by_depth=True, by_label=True)

    args = parser.parse_args()

    evaluator = Evaluator(args.encoder, args.dataset, args.encoding, args.device)
    all_results = []

    # Base directory for model evaluations
    model_dirs = f'logs/machamp/{evaluator.dataset}/{evaluator.encoder}/{evaluator.encoding}/'
    gold_data = f'data/{evaluator.dataset}/test.data'



    for seed in evaluator.seeds:
        print(f"🔍 Evaluating seed: {seed}")
        
        if args.predict:
            print(args.predict)
            predicted_labels = evaluator.predict(seed)
            predicted_labels = add_bos_eos(predicted_labels)
        else:
            predicted_labels = f'{model_dirs}seed_{seed}/output.labels'
            if not os.path.exists(predicted_labels):
                print(f"❌ Prediction file not found for seed {seed}. Skipping...")
                continue

        # Decode the predicted labels
        pred_trees = decode(args.encoding, predicted_labels, predicted_labels.replace('labels', 'trees'))
        pred_data = trees_to_data(pred_trees, pred_trees.replace('trees', 'data'))

        # Create a dictionary to store all results for this seed
        seed_results = {}
        
        # Calculate standard metrics
        print(f"📊 Calculating standard metrics for seed {seed}...")
        seed_results["overall"] = evaluator.calculate_metrics(gold_data, pred_data)

        # Calculate metrics by depth if requested
        #if args.by_depth:
        #    print(f"📊 Calculating metrics by depth for seed {seed}...")
        #    seed_results["by_depth"] = evaluator.calculate_metrics_by_depth(gold_data, pred_data)

        # Calculate metrics by label if requested
        if args.by_label:
            print(f"🏷️ Calculating metrics by label for seed {seed}...")
            seed_results["by_label"] = evaluator.calculate_metrics_by_label(gold_data, pred_data)

        # Add the consolidated results to our list
        all_results.append(seed_results)

        # Save individual seed result in proper format
        seed_dir = os.path.join(model_dirs, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, "results.json"), "w") as f:
            json.dump(seed_results, f, indent=2)

    # Compute average across all seeds
    avg_results = average_dictionary(all_results)

    # Save averaged results in the base model directory
    os.makedirs(model_dirs, exist_ok=True)
    with open(os.path.join(model_dirs, "avg_results.json"), "w") as f:
        json.dump(avg_results, f, indent=2)

    print("✅ Done! Per-seed results and averaged results saved.")
