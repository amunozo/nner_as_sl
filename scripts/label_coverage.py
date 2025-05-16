import os
import sys
import csv
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
from src.data.utils import to_parenthesized, encode, trees_to_data, decode, add_bos_eos, find_entities


def create_joint_file(dataset):
    data_dir = os.path.join("data", dataset)
    train, dev, test = [
        os.path.join(data_dir, f"{split}.data") 
        for split in ["train", "dev", "test"]
    ]
    joint_text = ""
    for file in [train, dev, test]:
        with open(file, 'r', encoding='utf-8') as file_:
            lines = file_.readlines()
            for line in lines:
                if '-BOS-' not in line and '-EOS-' not in line:
                    joint_text += line.strip() + "\n"

    joint_gold = tempfile.NamedTemporaryFile(delete=False)
    with open(joint_gold.name, 'w', encoding='utf-8') as file_:
        file_.write(joint_text)
    return joint_gold.name

def max_possible_recall(filename, encoding):
    """
    Calculate the maximum possible recall for a given dataset and encoding.
    """    
    # encoding
    trees_file = tempfile.NamedTemporaryFile(delete=False)
    trees_file_name = to_parenthesized(
        filename, trees_file.name
    )

    encoded_file = tempfile.NamedTemporaryFile(delete=False)
    encoded_file_name = encode(
        encoding, trees_file_name, encoded_file.name,
    )
    # decoding
    decoded_file = tempfile.NamedTemporaryFile(delete=False)
    decoded_file_name = decode(
        encoding, encoded_file_name, decoded_file.name
    )
    decoded_file_name = trees_to_data(
        decoded_file_name, decoded_file_name
    )

    # calculate recall
    gold_entities = find_entities(filename)  # Use input filename, not joint_gold.name
    predicted_entities = find_entities(decoded_file_name)
    
    n_correct = 0
    n_gold = 0
    id = 0
    
    for gold, pred in zip(gold_entities, predicted_entities):
        id += 1
        n_correct += len(gold.intersection(pred))
        n_gold += len(gold)
        missed_entities = gold - gold.intersection(pred)
        if missed_entities:
            print(f"Missed entities for sentence {id}: {missed_entities}")
            print(f"Missed entities: {missed_entities}")
    
    # Return max possible recall as a percentage
    max_recall = 0 if n_gold == 0 else (n_correct / n_gold)
    
    # Clean up temporary files
    os.unlink(trees_file.name)
    os.unlink(encoded_file.name)
    os.unlink(decoded_file.name)
    
    return {
        "max_recall": max_recall,
        "correct_entities": n_correct,
        "gold_entities": n_gold,
        "predicted_entities": sum(len(pred) for pred in predicted_entities)
    }

def save_results_to_csv(results, output_file):
    """
    Save results to a CSV file.
    
    Args:
        results (list): List of dictionaries with results
        output_file (str): Path to output CSV file
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all unique keys to use as headers
    headers = set()
    for result in results:
        headers.update(result.keys())
    headers = sorted(list(headers))
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Setup output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/label_coverage.csv"
    
    # Example usage
    datasets = ["ace2004", "ace2005", "nne", "genia"]
    encodings = ["ABS", "REL", "DYN", "4EC"]
    
    # Store all results for CSV
    all_results = []
    
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        data_file = create_joint_file(dataset_name)
        
        # Check if file exists
        if not os.path.exists(data_file):
            print(f"Warning: File {data_file} does not exist. Skipping.")
            continue
            
        for encoding_type in encodings:
            print(f"  Testing encoding: {encoding_type}")
            
            # Run the test
            result = max_possible_recall(data_file, encoding_type)
            
            # Add metadata to result
            result["dataset"] = dataset_name
            result["encoding"] = encoding_type
            result["max_recall_percentage"] = f"{result['max_recall']:.2%}"
            
            # Store result for CSV
            all_results.append(result)
            
            # Print progress
            print(f"  - Max Recall: {result['max_recall']:.2%}, "
                  f"Correct: {result['correct_entities']}, "
                  f"Gold: {result['gold_entities']}")
    
    # Save all results to CSV
    save_results_to_csv(all_results, output_file)