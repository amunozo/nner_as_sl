import os
import sys
import statistics
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.utils import find_entities


def calculate_nesting_depth(entities):
    """Sort entities by start position and then by end position (descending)"""
    sorted_entities = sorted(entities, key=lambda x: (x[1], -x[2]))
    depths = {}
    
    for entity in sorted_entities:
        depth = 1

        for other in sorted_entities:
            if entity == other:
                continue
            if other[1] <= entity[1] and entity[2] <= other[2]:
                if other in depths:
                    depth = max(depth, depths[other] + 1)
                else:
                    depth = max(depth, 2)

        depths[entity] = depth
    return depths

def entity_count(dataset):
    """
    Extract the number of entities in the three splits of a given dataset,
    and save the counts in a JSON file.
    
    Returns:
        dict: Statistics about entities including counts, percentages, average depth, median depth, etc.
    """
    n_gold = {
        "all": 0,
    }
    dataset_directory = f"data/{dataset}/"
    files = ["train.data", "dev.data", "test.data"]
    all_depths = set()
    total_sentence_count = 0
    entity_types_counter = Counter()
    
    # For calculating median and average depth
    all_entity_depths = []
    
    for file in files:
        entities = find_entities(f"{dataset_directory}{file}")
        total_sentence_count += len(entities)
        n_gold["all"] += sum(len(sentence) for sentence in entities)

        for sentence in entities:
            depths = calculate_nesting_depth(sentence)
            
            # Add all depth values to list for statistics
            all_entity_depths.extend(depths.values())
            
            # Count entities at each depth
            for depth in set(depths.values()):
                all_depths.add(depth)
                n_gold[depth] = n_gold.get(depth, 0) + len([e for e in sentence if depths[e] == depth])
            
            # Count entity types
            for entity in sentence:
                entity_type = entity[0]
                entity_types_counter[entity_type] += 1
        
    # Calculate percentages for each depth
    depth_percentages = {}
    for depth in all_depths:
        if n_gold["all"] > 0:  # Avoid division by zero
            depth_percentages[depth] = round((n_gold[depth] / n_gold["all"]) * 100, 2)
    
    # Calculate average and median depths
    average_depth = round(statistics.mean(all_entity_depths), 2) if all_entity_depths else 0
    median_depth = round(statistics.median(all_entity_depths), 2) if all_entity_depths else 0
    
    # Prepare result dictionary with all statistics
    result = {
        "counts": n_gold,
        "percentages": depth_percentages,
        "average_depth": average_depth,
        "median_depth": median_depth,
        "sentence_count": total_sentence_count,
        "entity_types": dict(entity_types_counter),
        "all_depths": sorted(list(all_depths))
    }
    
    return result


if __name__ == "__main__":
    import pandas as pd
    from tabulate import tabulate
    
    datasets = ["ace2004", "ace2005", "nne", "genia"]
    
    # Create dictionaries to store data for each dataset
    data = {
        "Dataset": [],
        "Total Entities": [],
        "Sentences": [],
        "Avg Depth": [],
        "Median Depth": [],
        "Unique Entity Types": []
    }
    
    # Create separate depth data
    depth_data = defaultdict(list)
    
    for dataset in datasets:
        result = entity_count(dataset)
        
        # Add basic info to main data
        data["Dataset"].append(dataset)
        data["Total Entities"].append(result["counts"]["all"])
        data["Sentences"].append(result["sentence_count"])
        data["Avg Depth"].append(result["average_depth"])
        data["Median Depth"].append(result["median_depth"])
        data["Unique Entity Types"].append(len(result["entity_types"]))
        
        # Add depths data
        for depth in result["all_depths"]:
            depth_key = f"Depth {depth}"
            
            # Ensure all depth columns exist for all datasets
            if depth_key not in depth_data:
                depth_data[depth_key] = [0] * (len(data["Dataset"]) - 1)
            
            # Add the count and percentage
            count = result["counts"].get(depth, 0)
            percentage = result["percentages"].get(depth, 0)
            depth_data[depth_key].append(f"{count} ({percentage}%)")
    
    # Make sure all depth columns have values for all datasets
    for depth_key in depth_data:
        while len(depth_data[depth_key]) < len(data["Dataset"]):
            depth_data[depth_key].append("0 (0.0%)")
    
    # Combine the main data with depth data
    for depth_key, values in depth_data.items():
        data[depth_key] = values
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print as formatted table
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    
    # Optionally save to CSV or Excel
    df.to_csv("entity_depth_statistics.csv", index=False)
    
    # Print entity type distribution as separate tables
    print("\nEntity Type Distribution:")
    for dataset in datasets:
        result = entity_count(dataset)
        entity_types = result["entity_types"]
        
        # Convert to DataFrame for nice display
        et_df = pd.DataFrame({
            "Entity Type": list(entity_types.keys()),
            "Count": list(entity_types.values())
        })
        et_df["Percentage"] = et_df["Count"] / et_df["Count"].sum() * 100
        et_df["Percentage"] = et_df["Percentage"].round(2).astype(str) + "%"
        
        print(f"\n{dataset.upper()} Entity Types:")
        print(tabulate(et_df.sort_values("Count", ascending=False), 
                       headers="keys", tablefmt="grid", showindex=False))