import os
import sys

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
    """
    metrics = {
        "all": 0,
    }
    dataset_directory = f"data/{dataset}/"
    files = ["train.data", "dev.data", "test.data"]
    all_depths = set()

    for file in files:
        entities = find_entities(f"{dataset_directory}{file}")
        n_gold["all"] += sum(len(sentence) for sentence in entities)

        for sentence in entities:
            depths = calculate_nesting_depth(sentence)
            for depth in set(depths.values()):
                all_depths.add(depth)
            n_gold[depth] = n_gold.get(depth, 0) + len([e for e in sentence if depths[e] == depth])
        
    n_gold["all"] = n_gold.get("all", 0)

    return n_gold, all_depths

print(entity_count("ace2005"))