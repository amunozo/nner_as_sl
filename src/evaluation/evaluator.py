import os
import subprocess
from typing import Set, Optional, Dict, List
from src.data.utils import find_entities


class Evaluator:
    def __init__(self, 
                 encoder: Optional[str] = None,
                 dataset: Optional[str] = None,
                 encoding: Optional[str] = None,
                 device: Optional[int] = None):
        self.reset()
        # Model specific attributes
        self.encoder = encoder
        self.dataset = dataset
        self.encoding = encoding
        self.device = device
        
        if encoder and dataset:
            self.model_dirs = f'logs/machamp/{dataset}/{encoder}/{encoding}/'
            self.seeds = sorted([d.split('_')[-1] for d in os.listdir(self.model_dirs) 
                               if d.startswith('seed_') and os.path.isdir(os.path.join(self.model_dirs, d))])

    def reset(self) -> None:
        self.n_correct = 0
        self.n_gold = 0
        self.n_pred = 0
        self.decoder_timing = 0

    def __call__(self, gold: Set, pred: Set) -> None:
        self.n_correct += len(gold.intersection(pred))
        self.n_gold += len(gold)
        self.n_pred += len(pred)

    def precision(self) -> float:
        return 0 if self.n_pred == 0 else self.n_correct / self.n_pred

    def recall(self) -> float:
        return 0 if self.n_gold == 0 else self.n_correct / self.n_gold

    def f1(self) -> float:
        prec = self.precision()
        rec = self.recall()
        return 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    def write(self, ostream, dataset_name: str) -> None:
        ostream.write(f"Eval on {dataset_name} with {getattr(self, 'algorithm', 'default')}:\t"
                     f"prec={100 * self.precision():.2f}\t"
                     f"rec={100 * self.recall():.2f}\t"
                     f"f1={100 * self.f1():.2f}\t"
                     f"timing={self.decoder_timing}")

    def calculate_metrics(self, gold_data: List, predicted_data: List) -> Dict[str, float]:        
        gold_entities = find_entities(gold_data)
        predicted_entities = find_entities(predicted_data)
        
        self.reset()
        for gold, pred in zip(gold_entities, predicted_entities):
            self(gold, pred)
        
        results = {}
        results["precision"] = self.precision()
        results["recall"] = self.recall()
        results["f1"] = self.f1()
        results["n_pred"] = self.n_pred
        results["n_gold"] = self.n_gold
        results["n_correct"] = self.n_correct

        return results
    
    def calculate_metrics_by_length(self, gold_data, predicted_data):
        gold_entities = find_entities(gold_data)
        predicted_entities = find_entities(predicted_data)
        lengths = set()
        for sentence in gold_entities:
            for entity in sentence:
                lengths.add(entity[2] - entity[1])
        
        results = {length: {} for length in lengths}

        for length in lengths:
            self.reset()
            for gold, pred in zip(gold_entities, predicted_entities):
                gold_by_length = {e for e in gold if e[2] - e[1] == length}
                pred_by_length = {e for e in pred if e[2] - e[1] == length}
                self(gold_by_length, pred_by_length)

            results[length]["precision"] = self.precision()
            results[length]["recall"] = self.recall()
            results[length]["f1"] = self.f1()
            results[length]["n_pred"] = self.n_pred
            results[length]["n_gold"] = self.n_gold
            results[length]["n_correct"] = self.n_correct
        
        return results

    def calculate_metrics_by_depth(self, gold_data, predicted_data):
        gold_entities = find_entities(gold_data)
        predicted_entities = find_entities(predicted_data)

        # Calculate depths from BOTH gold and predicted data
        all_depths = set()
        gold_depths_by_sentence = []
        pred_depths_by_sentence = []
        
        for gold, pred in zip(gold_entities, predicted_entities):
            gold_depths = self.calculate_nesting_depth(gold)
            pred_depths = self.calculate_nesting_depth(pred)
            gold_depths_by_sentence.append(gold_depths)
            pred_depths_by_sentence.append(pred_depths)
            all_depths.update(gold_depths.values())

        results = {depth: {} for depth in all_depths}

        for depth in all_depths:
            self.reset()
            for gold, pred, gold_depths, pred_depths in zip(gold_entities, predicted_entities, gold_depths_by_sentence, pred_depths_by_sentence):
                gold_at_depth = {e for e in gold if gold_depths.get(e, 0) == depth}
                pred_at_depth = {e for e in pred if pred_depths.get(e, 0) == depth}
                correct_for_precision = pred_at_depth.intersection(gold_at_depth)

                self.n_gold += len(gold_at_depth)
                self.n_pred += len(pred_at_depth)
                self.n_correct += len(correct_for_precision)

            # For recall, override n_correct for recall calculation
            n_correct_recall = 0
            for gold, pred, gold_depths in zip(gold_entities, predicted_entities, gold_depths_by_sentence):
                gold_at_depth = {e for e in gold if gold_depths.get(e, 0) == depth}
                n_correct_recall += len(gold_at_depth.intersection(pred))

            recall = 0 if self.n_gold == 0 else n_correct_recall / self.n_gold
            precision = self.precision()

            results[depth]["precision"] = precision
            results[depth]["recall"] = recall
            results[depth]["n_pred"] = self.n_pred
            results[depth]["n_gold"] = self.n_gold
            results[depth]["n_correct"] = self.n_correct

        return results
        
    def calculate_nesting_depth(self, entities):
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
    
    def calculate_metrics_by_label(self, gold_data: List, predicted_data: List) -> Dict[str, Dict[str, float]]:
        gold_entities = find_entities(gold_data)
        predicted_entities = find_entities(predicted_data)

        labels = set()
        for sentence in gold_entities:
            for entity in sentence:
                labels.add(entity[0])

        results = {label: {} for label in labels}

        for label in labels:
            self.reset()
            for gold, pred in zip(gold_entities, predicted_entities):
                gold_label = {e for e in gold if e[0] == label}
                pred_label = {e for e in pred if e[0] == label}
                self(gold_label, pred_label)

            results[label]["precision"] = self.precision()
            results[label]["recall"] = self.recall()
            results[label]["f1"] = self.f1()
            results[label]["n_pred"] = self.n_pred
            results[label]["n_gold"] = self.n_gold
            results[label]["n_correct"] = self.n_correct

        return results

    def predict(self, seed: str) -> str:
        if not all([self.encoder, self.dataset, self.encoding]):
            raise ValueError("Model parameters not configured for prediction")
            
        model_dir = f"{self.model_dirs}seed_{seed}"
        input_file = f'data/{self.dataset}/{self.encoding}/test.labels'    
        output_file = f'{model_dir}/output.labels'
        
        subprocess.run([
            'python', 'machamp/predict.py',
            f'{model_dir}/model.pt', input_file, output_file,
            '--device', str(self.device),
            '--dataset', self.dataset
        ], check=True)
                
        return output_file