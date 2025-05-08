from collections import defaultdict
import math


def average_dictionary(data_list):
    if not data_list:
        return {
            "overall": {},
            "by_label": {},
            "by_depth": {},
            "by_length": {},
        }

    count = len(data_list)

    # Initialize structures to store sums and sums of squares
    result_sums = {
        "overall": defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0}),
        "by_label": defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0})),
        "by_depth": defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0})),
        "by_length": defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0})),
    }

    # Sum up all values and their squares
    for data in data_list:
        # Handle overall metrics
        if "overall" in data:
            for metric, value in data["overall"].items():
                result_sums["overall"][metric]["sum"] += value
                result_sums["overall"][metric]["sum_sq"] += value ** 2
        
        # Handle by_label metrics
        if "by_label" in data:
            for label, metrics in data["by_label"].items():
                for metric, value in metrics.items():
                    result_sums["by_label"][label][metric]["sum"] += value
                    result_sums["by_label"][label][metric]["sum_sq"] += value ** 2
        
        # Handle by_depth metrics
        if "by_depth" in data:
            for depth, metrics in data["by_depth"].items():
                for metric, value in metrics.items():
                    result_sums["by_depth"][depth][metric]["sum"] += value
                    result_sums["by_depth"][depth][metric]["sum_sq"] += value ** 2

        # Handle by_length metrics
        if "by_length" in data:
            for length, metrics in data["by_length"].items():
                for metric, value in metrics.items():
                    result_sums["by_length"][length][metric]["sum"] += value
                    result_sums["by_length"][length][metric]["sum_sq"] += value ** 2

    # Prepare the final result dictionary
    final_result = {
        "overall": {},
        "by_label": defaultdict(dict),
        "by_depth": defaultdict(dict),
        "by_length": defaultdict(dict),
    }

    # Calculate means and standard deviations for overall metrics
    for metric, sums_data in result_sums["overall"].items():
        mean = sums_data["sum"] / count
        variance = (sums_data["sum_sq"] / count) - (mean ** 2)
        std_dev = math.sqrt(max(0, variance))  # max(0, ...) to avoid domain error with sqrt
        final_result["overall"][metric] = {"mean": mean, "std": std_dev}
    
    # Calculate means and standard deviations for by_label metrics
    for label, metrics_data in result_sums["by_label"].items():
        for metric, sums_data in metrics_data.items():
            mean = sums_data["sum"] / count
            variance = (sums_data["sum_sq"] / count) - (mean ** 2)
            std_dev = math.sqrt(max(0, variance))
            final_result["by_label"][label][metric] = {"mean": mean, "std": std_dev}
    
    # Calculate means and standard deviations for by_depth metrics
    for depth, metrics_data in result_sums["by_depth"].items():
        for metric, sums_data in metrics_data.items():
            mean = sums_data["sum"] / count
            variance = (sums_data["sum_sq"] / count) - (mean ** 2)
            std_dev = math.sqrt(max(0, variance))
            final_result["by_depth"][depth][metric] = {"mean": mean, "std": std_dev}

    # Calculate means and standard deviations for by_length metrics
    for length, metrics_data in result_sums["by_length"].items():
        for metric, sums_data in metrics_data.items():
            mean = sums_data["sum"] / count
            variance = (sums_data["sum_sq"] / count) - (mean ** 2)
            std_dev = math.sqrt(max(0, variance))
            final_result["by_length"][length][metric] = {"mean": mean, "std": std_dev}

    # Convert defaultdicts to regular dicts for the final output
    return {
        "overall": dict(final_result["overall"]),
        "by_label": {
            label: dict(metrics) 
            for label, metrics in final_result["by_label"].items()
        },
        "by_depth": {
            depth: dict(metrics)
            for depth, metrics in final_result["by_depth"].items()
        },
        "by_length": {
            length: dict(metrics)
            for length, metrics in final_result["by_length"].items()
        }
    }