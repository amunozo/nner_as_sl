from collections import defaultdict


def average_dictionary(data_list):
    result = {
        "overall": defaultdict(float),
        "by_label": defaultdict(lambda: defaultdict(float)),
        "by_depth": defaultdict(lambda: defaultdict(float)),
        "by_length": defaultdict(lambda: defaultdict(float)),  # <-- Add this line
    }
    count = len(data_list)

    # Sum up all values
    for data in data_list:
        # Handle overall metrics
        for metric, value in data["overall"].items():
            result["overall"][metric] += value
        
        # Handle by_label metrics
        if "by_label" in data:
            for label, metrics in data["by_label"].items():
                for metric, value in metrics.items():
                    result["by_label"][label][metric] += value
        
        # Handle by_depth metrics
        if "by_depth" in data:
            for depth, metrics in data["by_depth"].items():
                for metric, value in metrics.items():
                    result["by_depth"][depth][metric] += value

        # Handle by_length metrics
        if "by_length" in data:
            for length, metrics in data["by_length"].items():
                for metric, value in metrics.items():
                    result["by_length"][length][metric] += value

    # Calculate averages
    for metric in result["overall"]:
        result["overall"][metric] /= count
    
    for label in result["by_label"]:
        for metric in result["by_label"][label]:
            result["by_label"][label][metric] /= count
    
    for depth in result["by_depth"]:
        for metric in result["by_depth"][depth]:
            result["by_depth"][depth][metric] /= count

    for length in result["by_length"]:
        for metric in result["by_length"][length]:
            result["by_length"][length][metric] /= count

    # Convert defaultdicts to regular dicts
    return {
        "overall": dict(result["overall"]),
        "by_label": {
            label: dict(metrics) 
            for label, metrics in result["by_label"].items()
        },
        "by_depth": {
            depth: dict(metrics)
            for depth, metrics in result["by_depth"].items()
        },
        "by_length": {
            length: dict(metrics)
            for length, metrics in result["by_length"].items()
        }
    }