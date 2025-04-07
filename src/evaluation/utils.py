from collections import defaultdict


def average_dictionary(data_list):
    result = {
        "overall": defaultdict(float),
        "by_label": defaultdict(lambda: defaultdict(float))
    }
    count = len(data_list)

    # Sum up all values
    for data in data_list:
        # Handle overall metrics
        for metric, value in data["overall"].items():
            result["overall"][metric] += value
        
        # Handle by_label metrics
        for label, metrics in data["by_label"].items():
            for metric, value in metrics.items():
                result["by_label"][label][metric] += value

    # Calculate averages
    # Average overall metrics
    for metric in result["overall"]:
        result["overall"][metric] /= count
    
    # Average by_label metrics
    for label in result["by_label"]:
        for metric in result["by_label"][label]:
            result["by_label"][label][metric] /= count

    # Convert defaultdicts to regular dicts
    return {
        "overall": dict(result["overall"]),
        "by_label": {
            label: dict(metrics) 
            for label, metrics in result["by_label"].items()
        }
    }