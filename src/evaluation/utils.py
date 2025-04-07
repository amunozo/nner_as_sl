from collections import defaultdict


def average_dictionary(data_list):
    result = {
        "overall": defaultdict(float),
        "by_label": defaultdict(lambda: defaultdict(float)),
        "by_depth": defaultdict(lambda: defaultdict(float))
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

    # Calculate averages
    # Average overall metrics
    for metric in result["overall"]:
        result["overall"][metric] /= count
    
    # Average by_label metrics
    for label in result["by_label"]:
        for metric in result["by_label"][label]:
            result["by_label"][label][metric] /= count
    
    # Average by_depth metrics
    for depth in result["by_depth"]:
        for metric in result["by_depth"][depth]:
            result["by_depth"][depth][metric] /= count

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
        }
    }