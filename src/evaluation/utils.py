from collections import defaultdict


def average_dictionary(data_list):
    result = defaultdict(lambda: defaultdict(float))
    count = len(data_list)

    for data in data_list:
        for outer_key, inner_value in data.items():
            if isinstance(inner_value, dict):
                for key, value in inner_value.items():
                    if isinstance(value, (int, float)):
                        result[outer_key][key] += value
            elif isinstance(inner_value, (int, float)):
                result["__flat__"][outer_key] += inner_value

    # Divide by count
    for outer_key, inner_dict in result.items():
        for key in inner_dict:
            result[outer_key][key] /= count

    return {outer_key: dict(inner_dict) for outer_key, inner_dict in result.items()}