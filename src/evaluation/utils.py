from collections import defaultdict


def average_dictionary(data_list):
    # Initialize a defaultdict of defaultdict to store sum of values
    result = defaultdict(lambda: defaultdict(float))
    count = len(data_list)
    
    for data in data_list:
        for outer_key, inner_dict in data.items():
            for key, value in inner_dict.items():
                if isinstance(value, (int, float)):  # Only sum numeric values
                    result[outer_key][key] += value
    
    for outer_key, inner_dict in result.items():
        for key in inner_dict:
            result[outer_key][key] /= count
    
    return {outer_key: dict(inner_dict) for outer_key, inner_dict in result.items()}