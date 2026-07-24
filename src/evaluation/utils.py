"""Aggregation helpers for repeated evaluation runs."""

import math


SECTIONS = ("overall", "by_label", "by_depth", "by_length")


def _summary(values, total_runs):
    values = [float(value) for value in values]
    values.extend([0.0] * (total_runs - len(values)))
    mean = sum(values) / total_runs
    variance = sum(value * value for value in values) / total_runs - mean * mean
    return {"mean": mean, "std": math.sqrt(max(0.0, variance))}


def average_dictionary(data_list):
    """Return population means and standard deviations across seeds.

    A subgroup absent from one run contributes zero for that run. This matters
    for predicted-only labels, lengths, or depths that are not produced by every
    seed.
    """
    if not data_list:
        return {section: {} for section in SECTIONS}

    total_runs = len(data_list)
    result = {section: {} for section in SECTIONS}
    overall_metrics = {
        metric
        for run in data_list
        for metric in run.get("overall", {})
    }
    for metric in sorted(overall_metrics):
        values = [
            run["overall"][metric]
            for run in data_list
            if metric in run.get("overall", {})
        ]
        result["overall"][metric] = _summary(values, total_runs)

    for section in ("by_label", "by_depth", "by_length"):
        groups = {
            group
            for run in data_list
            for group in run.get(section, {})
        }
        for group in sorted(groups, key=str):
            metrics = {
                metric
                for run in data_list
                for metric in run.get(section, {}).get(group, {})
            }
            result[section][group] = {}
            for metric in sorted(metrics):
                values = [
                    run[section][group][metric]
                    for run in data_list
                    if metric in run.get(section, {}).get(group, {})
                ]
                result[section][group][metric] = _summary(values, total_runs)
    return result
