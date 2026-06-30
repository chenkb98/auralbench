import json

from auralbench.statistics import bootstrap_mean, paired_difference, permutation_test

baseline, candidate = [0.3, 0.4, 0.2, 0.1, 0.5], [0.2, 0.3, 0.2, 0.1, 0.4]
differences = paired_difference(candidate, baseline)
print(
    json.dumps(
        {
            "synthetic": True,
            "difference_interval": bootstrap_mean(differences, samples=500, seed=7),
            "paired_test": permutation_test(candidate, baseline, samples=500, seed=7),
        }
    )
)
