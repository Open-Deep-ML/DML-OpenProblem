import numpy as np


def softmax(values):
    exps = np.exp(values - np.max(values))
    return exps / np.sum(exps)


def pattern_weaver(n, crystal_values, dimension):
    dimension_sqrt = np.sqrt(dimension)
    final_patterns = []

    for i in range(n):
        attention_scores = []
        for j in range(n):
            score = crystal_values[i] * crystal_values[j] / dimension_sqrt
            attention_scores.append(score)

        softmax_scores = softmax(attention_scores)
        weighted_sum = sum(softmax_scores[k] * crystal_values[k] for k in range(n))
        final_patterns.append(round(weighted_sum, 4))

    return final_patterns
