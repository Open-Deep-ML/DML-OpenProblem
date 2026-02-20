import math
from collections import Counter
from tinygrad.tensor import Tensor
from typing import List, Dict, Any, Union


def calculate_entropy_tg(labels) -> float:
    arr = labels.tolist() if isinstance(labels, Tensor) else labels
    total = len(arr)
    cnt = Counter(arr)
    return -sum((c / total) * math.log2(c / total) for c in cnt.values())


def calculate_information_gain_tg(
    examples: List[Dict[str, Any]], attr: str, target_attr: str
) -> float:
    total = [ex[target_attr] for ex in examples]
    total_ent = calculate_entropy_tg(total)
    n = len(examples)
    rem = 0.0
    for v in set(ex[attr] for ex in examples):
        subset = [ex[target_attr] for ex in examples if ex[attr] == v]
        rem += (len(subset) / n) * calculate_entropy_tg(subset)
    return total_ent - rem


def majority_class_tg(examples: List[Dict[str, Any]], target_attr: str) -> Any:
    return Counter(ex[target_attr] for ex in examples).most_common(1)[0][0]


def learn_decision_tree_tg(
    examples: List[Dict[str, Any]], attributes: List[str], target_attr: str
) -> Union[Dict[str, Any], Any]:
    if not examples:
        return "No examples"
    first_label = examples[0][target_attr]
    if all(ex[target_attr] == first_label for ex in examples):
        return first_label
    if not attributes:
        return majority_class_tg(examples, target_attr)
    gains = {
        a: calculate_information_gain_tg(examples, a, target_attr) for a in attributes
    }
    best = max(gains, key=gains.get)
    tree: Dict[str, Any] = {best: {}}
    for v in set(ex[best] for ex in examples):
        subset = [ex for ex in examples if ex[best] == v]
        rem_attrs = [a for a in attributes if a != best]
        tree[best][v] = learn_decision_tree_tg(subset, rem_attrs, target_attr)
    return tree
