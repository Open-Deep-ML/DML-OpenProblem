from typing import List, Dict, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    # Your implementation here
    pass


def calculate_information_gain(
    examples: List[Dict[str, Any]], attr: str, target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    # Your implementation here
    pass


def majority_class(examples: List[Dict[str, Any]], target_attr: str) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    """
    # Your implementation here
    pass


def learn_decision_tree(
    examples: List[Dict[str, Any]], attributes: List[str], target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """
    # Your implementation here
    pass
