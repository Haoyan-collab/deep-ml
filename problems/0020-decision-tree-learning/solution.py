import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    # Your implementation here
    if len(labels)==0:
        return 0.0
    entropy = 0
    cnt = Counter(labels)
    for label,times in cnt.items():
        p = times / len(labels)
        entropy -= math.log2(p)*p
    return entropy


def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    # Your implementation here
    labels = [example[target_attr] for example in examples]
    total = calculate_entropy(labels)
    group = {}
    for example in examples:
        feat = example[attr]
        if feat not in group:
            group[feat] = []
        group[feat].append(example[target_attr])
    weighted = 0
    for labels in group.values():
        weighted += (len(labels)/len(examples))*calculate_entropy(labels)
    
    return total - weighted


def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """
    # Your implementation here
    labels = [example[target_attr] for example in examples]
    cnt = Counter(labels)
    max_cnt = max(cnt.values())
    candidates = [value for value,times in cnt.items() if times == max_cnt]
    return min(candidates)


def learn_decision_tree(
    examples: List[Dict[str, Any]],
    attributes: List[str],
    target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """
    # Your implementation here
    labels = [example[target_attr] for example in examples]
    if len(set(labels)) == 1:
        return labels[0]
    
    if len(attributes) == 0:
        return majority_class(examples,target_attr)
    
    best_attr = max(sorted(attributes),
        key = lambda attr:calculate_information_gain(examples,attr,target_attr))

    remain_attr = [attr for attr in attributes if attr != best_attr]
    tree = {best_attr:{}}

    group = {}
    for example in examples:
        feat = example[best_attr]
        if feat not in group:
            group[feat] = []
        group[feat].append(example)
    
    for feat in sorted(group.keys()):
        sub_example = group[feat]
        subtree = learn_decision_tree(sub_example,remain_attr,target_attr)
        tree[best_attr][feat] = subtree
    
    return tree









