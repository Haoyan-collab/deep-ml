import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    entropy = 0
    for label,times in cnt.items():
        p = times / n
        entropy -= p * math.log2(p)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    labels = [example[target_attr] for example in examples]
    total_entropy = calculate_entropy(labels)
    # calculate_ig
    group = {}
    for example in examples:
        value = example[attr]
        label = example[target_attr]
        if value not in group:
            group[value] = []
        group[value].append(label)
    weighted_entropy = 0
    for labels in group.values():
        entropy = calculate_entropy(labels)
        weighted_entropy += (len(labels)/len(examples)) * entropy
    
    return total_entropy - weighted_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    # Your code here
    labels = [example[target_attr] for example in examples]
    cnt = Counter(labels)
    max_cnt = max(cnt.values())
    candidates = [label for label,times in cnt.items() if times == max_cnt]
    return min(candidates)

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Your code here
    labels = [example[target_attr] for example in examples]
    if len(set(labels)) == 1:
        return labels[0]

    if len(attributes) == 0:
        return majority_class(examples, target_attr)

    best_attr = max(
        attributes,
        key = lambda attr: calculate_information_gain(examples,attr,target_attr)
    )

    tree = {best_attr: {}}
    remaining_attrs = [attr for attr in attributes if attr != best_attr]

    group = {}
    for example in examples:
        value = example[best_attr]
        if value not in group:
            group[value] = []
        # subtree might use some other attributes
        group[value].append(example)

    for value in sorted(group.keys()):
        examples_subset = group[value]
        subtree = learn_decision_tree(examples_subset,remaining_attrs,target_attr)
        tree[best_attr][value] = subtree
    
    return tree

