import math
from collections import Counter
import numpy as np

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    counts = np.array(list(Counter(labels).values()))
    probs = counts/counts.sum()
    return -1 * (probs*np.log2(probs)).sum()


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    IG = calculate_entropy([example[target_attr] for example in examples])
    for val in {example[attr] for example in examples}:
            IG -= calculate_entropy([example[target_attr] for example in examples if example[attr] == val])
    return IG

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    # Your code here
    labels = [example[target_attr] for example in examples]
    return min([(-1*count,l) for l, count in Counter(labels).items()])[1]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Your code here
    if len({example[target_attr] for example in examples}) < 2 or len(attributes)==0:
        return majority_class(examples, target_attr)

    result = {}
    best_attr, best_ig = None, float('-inf')
    for attr in attributes:
        IG = calculate_information_gain(examples,  attr, target_attr)
        if IG > best_ig:  
            best_attr, best_ig = attr, IG
    result[best_attr] = dict()
    for val in sorted({example[best_attr] for example in examples}):
        split = [{attr:example[attr] for attr in example if attr!=best_attr} for example in examples if example[best_attr]==val]
        result[best_attr][val] = learn_decision_tree(split, [attr for attr in attributes if attr!=best_attr], target_attr)
    
    return result
