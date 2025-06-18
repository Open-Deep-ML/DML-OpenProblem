import math
from collections import Counter

def calculate_entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())
    return entropy

def calculate_information_gain(examples, attr, target_attr):
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0
    for value in values:
        value_subset = [example[target_attr] for example in examples if example[attr] == value]
        value_entropy = calculate_entropy(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy
    return total_entropy - attr_entropy

def majority_class(examples, target_attr):
    return Counter([example[target_attr] for example in examples]).most_common(1)[0][0]

def learn_decision_tree(examples, attributes, target_attr):
    if not examples:
        return 'No examples'
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)
    
    gains = {attr: calculate_information_gain(examples, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}
    
    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree
    
    return tree

def test_learn_decision_tree():
    # Test case
    examples = [
        {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
    ]
    attributes = ['Outlook', 'Wind']
    expected_output = {
        'Outlook': {
            'Sunny': {'Wind': {'Weak': 'No', 'Strong': 'No'}},
            'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}},
            'Overcast': 'Yes'
        }
    }
    assert learn_decision_tree(examples, attributes, 'PlayTennis') == expected_output, "Test case failed"

if __name__ == "__main__":
    test_learn_decision_tree()
    print("All learn_decision_tree tests passed.")
