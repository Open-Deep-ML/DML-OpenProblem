import itertools
from collections import defaultdict

def apriori(transactions, min_support=0.5, max_length=None):
    if not transactions:
        raise ValueError("Transaction list cannot be empty")
    if not 0 < min_support <= 1:
        raise ValueError("Minimum support must be between 0 and 1")
    
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions
    
    # Find frequent 1-itemsets
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
    
    frequent_itemsets = {itemset: count for itemset, count in item_counts.items() 
                         if count >= min_support_count}
    
    k = 1  # Current itemset size
    all_frequent_itemsets = dict(frequent_itemsets)
    
    # Generate frequent itemsets of size k+1 until no more can be found
    while frequent_itemsets and (max_length is None or k < max_length):
        k += 1
        candidates = generate_candidates(frequent_itemsets.keys(), k)
        
        candidate_counts = defaultdict(int)
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1
        
        frequent_itemsets = {itemset: count for itemset, count in candidate_counts.items()
                            if count >= min_support_count}
        
        all_frequent_itemsets.update(frequent_itemsets)
    
    return {itemset: count / num_transactions for itemset, count in all_frequent_itemsets.items()}

def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    prev_frequent_list = list(prev_frequent_itemsets)
    
    for i in range(len(prev_frequent_list)):
        for j in range(i + 1, len(prev_frequent_list)):
            itemset1 = prev_frequent_list[i]
            itemset2 = prev_frequent_list[j]

            # Merge only if they have k-1 common items
            union_set = itemset1 | itemset2
            if len(union_set) == k:
                # Only add if all subsets of size k-1 are frequent
                if all(frozenset(subset) in prev_frequent_itemsets 
                       for subset in get_subsets(union_set, k-1)):
                    candidates.add(union_set)
    
    return candidates

def get_subsets(itemset, size):
    return [frozenset(subset) for subset in itertools.combinations(itemset, size)]

def test_apriori():
    transactions1 = [{'bread', 'milk'}, 
                     {'bread', 'diaper', 'beer', 'eggs'}, 
                     {'milk', 'diaper', 'beer', 'cola'}, 
                     {'bread', 'milk', 'diaper', 'beer'}, 
                     {'bread', 'milk', 'diaper', 'cola'}]
    
    result1 = apriori(transactions1, min_support=0.6)
    expected1 = {
        frozenset({'bread'}): 0.8,
        frozenset({'milk'}): 0.8,
        frozenset({'diaper'}): 0.8,
        frozenset({'bread', 'milk'}): 0.6,
        frozenset({'milk', 'diaper'}): 0.6,
        frozenset({'bread', 'diaper'}): 0.6
    }
    assert set(result1.keys()) == set(expected1.keys()), "Test Case 1 Failed"
    assert all(abs(result1[k] - expected1[k]) < 0.001 for k in expected1), "Test Case 1 Failed"

    transactions2 = [{'a', 'b'}, {'c', 'd'}, {'e', 'f'}, {'g', 'h'}]
    result2 = apriori(transactions2, min_support=0.5)
    expected2 = {}  # No itemset appears in at least 2 transactions
    assert set(result2.keys()) == set(expected2.keys()), "Test Case 2 Failed"

    transactions3 = [{'a', 'b', 'c', 'd'}, {'a', 'b', 'c', 'd'}, {'a', 'b', 'c', 'd'}]
    result3 = apriori(transactions3, min_support=0.5, max_length=2)
    expected3 = {
        frozenset({'a'}): 1.0,
        frozenset({'b'}): 1.0,
        frozenset({'c'}): 1.0,
        frozenset({'d'}): 1.0,
        frozenset({'a', 'b'}): 1.0,
        frozenset({'a', 'c'}): 1.0,
        frozenset({'a', 'd'}): 1.0,
        frozenset({'b', 'c'}): 1.0,
        frozenset({'b', 'd'}): 1.0,
        frozenset({'c', 'd'}): 1.0
    }
    assert set(result3.keys()) == set(expected3.keys()), "Test Case 3 Failed"
    
    try:
        apriori([], min_support=0.5)
        assert False, "Test Case 4 Failed: Should raise ValueError"
    except ValueError:
        pass

    transactions5 = [
        {'apple', 'banana', 'orange'},
        {'apple', 'banana', 'grape'},
        {'apple', 'orange', 'grape'},
        {'banana', 'orange', 'grape'},
        {'apple', 'banana', 'orange', 'grape'}
    ]
    result5 = apriori(transactions5, min_support=0.6)
    expected5 = {
        frozenset({'apple'}): 0.8,
        frozenset({'banana'}): 0.8,
        frozenset({'orange'}): 0.8,
        frozenset({'grape'}): 0.8,
        frozenset({'apple', 'banana'}): 0.6,
        frozenset({'apple', 'orange'}): 0.6,
        frozenset({'apple', 'grape'}): 0.6,
        frozenset({'banana', 'orange'}): 0.6,
        frozenset({'banana', 'grape'}): 0.6,
        frozenset({'orange', 'grape'}): 0.6
    }
    assert set(result5.keys()) == set(expected5.keys()), "Test Case 5 Failed"

if __name__ == "__main__":
    test_apriori()
    print("All Test Cases Passed!")
