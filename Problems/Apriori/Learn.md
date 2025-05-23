The **Apriori algorithm** is a method for **association rule mining**, used to discover frequent itemsets. 
It follows a **"bottom-up" approach**, iteratively finding patterns by leveraging the 
**anti-monotonicity principle**: *"If an itemset is infrequent, all its supersets must also be infrequent."*  

## Algorithm Steps
1. **Generate** candidate itemsets of size k
2. **Count** occurrences in transactions (support)
3. **Filter** by minimum support threshold
4. **Repeat** for k+1 until no more frequent itemsets

## Example Calculation

### Input Data
| Transaction | Items                      |
|-------------|----------------------------|
| 1           | Bread, Milk                |
| 2           | Bread, Diaper, Beer        |
| 3           | Milk, Diaper, Beer         |
| 4           | Bread, Milk, Diaper        |
| 5           | Bread, Milk, Cola          |

### Parameters
- `min_support = 0.6` (must appear in ≥3/5 transactions)

### Step 1: Find Frequent 1-itemsets
| Itemset | Count | Support | Frequent? |
|---------|-------|---------|-----------|
| Bread   | 4     | 4/5     | Yes       |
| Milk    | 4     | 4/5     | Yes       |
| Diaper  | 3     | 3/5     | Yes       |
| Beer    | 2     | 2/5     | No        |
| Cola    | 1     | 1/5     | No        |

### Step 2: Generate and Test 2-itemsets
| Itemset        | Count | Support | Frequent? |
|----------------|-------|---------|-----------|
| Bread, Milk    | 3     | 3/5     | Yes       |
| Bread, Diaper  | 2     | 2/5     | No        |
| Milk, Diaper   | 2     | 2/5     | No        |

### Final Frequent Itemsets
- **1-itemsets**: Bread, Milk, Diaper
- **2-itemsets**: {Bread, Milk}
