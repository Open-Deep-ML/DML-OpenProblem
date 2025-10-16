## Solution Explanation

We compare two numeric samples (reference vs. current) using mean and variance with user-defined thresholds.

### Definitions
- Mean: \( \mu = \frac{1}{N}\sum_i x_i \)
- Population variance: \( \sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2 \)

### Drift rules
- Mean drift if \(|\mu_{ref} - \mu_{cur}| > \tau_{mean}\)
- Variance drift if \(|\sigma^2_{ref} - \sigma^2_{cur}| > \tau_{var}\)

### Edge cases
- If either sample is empty, return `(False, False)` to avoid false alarms.
- Population vs. sample variance: we use population here to match many monitoring setups. Either is fine if used consistently.

### Complexity
- O(N + M) to compute stats; O(1) extra space.
