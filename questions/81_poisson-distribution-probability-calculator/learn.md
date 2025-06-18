## Understanding Poisson Distribution

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space, provided these events occur with a known constant mean rate and independently of the time since the last event.

### Mathematical Definition

The probability of observing \( k \) events in a given interval is defined as:

$$
P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

- **\( k \)**: Number of events (non-negative integer)
- **\( \lambda \)**: The mean number of events in the given interval (rate parameter)
- **\( e \)**: Euler's number, approximately 2.718

### Key Properties

- **Mean**: \( \lambda \)
- **Variance**: \( \lambda \)
- The Poisson distribution is used for modeling rare or random events.

### Example Calculation

Suppose the mean number of calls received in an hour (\( \lambda \)) is 5. Calculate the probability of receiving exactly 3 calls in an hour:

1. **Substitute into the formula**:
   $$
   P(3; 5) = \frac{5^3 e^{-5}}{3!}
   $$

2. **Calculate step-by-step**:
   $$
   P(3; 5) = \frac{125 \cdot e^{-5}}{6} \approx 0.14037
   $$

### Applications

The Poisson distribution is widely used in:

- Modeling the number of arrivals at a queue (e.g., calls at a call center)
- Counting occurrences over time (e.g., number of emails received per hour)
- Biology (e.g., distribution of mutations in a DNA strand)
- Traffic flow analysis (e.g., number of cars passing through an intersection)

This distribution is essential for understanding and predicting rare events in real-world scenarios.
