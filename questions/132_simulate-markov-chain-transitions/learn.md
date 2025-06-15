## Markov Chains: A Stochastic Process

Markov Chains are a fundamental concept in probability theory, used to model systems that transition between different states over time. In this explanation, we will explore the key ideas behind Markov Chains, focusing on their mathematical foundations and intuitive meaning.

### 1. Definition of a Markov Chain

A Markov Chain is a sequence of events or states where the probability of moving to the next state depends only on the current state, not on any previous states. This property is known as the "memoryless" or Markov property.

To illustrate, imagine a system with a set of possible states, such as weather conditions (e.g., sunny or rainy). At any given moment, the system occupies one state, and the likelihood of transitioning to another state is determined solely by the current one. Mathematically, if we denote the states as $S_1, S_2, \dots, S_n$, the process evolves according to probabilities that satisfy the Markov property.

For example, the equation for the probability of being in state $S_j$ at time $t+1$, given the current state $S_i$ at time $t$, can be expressed as:

$
P(S_{t+1} = S_j \mid S_t = S_i, S_{t-1}, \dots) = P(S_{t+1} = S_j \mid S_t = S_i)
$

Here, $P$ represents probability, $S_t$ is the state at time $t$, and the right side shows that only the current state $S_t$ matters. This equation highlights how the process simplifies decision-making by ignoring historical data, making it useful for modeling random phenomena like random walks or population dynamics.

### 2. Transition Probabilities and the Matrix

At the heart of a Markov Chain is the concept of transition probabilities, which quantify the likelihood of moving from one state to another. These probabilities are organized into a structure called a transition matrix.

A transition matrix is a square array where each entry represents the probability of transitioning from a specific row state to a specific column state. For a system with $n$ states, the matrix is an $n \times n$ grid, and each row sums to 1, ensuring that the probabilities for all possible outcomes from a given state add up to certainty.

The general form of a transition matrix $P$ is:

$
P = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{pmatrix}
$

In this matrix, $p_{ij}$ is the probability of transitioning from state $i$ to state $j$. For instance, $p_{11}$ might represent the probability of staying in state 1, while $p_{12}$ represents the probability of moving from state 1 to state 2. Each $p_{ij}$ is a value between 0 and 1, and the sum of each row equals 1, as expressed by:

$
\sum_{j=1}^{n} p_{ij} = 1 \quad \text{for each row } i
$

This equation ensures that the matrix reflects a complete set of possibilities for each starting state, providing a clear framework for predicting future behavior based on the current position.

### 3. Evolution of States Over Time

Once the transition matrix is defined, we can describe how the states of a Markov Chain evolve through successive steps. Starting from an initial state, the process generates a sequence of states by applying the transition probabilities repeatedly.

At each step, the next state is determined by the probabilities associated with the current state. Over multiple steps, this leads to a sequence that can be analyzed to understand long-term patterns, such as whether the system tends to settle into certain states or remains unpredictable.

Mathematically, if we begin in state $i$ at time 0, the probability of being in state $j$ after one step is given by the entry $p_{ij}$ in the matrix. After multiple steps, the overall probability distribution can be computed by multiplying the initial state probabilities by the transition matrix raised to the power of the number of steps. For a probability vector $\mathbf{v}_t$ representing the likelihood of being in each state at time $t$, the evolution is:

$
\mathbf{v}_{t+1} = \mathbf{v}_t \cdot P
$

Here, $\mathbf{v}_t$ is a row vector of probabilities summing to 1, and $P$ is the transition matrix. This operation shows how the distribution shifts over time, with each multiplication reflecting the application of transition rules. In the long run, many Markov Chains reach a steady-state distribution, where the probabilities no longer change, offering insights into stable behaviors of the system.

---

### Example Walkthrough

To make the concept more concrete, consider a simple two-state system modeling weather patterns: State 1 as "Sunny" and State 2 as "Rainy." Suppose the transition matrix is:

$
P = \begin{pmatrix}
0.7 & 0.3 \\
0.4 & 0.6
\end{pmatrix}
$

In this matrix, the entry 0.7 means there is a 70% chance of staying Sunny if it is currently Sunny, while 0.3 means a 30% chance of becoming Rainy. Similarly, 0.4 indicates a 40% chance of becoming Sunny if it is currently Rainy, and 0.6 means a 60% chance of staying Rainy.

Starting from State 1 (Sunny), after one step, there is a 70% chance of remaining Sunny and a 30% chance of moving to Rainy. If it becomes Rainy, the next step would follow the second row of the matrix. Over several steps, this process might fluctuate, but eventually, it could approach a balance where the probabilities stabilize, reflecting typical weather patterns in this model. This example demonstrates how transition probabilities guide the system's behavior in a predictable yet random manner.
