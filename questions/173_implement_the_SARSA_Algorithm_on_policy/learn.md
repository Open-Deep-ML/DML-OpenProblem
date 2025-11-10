# SARSA: On-Policy TD Control

**Goal**: Estimate the action-value function $Q^\pi \approx q^*$ using the SARSA algorithm (on-policy Temporal-Difference control).

## Parameters
- Step size $\alpha \in (0, 1]$
- Discount factor $\gamma \in [0, 1]$

## Initialization
- Initialize $Q(s, a)$ arbitrarily for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$  
- Set $Q(\text{terminal}, \cdot) = 0$

## Algorithm

**Loop for each episode:**
1. Initialize state $S$
2. Choose action $A$ from $S$ using a policy derived from $Q$ (e.g., greedy)

    **Loop for each step of the episode:**
    1. Take action $A$, observe reward $R$ and next state $S'$
    2. Choose next action $A'$ from $S'$ using a policy derived from $Q$ (e.g., greedy)
    3. Update the action-value:
       $
       Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]
       $
    4. Set $S \leftarrow S'$, $A \leftarrow A'$
    5. Repeat until $S$ is terminal

This algorithm continuously improves the policy as it explores and learns from interaction, making it suitable for online reinforcement learning scenarios.

    