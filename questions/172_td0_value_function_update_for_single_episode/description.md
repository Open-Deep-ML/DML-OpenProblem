Implement the **TD(0) policy evaluation update** for a single episode under a given deterministic policy.  
The episode is a list of  $ (state, action, reward, nextstate) $ transitions that are all consistent with the provided policy $\pi$.

Use the TD(0) update rule to compute a **single pass of value updates** for each state in the episode.

Assume **discounting factor** to be $1$

**Constraint**:  
All transitions in the episode **must adhere to the given policy** $\pi $ , i.e., the action taken in each transition must match $\pi(\text{state})$