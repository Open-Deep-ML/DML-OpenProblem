# TD(0) Policy Evaluation Algorithm

**Input**: the policy $\pi$ to be evaluated  
**Algorithm parameter**: step size $( \alpha \in (0, 1] ) $

**Initialize** $V(s)$, for all $( s \in \mathcal{S}^+ )$, arbitrarily except that $ V(\text{terminal}) = 0 $

## Loop for each episode:
1. Initialize state $( S )$
2. **Loop for each step of the episode**:
   - $( A \leftarrow \pi(S) ) $ action given by the $\pi$ for $S$
   - Take action $A$, observe reward $R$, next state $S'$
   - Update value:
     $
     V(S) \leftarrow V(S) + \alpha \left[ R + V(S') - V(S) \right]
     $
   - $S \leftarrow S' $
3. **Until** $S$ is terminal

    