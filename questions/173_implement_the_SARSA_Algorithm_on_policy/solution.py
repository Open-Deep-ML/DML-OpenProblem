from collections import defaultdict
def select_greedy_action(state,action_after_state,Q):
    actions = action_after_state.get(state,[])
    if not actions:
        return None
    else:
        max_q = max(Q[(state,a)] for a in actions)
        action_required = []
        for a in actions:
            if(Q[(state,a)] == max_q):
                action_required.append(a)
        final_action = min(action_required)
        return final_action
def sarsa_update(transitions, initial_states, alpha, gamma, max_steps):
    Q = defaultdict(float)
    action_after_state = defaultdict(set)
    for (s,a) in transitions:
        action_after_state[s].add(a)

    for state in initial_states:
        steps = 0
        s = state
        action = select_greedy_action(s,action_after_state,Q)
        while s!="terminal" and steps<max_steps:
            reward,next_state = transitions[(s,action)]
            steps+=1
            if next_state == "terminal":
                action_next = None
                next_q = 0
            else:
                action_next = select_greedy_action(next_state,action_after_state,Q)
                next_q = Q[next_state,action_next]

            Q[(s,action)] += alpha*(reward+ gamma*next_q- Q[(s,action)])
            s = next_state
            action = action_next

    return Q