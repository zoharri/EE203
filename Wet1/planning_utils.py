def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    curr_prev = prev[goal_state.to_string()]
    result = [(goal_state, curr_prev)]
    while curr_prev:
        curr = curr_prev.copy()
        curr_prev = prev[curr.to_string()]
        result.append((curr, curr_prev))
    return result

def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
