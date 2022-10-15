import random

from base import check_useful_action, find_successors, find_successor_prob, get_cost, is_final_state, \
    no_possible_successors


def compute_cost(mcts_stats, statistics, state):
    if is_final_state(state):
        return 0
    elif no_possible_successors(statistics, state):
        return 0
    total_cost = mcts_stats[state][0]
    num_visited = mcts_stats[state][1]
    if num_visited == 0 or statistics["total_simulations"] == 0:
        return -50
    # cost = total_cost/num_visited + ucb_const * math.sqrt(math.log(total_sim)/num_visited)
    cost = total_cost / num_visited
    return cost


def compute_expected_cost_of_action(mcts_stats, statistics, state, action):
    successors = find_successors(statistics, state, action)
    prob = find_successor_prob(statistics, state, action)
    cost = get_cost(statistics, action)
    cost += (prob[0] * compute_cost(mcts_stats, statistics, successors[0]) +
             prob[1] * compute_cost(mcts_stats, statistics, successors[1]))
    return cost


def simulate_one_step(statistics, state, action):
    rand = random.random()
    successors = find_successors(statistics, state, action)
    successor_prob = find_successor_prob(statistics, state, action)
    if rand < successor_prob[0]:
        return successors[0]
    else:
        return successors[1]


def get_useful_actions(statistics, state):
    if statistics["available_actions"].get(state) is None:
        useful_actions = []
        for action in statistics["all_actions"]:
            if check_useful_action(statistics, state, action):
                useful_actions.append(action)
        statistics["available_actions"][state] = useful_actions
        return useful_actions
    else:
        return statistics["available_actions"][state]


def pick_random_action(statistics, state):
    useful_actions = get_useful_actions(statistics, state)
    i = random.randrange(len(useful_actions))
    return useful_actions[i]


def compute_action(statistics, from_state, to_state):
    # avail_actions = graph.nodes[from_state]['actions']
    avail_actions = statistics["available_actions"][from_state]
    for action in avail_actions:
        successor1, successor2 = find_successors(statistics, from_state, action)
        if successor1 == to_state or successor2 == to_state:
            return action
    print("Action not available ")
    return 0


def dfs(graph, statistics, from_state, to_state, path, action_path, visited):
    if from_state == to_state:
        return path, action_path
    successors = list(graph.succ[from_state])
    for successor in successors:
        if successor in visited:
            continue
        visited.append(successor)
        action = compute_action(statistics, from_state, successor)
        new_path, new_action_path = dfs(graph, statistics, successor, to_state, path + [successor],
                                        action_path + [action], visited)
        if new_path:
            return new_path, new_action_path
    return [], []


def compute_state_action_paths(mcts_graph, statistics, from_state, to_state):
    path = [from_state]
    action_path = []
    visited = [from_state]
    return dfs(mcts_graph, statistics, from_state, to_state, path, action_path, visited)


def mcts_select(mcts_graph, statistics, init_node):
    i = random.randrange(len(statistics["nodes_to_explore"]))
    selected_state = statistics["nodes_to_explore"][i]
    path, action_path = compute_state_action_paths(mcts_graph, statistics, init_node, selected_state)
    return selected_state, path, action_path
