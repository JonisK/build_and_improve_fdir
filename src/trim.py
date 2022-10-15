import logging

import networkx as nx

from expand import add_edge
from selec import compute_expected_cost_of_action
from base import find_successors


def delete_edges(mcts_graph, statistics, state, action):
    successors = find_successors(statistics, state, action)
    try:
        mcts_graph.remove_edge(state, successors[0])
    except nx.exception.NetworkXError:
        pass
    try:
        mcts_graph.remove_edge(state, successors[1])
    except nx.exception.NetworkXError:
        pass

    return mcts_graph


def find_max_cost(list_cost):
    max_cost = list_cost[0]
    max_cost_index = 0
    length = len(list_cost)
    for i in range(length):
        if list_cost[i] > max_cost:
            max_cost = list_cost[i]
            max_cost_index = i
    return max_cost, max_cost_index


def find_best_k_action_indices(list_costs, k):
    best_costs = list_costs[0:k]
    best_indices = list(range(k))
    for i in range(k, len(list_costs)):
        max_cost, max_cost_index = find_max_cost(best_costs)
        if max_cost > list_costs[i]:
            best_costs[max_cost_index] = list_costs[i]
            best_indices[max_cost_index] = i
    return best_indices


def find_best_k_actions(mcts_data, statistics, parameters, state):
    available_actions = statistics["available_actions"][state]
    available_actions_cost = []

    for action in available_actions:
        available_actions_cost.append(
            compute_expected_cost_of_action(mcts_data, statistics, state, action))

    best_action_indices = find_best_k_action_indices(available_actions_cost, parameters["successors_to_keep"])

    best_k_actions = []
    for i in best_action_indices:
        best_k_actions.append(available_actions[i])

    return best_k_actions


def mcts_trim(mcts_graph, mcts_data, statistics, parameters, state):
    available_actions = statistics["available_actions"][state]

    if len(available_actions) < parameters["successors_to_keep"] or parameters["successors_to_keep"] == 0:
        best_actions = available_actions
        for action in best_actions:
            successor1, successor2 = find_successors(statistics, state, action)
            if successor1 not in statistics["nodes_to_explore"] and successor1 not in statistics["nodes_explored"]:
                statistics["nodes_to_explore"].append(successor1)
            if successor2 not in statistics["nodes_to_explore"] and successor2 not in statistics["nodes_explored"]:
                statistics["nodes_to_explore"].append(successor2)
    else:
        available_action_costs = []
        for action in available_actions:
            cost = compute_expected_cost_of_action(mcts_data, statistics, state, action)
            available_action_costs.append(cost)

        if parameters["debug"]:
            logging.debug("Actions and their computed costs:")
            temp = ""
            for x, y in zip(available_actions, available_action_costs):
                temp += str(x) + " : " + str(round(y, 1)) + ","
            logging.debug(temp)
        best_actions = find_best_k_actions(mcts_data, statistics, parameters, state)
        if parameters["debug"]:
            logging.debug("Best actions: " + ' '.join(map(str, best_actions)) + "\n")

        for action in available_actions:
            if action not in best_actions:
                delete_edges(mcts_graph, statistics, state, action)

        for action in best_actions:
            successor1, successor2 = find_successors(statistics, state, action)
            add_edge(mcts_graph, state, successor1)
            add_edge(mcts_graph, state, successor2)
            if successor1 not in statistics["nodes_to_explore"] and successor1 not in statistics["nodes_explored"]:
                statistics["nodes_to_explore"].append(successor1)
            if successor2 not in statistics["nodes_to_explore"] and successor2 not in statistics["nodes_explored"]:
                statistics["nodes_to_explore"].append(successor2)


    statistics["available_actions"][state] = best_actions
    return mcts_graph
