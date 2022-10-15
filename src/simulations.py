import math
import random

from expand import find_useful_actions
from selec import pick_random_action, simulate_one_step
from base import get_cost, int_to_list, find_successors, no_possible_successors


def simulate_default(statistics, state):
    acc_cost = 0
    if statistics["available_actions"].get(state) is None:
        statistics["available_actions"][state] = find_useful_actions(statistics, state)
    while not no_possible_successors(statistics, state):
        # available_actions = statistics
        action = pick_random_action(statistics, state)
        state = simulate_one_step(statistics, state, action)
        if statistics["available_actions"].get(state) is None:
            statistics["available_actions"][state] = find_useful_actions(statistics, state)
        acc_cost += get_cost(statistics, action)
    return acc_cost


def update_state_costs(mcts_stats, state, cost):
    if mcts_stats[state] is not None and mcts_stats[state][0] != math.inf and mcts_stats[state][0] != -1 * math.inf:
        mcts_stats[state][0] += cost
        mcts_stats[state][1] += 1
    else:
        mcts_stats[state] = [cost, 1]
    return mcts_stats


def mcts_back_propagate(mcts_stats, statistics, path, action_path, cost):
    mcts_stats = update_state_costs(mcts_stats, path[len(path) - 1], cost)
    for i in range(len(path) - 2, -1, -1):
        cost += get_cost(statistics, action_path[i])
        mcts_stats = update_state_costs(mcts_stats, path[i], cost)
    return mcts_stats


def get_defect_distribution(statistics, state):
    state_vector = int_to_list(statistics, state)
    length = len(state_vector)
    sum_prob = 0
    for i in range(length):
        if state_vector[i] == 1:
            sum_prob += statistics["equipment_fault_probabilities"][i]
    distribution = []
    for i in range(length):
        if state_vector[i] == 1:
            distribution.append(statistics["equipment_fault_probabilities"][i] / sum_prob)
        else:
            distribution.append(0)
    return distribution


def sample_a_defect(distribution):
    rand = random.random()
    i = 0
    while distribution[i] < rand:
        rand -= distribution[i]
        i += 1
    return i


def simulate_one_step_for_defect(statistics, state, action, defect):
    successor1, successor2 = find_successors(statistics, state, action)
    successor1_list = int_to_list(statistics, successor1)
    if successor1_list[defect] == 1:
        return successor1
    else:
        return successor2


def simulate_type_one(statistics, state, defect_equip_index):
    acc_cost = 0
    if statistics["available_actions"].get(state) is None:
        statistics["available_actions"][state] = find_useful_actions(statistics, state)
    while not no_possible_successors(statistics, state):
        action = pick_random_action(statistics, state)
        state = simulate_one_step_for_defect(statistics, state, action, defect_equip_index)
        if statistics["available_actions"].get(state) is None:
            statistics["available_actions"][state] = find_useful_actions(statistics, state)
        acc_cost += get_cost(statistics, action)
    return acc_cost


def mcts_simulate(mcts_stats, statistics, parameters, state, path, action_path, max_sim):
    num_sim = 0
    i = 0
    if parameters["sampling_type"] == 0:
        while i < max_sim:
            action = pick_random_action(statistics, state)
            new_state = simulate_one_step(statistics, state, action)
            if new_state in statistics["nodes_explored"]:
                i += 1
                continue
            new_path = path + [new_state]
            new_action_path = action_path + [action]
            # new_state, new_path, new_action_path = mcts_select(mcts_graph, mcts_stats, total_sim, ucb_const, state)
            cost = simulate_default(statistics, new_state)
            mcts_stats = mcts_back_propagate(mcts_stats, statistics, new_path, new_action_path, cost)
            # mcts_stats = mcts_back_propagate(mcts_stats, path+new_path[1:], action_path+new_action_path, cost)
            num_sim += 1
            i += 1
        return num_sim
    elif parameters["sampling_type"] == 1:
        while i < max_sim:
            distribution = get_defect_distribution(statistics, state)
            defect_equip_index = sample_a_defect(distribution)
            action = pick_random_action(statistics, state)
            new_state = simulate_one_step_for_defect(statistics, state, action, defect_equip_index)
            new_path = path + [new_state]
            new_action_path = action_path + [action]
            # new_state, new_path, new_action_path = mcts_select(mcts_graph, mcts_stats, total_sim, ucb_const, state)
            cost = simulate_type_one(statistics, new_state, defect_equip_index)
            mcts_stats = mcts_back_propagate(mcts_stats, statistics, new_path, new_action_path, cost)
            # mcts_stats = mcts_back_propagate(mcts_stats, path+new_path[1:], action_path+new_action_path, cost)
            num_sim += 1
            i += 1
        return num_sim
