import random
import time

from tqdm import tqdm

from simulations import simulate_one_step_for_defect
from base import find_successors, find_successor_prob, get_cost, int_to_list, no_possible_successors, get_action_name


def sample_a_defect(statistics):
    dist = statistics["equipment_fail_probabilities"]
    sum_prob = 0
    for prob in dist:
        sum_prob += prob
    for i in range(len(dist)):
        dist[i] = dist[i] / sum_prob
    rand = random.random()
    i = 0
    while rand > dist[i]:
        rand -= dist[i]
        i += 1
    return i


def pick_best_available_action(data, statistics, state):
    avail_actions = statistics["available_actions"][state]
    action_values = []
    for action in avail_actions:
        successor1, successor2 = find_successors(statistics, state, action)
        prob1, prob2 = find_successor_prob(statistics, state, action)
        value = get_cost(statistics, action)
        if data[successor1][1] != 0:
            value += prob1 * data[successor1][0] / data[successor1][1]
        if data[successor2][1] != 0:
            value += prob2 * data[successor2][0] / data[successor2][1]
        action_values.append(value)

    if len(action_values) == 0:
        return 0

    best_value = action_values[0]
    best_index = 0
    for i in range(len(action_values)):
        if action_values[i] < best_value:
            best_index = i
            best_value = action_values[i]

    return avail_actions[best_index]


def sample_initial_state(statistics, defect):
    successors = []
    for action in statistics["all_actions"]:
        action_list = int_to_list(statistics, action)
        if action_list[defect] == 1:
            successors.append(action)
    rand = random.randrange(len(successors))
    return successors[rand]


def simulate_a_path(data, statistics, defect):
    state = sample_initial_state(statistics, defect)
    init_state = state
    acc_cost = 0
    while len(statistics["available_actions"][state]) != 0:
        action = pick_best_available_action(data, statistics, state)
        state = simulate_one_step_for_defect(statistics, state, action, defect)
        acc_cost += get_cost(statistics, action)
    if no_possible_successors(statistics, state):
        return acc_cost, init_state
    # fail-safe to check if the exploration is complete
    print("Error: Strategy not complete for state: ", state)
    return acc_cost, init_state


# noinspection DuplicatedCode
def export_weakness_report(statistics, result):
    f = open("./temp/weakness_report.txt", "w")
    f.write("Mode configuration:\tAverage cost\t:\tNumber of simulations\n")
    for mode in result:
        f.write(get_action_name(statistics, mode) + ":\t" + str(result[mode][0] / result[mode][1]) + "\t:\t" + str(
            result[mode][1]) + "\n")
    f.close()


# noinspection DuplicatedCode
def evaluate_mcts_strategy(data, statistics):
    max_num_simulations = 100000
    total_cost = 0
    defects = []
    result = {}
    print("\nStarting evaluation of MCTS strategy...")
    time.sleep(0.01)
    for _ in tqdm(range(max_num_simulations)):
        defect = sample_a_defect(statistics)
        defects.append(defect)
        cost, mode = simulate_a_path(data, statistics, defect)
        if result.get(mode) is not None:
            result[mode][0] += cost
            result[mode][1] += 1
        else:
            result[mode] = [cost, 1]
        total_cost += cost
    print("done")
    print("Average cost for", max_num_simulations, "faults:", total_cost / max_num_simulations)
    export_weakness_report(statistics, result)
