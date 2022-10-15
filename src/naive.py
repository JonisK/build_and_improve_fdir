import time
import random

from tqdm import tqdm

from evaluate_mcts_strategy import sample_a_defect, sample_initial_state
from base import get_cost, check_useful_action
from simulations import simulate_one_step_for_defect


# noinspection DuplicatedCode
def pick_best_available_action(statistics, state):
    # avail_actions = statistics["available_actions"][state]
    # action_values = []
    avail_actions = statistics["all_actions"]
    useful_actions = []
    for action in avail_actions:
        # successor1, successor2 = find_successors(statistics, state, action)
        # prob1, prob2 = find_successor_prob(statistics, state, action)
        # value = get_cost(statistics, action)
        # if data[successor1][1] != 0:
        #     value += prob1 * data[successor1][0] / data[successor1][1]
        # if data[successor2][1] != 0:
        #     value += prob2 * data[successor2][0] / data[successor2][1]
        # action_values.append(value)
        if check_useful_action(statistics, state, action):
            useful_actions.append(action)

    if len(useful_actions) == 0:
        return 0

    # best_value = action_values[0]
    # best_index = 0
    # for i in range(len(action_values)):
    #     if action_values[i] < best_value:
    #         best_index = i
    #         best_value = action_values[i]

    i = random.randrange(len(useful_actions))
    return useful_actions[i]


def no_possible_successors(statistics, state):
    for action in statistics["all_actions"]:
        if check_useful_action(statistics, state, action):
            return False
    return True


def simulate_a_path(statistics, defect):
    state = sample_initial_state(statistics, defect)
    # init_state = state
    acc_cost = 0
    while not no_possible_successors(statistics, state):
        action = pick_best_available_action(statistics, state)
        state = simulate_one_step_for_defect(statistics, state, action, defect)
        acc_cost += get_cost(statistics, action)
    # if no_possible_successors(statistics, state):
    return acc_cost
    # fail-safe to check if the exploration is complete
    # print("Error: Strategy not complete for state: ", state)
    # return acc_cost


# noinspection DuplicatedCode
def evaluate_naive(statistics):
    max_num_simulations = 100000
    total_cost = 0
    defects = []
    print("\nStarting evaluation of naive strategy...")
    time.sleep(0.1)
    for _ in tqdm(range(max_num_simulations)):
        defect = sample_a_defect(statistics)
        defects.append(defect)
        cost = simulate_a_path(statistics, defect)
        total_cost += cost
    print("done")
    print("Average cost for", max_num_simulations, "faults:", total_cost / max_num_simulations)
