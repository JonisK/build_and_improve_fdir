import os
import re
import time

from tqdm import tqdm
from evaluate_mcts_strategy import sample_initial_state, sample_a_defect, export_weakness_report
from simulations import simulate_one_step_for_defect
from base import get_cost, no_possible_successors, get_action_from_string, get_action_name, int_to_list


def simulate_a_path(statistics, strategy, defect):
    state = sample_initial_state(statistics, defect)
    init_state = state
    acc_cost = 0
    while len(statistics["available_actions"][state]) != 0:
        action = strategy[state]
        state = simulate_one_step_for_defect(statistics, state, action, defect)
        acc_cost += get_cost(statistics, action)
    if no_possible_successors(statistics, state):
        return acc_cost, init_state
    # fail-safe to check if the exploration is complete
    print("Error: Strategy not complete for state: ", state)
    return acc_cost, init_state


# noinspection DuplicatedCode
def evaluate_strategy(statistics, strategy):
    max_num_simulations = 100000
    total_cost = 0
    defects = []
    result = {}
    print("\nStarting evaluation of prism strategy...")
    time.sleep(0.01)
    for _ in tqdm(range(max_num_simulations)):
        defect = sample_a_defect(statistics)
        defects.append(defect)
        cost, mode = simulate_a_path(statistics, strategy, defect)
        if result.get(mode) is not None:
            result[mode][0] += cost
            result[mode][1] += 1
        else:
            result[mode] = [cost, 1]
        total_cost += cost
    print("done")
    print("Average cost for", max_num_simulations, "faults:", total_cost / max_num_simulations)
    return result


def export_state_values(statistics, prism_state_to_state_mapping):
    f = open("./temp/prism_strategy_states.prism", 'w')
    f.write("(")
    for i in range(len(statistics["all_equipments"])):
        f.write(statistics["all_equipments"][i])
        if i < len(statistics["all_equipments"])-1:
            f.write(",")
    f.write(")\n")

    for key in prism_state_to_state_mapping:
        f.write(str(key))
        f.write(":(")
        state = prism_state_to_state_mapping[key]
        state_vector = int_to_list(statistics, state)
        for i in range(len(state_vector)):
            f.write(str(state_vector[i]))
            if i < len(state_vector)-1:
                f.write(",")
        f.write(")\n")
    f.close()


def evaluate_prism_strategy(statistics, prism_state_to_state_mapping):
    command = './prism/bin/prism ./temp/model.prism ./temp/model.props -prop 1 -explicit -exportstrat ' \
              './temp/prism_strategy.prism -exportstates ./temp/prism_strategy_states.prism -javamaxmem 4g > ' \
              './temp/prism_output.txt '
    os.system(command)
    state_file = open("./temp/prism_strategy_states.prism", "r")
    state_file.readline()
    output_state_to_prism_state = {}
    for line in state_file:
        x = re.search(r"(\d*):\((\d*)\)", line)
        output_state_to_prism_state[int(x.groups()[0])] = int(x.groups()[1])
    state_file.close()

    strategy_file = open("./temp/prism_strategy.prism", "r")
    strategy = {}
    for line in strategy_file:
        x = re.search(r"(\d*):(.*)", line)
        output_state = int(x.groups()[0])
        prism_state = output_state_to_prism_state[output_state]
        state = prism_state_to_state_mapping[prism_state]
        action_string = x.groups()[1]
        action = get_action_from_string(statistics, action_string)
        strategy[state] = action
    strategy_file.close()
    export_state_values(statistics, prism_state_to_state_mapping)

    result = evaluate_strategy(statistics, strategy)
    export_weakness_report(statistics, result)
