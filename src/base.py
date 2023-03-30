import math

import networkx as nx
import pydot

from graph_analysis.graph_analysis import create_graph_list, get_layers, \
    get_node_name, find_root_nodes, find_leaf_nodes, find_isolated_nodes


def get_configuration_all_modes(statistics, parameters):
    # configuration_path_trimmed = configuration_path.split(".")[0]
    graphs = pydot.graph_from_dot_file(parameters["input_file"])
    graph = graphs[0]
    dependency_graph = nx.DiGraph(nx.nx_pydot.from_pydot(graph))
    if len(find_isolated_nodes(dependency_graph)) > 0:
        for node in find_isolated_nodes(dependency_graph):
            dependency_graph.remove_node(node)

    layers = get_layers(dependency_graph)
    all_equipment = sorted(find_leaf_nodes(dependency_graph, layers))
    all_equipment_names = sorted([get_node_name(dependency_graph, n) for n in all_equipment])
    all_modes = [get_node_name(dependency_graph, n) for n in find_root_nodes(dependency_graph)]
    statistics["all_equipments"] = all_equipment_names
    statistics["number_of_equipments"] = len(all_equipment_names)
    statistics["all_modes"] = all_modes

    all_actions, all_list_actions, all_actions_cost, action_to_name_mapping, \
        name_to_action_mapping = get_all_actions(dependency_graph,
                                                 all_equipment_names,
                                                 statistics)
    # return configuration_all_modes, all_modes, all_equipment_names
    statistics["all_actions"] = all_actions
    statistics["all_list_actions"] = all_list_actions
    statistics["all_actions_cost"] = all_actions_cost
    statistics["name_to_action_mapping"] = name_to_action_mapping
    statistics["action_to_name_mapping"] = action_to_name_mapping


def get_all_actions(dependency_graph, all_equipment, statistics):
    threading = True
    unique_graph_list, unique_node_lists, leaf_name_lists, \
        configuration_list, configuration_space = \
        create_graph_list(dependency_graph, threading)

    all_list_actions = []
    all_actions = []
    all_actions_cost = {}
    action_to_name_mapping = {}
    name_to_action_mapping = {}
    for m in leaf_name_lists:
        all_perm = leaf_name_lists[m]
        for i in range(len(all_perm)):
            act = all_perm[i]
            action_vector = []
            for equip in all_equipment:
                if equip in act:
                    action_vector.append(1)
                else:
                    action_vector.append(0)
            if action_vector not in all_list_actions:
                all_list_actions.append(action_vector)
                action = list_to_int(statistics, action_vector)
                all_actions_cost[action] = statistics["mode_costs"][get_node_name(dependency_graph, m)]
                all_actions.append(action)
                action_to_name_mapping[action] = get_node_name(dependency_graph, m) + "_" + str(i)
                name_to_action_mapping[get_node_name(dependency_graph, m) + "_" + str(i)] = action
    return all_actions, all_list_actions, all_actions_cost, action_to_name_mapping, name_to_action_mapping


def list_to_int(statistics, mylist):
    my_int = 0
    for i in range(statistics["number_of_equipments"]):
        my_int += mylist[i] * math.pow(2, (statistics["number_of_equipments"] - 1 - i))
    return int(my_int)


def int_to_list(statistics, my_int):
    my_int_copy = my_int
    if statistics["int_to_list_mapping"].get(my_int) is None:
        mylist = []
        for i in range(statistics["number_of_equipments"] - 1, -1, -1):
            if my_int >= math.pow(2, i):
                mylist.append(1)
                my_int -= math.pow(2, i)
            else:
                mylist.append(0)
        statistics["int_to_list_mapping"][my_int_copy] = mylist
        return mylist
    else:
        return statistics["int_to_list_mapping"][my_int]


def find_successor_prob(statistics, state, action):
    state_vector = int_to_list(statistics, state)
    action_vector = int_to_list(statistics, action)
    prob1 = 0
    prob2 = 0
    for i in range(len(state_vector)):
        if state_vector[i] == 1:
            if action_vector[i] == 0:  # mode works fine
                prob1 += statistics["equipment_fail_probabilities"][i]
            elif action_vector[i] == 1:  # mode doesn't work
                prob2 += statistics["equipment_fail_probabilities"][i]
    if prob1 + prob2 == 0:
        return 0, 0
    prob1 = prob1 / (prob1 + prob2)
    prob2 = 1 - prob1
    return prob1, prob2


def find_successors(statistics, state, action):
    state_vector = int_to_list(statistics, state)
    action_vector = int_to_list(statistics, action)
    successor1 = list(state_vector)  # action works
    successor2 = list(state_vector)  # action doesn't work
    for i in range(len(state_vector)):
        if state_vector[i] == 1:
            if action_vector[i] == 0:  # mode works fine
                successor2[i] = 0
            elif action_vector[i] == 1:  # mode doesn't work
                successor1[i] = 0
    return list_to_int(statistics, successor1), list_to_int(statistics, successor2)


def check_useful_action(statistics, state, action):
    state_vector = int_to_list(statistics, state)
    action_vector = int_to_list(statistics, action)
    prob1 = 0
    prob2 = 0
    for i in range(len(state_vector)):
        if state_vector[i] == 1:
            if action_vector[i] == 0:  # mode works fine
                prob2 += 1
            elif action_vector[i] == 1:  # mode doesn't work
                prob1 += 1
    if prob1 == 0 or prob2 == 0:
        return False
    return True


def remove_unnecessary_nodes(graph):
    remove = []
    for node in graph.nodes():
        if graph.in_degree(node) == 0 and node != 0:
            remove.append(node)
    for node in remove:
        graph.remove_node(node)
    return


def is_final_state(state):
    n = state
    while n > 1:
        if n % 2 == 0:
            n = n / 2
        else:
            return False
    return True


def no_possible_successors(statistics, state):
    for action in statistics["available_actions"][state]:
        if check_useful_action(statistics, state, action):
            return False
    return True


def get_action_name(statistics, action):
    return statistics["action_to_name_mapping"][action]


def get_cost(statistics, action):
    return statistics["all_actions_cost"][action]


def get_action_from_string(statistics, action_name):
    return statistics["name_to_action_mapping"][action_name]


def get_fault_probabilities(statistics, fault_prob_dict):
    fault_probabilities = []
    for equip in statistics["all_equipments"]:
        fault_probabilities.append(fault_prob_dict[equip])
    return fault_probabilities
