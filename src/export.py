import re

import networkx as nx

from evaluate_mcts_strategy import pick_best_available_action
from base import remove_unnecessary_nodes, int_to_list, find_successors, find_successor_prob, get_action_name, \
    no_possible_successors, get_cost, list_to_int


def export_mcts_strategy(graph, data, statistics, parameters):
    remove_unnecessary_nodes(graph)

    print("Exporting strategy to file:", parameters["strategy_file"])
    f = open(parameters["strategy_file"], "w")

    for node in graph.nodes:
        if node == 0:
            continue
        action = pick_best_available_action(data, statistics, node)
        if action != 0:
            f.write(str(node) + " : " + str(action) + "\n")
    f.close()


def export_graph(mcts_graph, filename):
    print("Exporting graph to file:", filename)
    nx.drawing.nx_pydot.write_dot(mcts_graph, filename)


def vector_to_string(statistics, state, all_equip):
    state_list = int_to_list(statistics, state)
    # all_equip = get_all_equipments()
    state_string = ""
    for i in range(len(state_list)):
        state_string += "("
        state_string += all_equip[i]
        if state_list[i] == 0:
            state_string += "=0)"
        elif state_list[i] == 1:
            state_string += "=1)"
        if i != len(state_list) - 1:
            state_string += " & "
    return state_string


def vector_to_primed_string(statistics, state, all_equip):
    state_list = int_to_list(statistics, state)
    # all_equip = get_all_equipments()
    state_string = ""
    for i in range(len(state_list)):
        state_string += "("
        state_string += all_equip[i]
        if state_list[i] == 0:
            state_string += "'=0)"
        elif state_list[i] == 1:
            state_string += "'=1)"
        if i < len(state_list) - 1:
            state_string += " & "
    return state_string


def get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, state):
    if state_to_prism_state_mapping.get(state) is None:
        prism_state = len(state_to_prism_state_mapping)
        state_to_prism_state_mapping[state] = prism_state
        prism_state_to_state_mapping[prism_state] = state
    else:
        prism_state = state_to_prism_state_mapping[state]
    return prism_state


def get_state_from_file(statistics, filename):
    f = open(filename, "r")
    text = f.read()
    x = re.search(r"\[(.*)]", text)
    state = x.group(1).split(", ")
    for i in range(len(state)):
        if state[i] == "1":
            state[i] = 1
        elif state[i] == "0":
            state[i] = 0
    return list_to_int(statistics, state)


def export_prism_file(mcts_graph, parameters, statistics, filename):
    # initial stuff
    model_file = open(filename + ".prism", "w")
    model_file.write("mdp\n\n")
    model_file.write("module mcts\n\n")
    state_to_prism_state_mapping = {}
    prism_state_to_state_mapping = {}

    # add variables
    model_file.write("\ts: [0.." + str(len(mcts_graph.nodes)-1) + "];\n\n")
    # for equip in get_all_equipments():
    #     model_file.write("\t" + equip + ": [0..1];\n")
    # model_file.write("\n")

    # and transitions and compute the actions which are used in the graph
    used_actions = []
    for state in mcts_graph.nodes:
        if state == 0:
            continue
        prism_state = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, state)
        actions = statistics["available_actions"][state]
        for action in actions:
            if action not in used_actions:
                used_actions.append(action)
            successor1, successor2 = find_successors(statistics, state, action)
            prism_successor1 = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, successor1)
            prism_successor2 = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, successor2)
            prob1, prob2 = find_successor_prob(statistics, state, action)
            action_name = get_action_name(statistics, action)
            model_file.write("\t[" + action_name + "] (s=" + str(prism_state) + ") -> " + str(prob1) + ":(s'=" +
                             str(prism_successor1) + ") + " + str(prob2) + ":(s'=" + str(prism_successor2) + ");\n")
    model_file.write("endmodule\n")

    # label
    leaf_nodes = []
    for state in mcts_graph.nodes:
        if state != 0 and no_possible_successors(statistics, state):
            leaf_nodes.append(state)
    model_file.write("\nlabel \"final\" = ")
    for state in leaf_nodes[:-1]:
        prism_state = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, state)
        model_file.write("(s=" + str(prism_state) + ") | ")
    prism_state = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, leaf_nodes[-1])
    model_file.write("(s=" + str(prism_state) + ");\n")

    # rewards
    model_file.write("\nrewards \"cost\"\n")
    for action in used_actions:
        action_name = get_action_name(statistics, action)
        action_cost = get_cost(statistics, action)
        model_file.write("\t[" + action_name + "] true : " + str(action_cost) + ";\n")
    model_file.write("endrewards\n")

    # init states
    model_file.write("\ninit\n\t")
    if parameters["initial_state_file"] != "":
        state = get_state_from_file(statistics, parameters["initial_state_file"])
        prism_state = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, state)
        model_file.write("(s=" + str(prism_state) + ") | ")
    else:
        for state in statistics["all_actions"][:-1]:
            prism_state = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, state)
            model_file.write("(s=" + str(prism_state) + ") | ")
    prism_state = get_prism_state(state_to_prism_state_mapping, prism_state_to_state_mapping, statistics["all_actions"][-1])
    model_file.write("(s=" + str(prism_state) + ")\n")
    model_file.write("endinit")
    model_file.close()

    # property
    prop_file = open(filename + ".props", "w")
    prop_file.write("Rmin=? [ F \"final\" ]")
    prop_file.close()

    return prism_state_to_state_mapping
