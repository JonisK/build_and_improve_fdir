import logging

from base import check_useful_action, find_successors


def find_useful_actions(statistics, state):
    a_list = []
    for a in statistics["all_actions"]:
        if check_useful_action(statistics, state, a):
            a_list.append(a)
    return a_list


# def find_actions_to_add(statistics, useful_configurations, list_state, available_actions):
#     configurations_to_consider = {}
#     for mode in useful_configurations:
#         configuration_mode = useful_configurations[mode]
#         for equip in configuration_mode:
#             configuration = configuration_mode[equip]
#
#             # if num_of_vectors_to_select is len(configuration[1]):
#             #     configurations_to_consider[mode][equip] =
#             #
#             # elif num_of_vectors_to_select < len(configuration[1]):
#             #     list_of = find_best_options_for_or_node(configuration[1], list_state, available_actions)
#

# def compute_useful_configurations(configuration_all_modes, list_state):
#     useful_configuration = {}
#     for mode in configuration_all_modes:
#         useful_configuration[mode] = {}
#         configuration_mode = configuration_all_modes[mode]
#         for equip in configuration_mode:
#             configuration = configuration_mode[equip]
#             num_of_vectors_to_select = configuration[0]
#             if num_of_vectors_to_select is len(configuration[1]):
#                 useful_configuration[mode][equip] = configuration
#             elif num_of_vectors_to_select < len(configuration[1]):
#                 useful_configuration[mode][equip] = (num_of_vectors_to_select, [])
#                 for i in range(len(configuration[1])):
#                     vector_useful = False
#                     for entry in configuration[1][i]:
#                         if list_state[entry] == 1:
#                             vector_useful = True
#                     if vector_useful:
#                         for j in range(len(configuration[1])):
#                             if i is not j and configuration[1][j] not in useful_configuration[mode][equip][1]:
#                                 useful_configuration[mode][equip][1].append(configuration[1][j])
#     return useful_configuration
#
# def compute_all_configurations_or_nodes(configurations, num_of_vectors_to_select):
#     if num_of_vectors_to_select == 1:
#         return configurations
#
#     all_configurations = []
#     for i in range(len(configurations)):
#         new_configuration = [configurations[i]]
#         all_small_configurations = compute_all_configurations_or_nodes(configurations[:i] + configurations[i + 1:],
#                                                                        num_of_vectors_to_select - 1)
#         for config in all_small_configurations:
#             new_configuration_copy = new_configuration
#             new_configuration_copy.append(config)
#             all_configurations.append(new_configuration_copy)
#     return all_configurations
#
#
# def is_equal_configuration(config1, config2, list_state):
#     for i in range(len(config1)):
#         if list_state[config1[i]] == 1:
#             if config1[i] not in config2:
#                 return False
#     return True
#
#
# def transform_configurations(configuration_all_modes, list_state):
#     useful_configuration = {}
#     for mode in configuration_all_modes:
#         useful_configuration[mode] = {}
#         configuration_mode = configuration_all_modes[mode]
#         for equip in configuration_mode:
#             configuration = configuration_mode[equip]
#             num_of_vectors_to_select = configuration[0]
#             if num_of_vectors_to_select != 1:
#                 all_configurations_or_nodes = compute_all_configurations_or_nodes(configuration[1],
#                                                                                   num_of_vectors_to_select)
#                 configurations_to_add = []
#                 for configuration_or in all_configurations_or_nodes:
#                     configuration_added = False
#                     for added_configuration in configurations_to_add:
#                         if is_equal_configuration(configuration_or, added_configuration, list_state):
#                             configuration_added = True
#                     if not configuration_added:
#                         configurations_to_add.append(configuration_or)
#                 useful_configuration[mode][equip] = configurations_to_add
#             else:
#                 useful_configuration[mode][equip] = configuration[1]
#     return useful_configuration


# def actions_to_add(statistics, state, available_actions):
#     configuration_all_modes = statistics["configuration_all_modes"]
#     list_state = int_to_list(statistics, state)
#     useful_configurations = transform_configurations(configuration_all_modes, list_state)
#     return find_actions_to_add(statistics, useful_configurations, list_state)

def actions_to_add(statistics, state, available_actions):
    useful_actions = []
    successors1_useful_actions = []
    for action in available_actions:
        successor1, _ = find_successors(statistics, state, action)
        found = False
        for i in range(len(useful_actions)):
            if successor1 is successors1_useful_actions[i]:
                found = True
        if not found:
            useful_actions.append(action)
            successors1_useful_actions.append(successor1)
    return useful_actions


def add_state(mcts_graph, mcts_data, statistics, state):
    if mcts_data.get(state) is None:
        mcts_graph.add_node(state)
        mcts_data[state] = [0, 0]
        if statistics["available_actions"].get(state) is None:
            statistics["available_actions"][state] = find_useful_actions(statistics, state)


def add_edge(mcts_graph, node1, node2):
    mcts_graph.add_edge(node1, node2)


def mcts_expand(mcts_graph, mcts_stats, statistics, parameters, state):
    available_actions = statistics["available_actions"][state]
    actions = actions_to_add(statistics, state, available_actions)
    for action in actions:
        if check_useful_action(statistics, state, action):
            successor1, successor2 = find_successors(statistics, state, action)
            add_state(mcts_graph, mcts_stats, statistics, successor1)
            add_state(mcts_graph, mcts_stats, statistics, successor2)
            mcts_graph.add_edge(state, successor1)
            mcts_graph.add_edge(state, successor2)

    if parameters["debug"]:
        logging.debug("States added while expanding: " + ' '.join(map(str, mcts_graph.succ[state])))

    return len(mcts_graph.succ[state])
