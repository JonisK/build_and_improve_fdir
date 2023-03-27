import itertools
import logging
import math
import re

import networkx as nx
import pydot


# logging.basicConfig(
#     format="[%(levelname)s] %(funcName)s: %(message)s")
# logging.getLogger().setLevel(logging.DEBUG)


# logging.getLogger().setLevel(logging.INFO)

def find_root_nodes(dependency_graph):
    root_nodes = []
    for node in dependency_graph:
        if len(list(dependency_graph.predecessors(node))) == 0:
            root_nodes.append(node)
    return root_nodes


def find_root_node(dependency_graph):
    for node in dependency_graph:
        if len(list(dependency_graph.predecessors(node))) == 0:
            return node


def find_leaf_nodes(dependency_graph, layers):
    leaf_nodes = []
    for layer in layers:
        leaf_nodes_layer = []
        for node in layer:
            if not list(dependency_graph.successors(node)):
                leaf_nodes_layer.append(node)
        # convert to names, sort alphabetically, convert back to node IDs
        leaf_node_names_layer = sorted([get_node_name(dependency_graph, leaf_node) for leaf_node in leaf_nodes_layer])
        # logging.debug(f"Leaf_node_names_layer: {leaf_node_names_layer}")
        # logging.debug(f"Leaf_nodes_layer: {[get_node_id(G, leaf_node_name) for leaf_node_name in leaf_node_names_layer]}")
        leaf_nodes.extend([get_node_id(dependency_graph, leaf_node_name) for leaf_node_name in leaf_node_names_layer])
    return leaf_nodes


def find_leaf_nodes_per_root_node(dependency_graph, root_node):
    leaf_nodes = []
    subgraph = nx.bfs_tree(dependency_graph, root_node)
    for node in subgraph:
        if len(list(subgraph.successors(node))) == 0:
            leaf_nodes.append(node)
    # convert to names, sort alphabetically, convert back to node IDs
    leaf_node_names = sorted([get_node_name(dependency_graph, leaf_node) for leaf_node in leaf_nodes])
    return [get_node_id(dependency_graph, leaf_node_name) for leaf_node_name in leaf_node_names]


def get_node_name(dependency_graph, node):
    attr = dependency_graph.nodes[node]
    if 'xlabel' in attr:
        return attr['xlabel'].strip('\"')
    else:
        return node


def get_node_names(dependency_graph):
    return [get_node_name(dependency_graph, node) for node in list(dependency_graph.nodes)]


def get_root_node_names(dependency_graph):
    root_nodes = find_root_nodes(dependency_graph)
    root_node_names = {}
    for root_node in root_nodes:
        root_node_names[get_node_name(dependency_graph, root_node)] = root_node
    # logging.info(root_node_names)
    return root_node_names


def get_mode_indices(dependency_graph):
    roots = sorted([get_node_name(dependency_graph, root) for root in find_root_nodes(dependency_graph)])
    mode_indices = {}
    for index, root in enumerate(roots):
        mode_indices[root] = index
    # logging.info(mode_indices)
    return mode_indices


def get_mode_indices_appended(dependency_graph):
    mode_indices_appended = get_mode_indices(dependency_graph)
    if "off" not in mode_indices_appended.keys():
        mode_indices_appended["off"] = len(mode_indices_appended)
    return mode_indices_appended


def get_node_id(dependency_graph, name):
    for node in list(dependency_graph.nodes):
        attr = dependency_graph.nodes[node]
        if 'xlabel' in attr:
            if name == attr['xlabel'].strip('\"'):
                return node
    # logging.error(f"Could not find ID for node with name {name}. Returning passed name as an ID")
    return name


def get_dependencies(dependency_graph, root_node):
    return [get_node_name(dependency_graph, dependency) for dependency in list(dependency_graph.successors(root_node))]


def create_statespace(dependency_graph, node_list):
    import re
    # graph_list = []
    statespace = []
    for node in node_list:
        attr = dependency_graph.nodes[node]
        if 'xlabel' in attr:
            name = attr['xlabel'].strip('\"')
            if name.startswith(">="):
                available = len(list(dependency_graph.successors(node)))
                required = int(re.findall(r"\d+", name)[0])
                # the length should be equivalent to the binomial coefficient available nCr required
                statespace.append(len(list(itertools.combinations(range(available), available - required))) + 1)
            elif name.startswith("OR"):
                statespace.append(2)
    # if there are no disjunctive assemblies, there is only one state
    if not statespace:
        statespace.append(1)
    return statespace


def create_permutations(statespace):
    permutations = []
    reversed_statespace = statespace
    reversed_statespace.reverse()
    # logging.info(reversed_statespace)
    number_of_permutations = [reversed_statespace[0]]
    for index, variable in enumerate(reversed_statespace[1:]):
        number_of_permutations.append(variable * number_of_permutations[index])
    number_of_permutations.reverse()
    for i in range(max(number_of_permutations)):
        temp = i
        out_list = []
        # build a list for every state
        for j in range(len(statespace)):
            if j != (len(statespace) - 1):
                quotient = temp // number_of_permutations[j + 1]
                out_list.append(quotient)
                temp -= quotient * number_of_permutations[j + 1]
            else:  # last element
                out_list.append(temp)
        permutations.append(out_list)
    # logging.debug(permutations)
    return permutations


def check_for_upstream_dependencies(dependency_graph, nodes_to_be_deleted, root_node, verbose):
    try:
        for node in nodes_to_be_deleted:
            # logging.debug(
            #     f"[{get_node_name(G, root_node)}] Check node {node} ({get_node_name(G, node)}) that is about to be deleted") if verbose else None
            if len(list(dependency_graph.predecessors(node))) > 1:
                # logging.debug(
                #     f"[{get_node_name(G, root_node)}] Node {node} ({get_node_name(G, node)}) has {str(len(list(G.predecessors(node))))} predecessors") if verbose else None
                # logging.debug(
                #     f"[{get_node_name(G, root_node)}] Therefore, we will not delete: {list(nx.bfs_tree(G, node, reverse=False))}") if verbose else None
                for node_to_keep in list(nx.bfs_tree(dependency_graph, node, reverse=False)):
                    if node_to_keep in nodes_to_be_deleted:
                        nodes_to_be_deleted.remove(node_to_keep)
                    else:
                        logging.debug(
                            f"[{get_node_name(dependency_graph, root_node)}] Node {node} ({get_node_name(dependency_graph, node)}) is not about to be deleted. Skipping...")
            elif len(list(dependency_graph.predecessors(node))) == 0:
                logging.warning(
                    f"[{get_node_name(dependency_graph, root_node)}] That's strange. Node {node} ({get_node_name(dependency_graph, node)}) has no predecessors") if verbose else None
            # else:
            # logging.debug(
            #     f"[{get_node_name(G, root_node)}] The node has just one predecessor and will be deleted") if verbose else None
            # logging.debug(
            #     f"[{get_node_name(G, root_node)}] New state of node_to_be_deleted: {nodes_to_be_deleted}") if verbose else None
        return nodes_to_be_deleted
    except ValueError:
        logging.info(ValueError)


# create n graphs for every permutation
def create_graphs(dependency_graph, root_node, node_list, permutations, verbose):
    graph_list = []
    node_lists = []
    for permutation in permutations:
        # current_index = permutations.index(permutation)
        # all_permutations = len(permutations)
        index = 0
        new_dependency_graph = dependency_graph.copy()
        for node in dependency_graph:
            if node not in node_list:
                new_dependency_graph.remove_node(node)
        for node in node_list:
            attr = new_dependency_graph.nodes[node]

            # remove the other options
            if 'xlabel' in attr:
                name = attr['xlabel'].strip('\"')
                if name.startswith(">="):
                    available = len(list(new_dependency_graph.successors(node)))
                    required = int(re.findall(r"\d+", name)[0])
                    difference = available - required
                    if permutation[index] == len(list(itertools.combinations(range(available), available - required))):
                        # logging.debug(
                        #     f"[{get_node_name(G, root_node)}] [{current_index + 1}/{all_permutations}] For the assembly {get_node_name(G, list(G.predecessors(node))[0])}, we are in the +1 configuration where all equipment is used")
                        index += 1
                    else:
                        if difference == 1:
                            root_to_be_deleted = list(new_dependency_graph.successors(node))[permutation[index]]
                            nodes_to_be_deleted = set(nx.bfs_tree(new_dependency_graph, root_to_be_deleted, reverse=False))
                        elif difference >= 2:
                            nodes_to_be_deleted = set()
                            for redundancy in range(difference):
                                # logging.debug(
                                #     f"[{get_node_name(G, root_node)}] [{current_index + 1}/{all_permutations}] Deleting successor #{list(itertools.combinations(range(available), difference))[permutation[index]][redundancy]}") if verbose else None
                                root_to_be_deleted = list(new_dependency_graph.successors(node))[
                                    list(itertools.combinations(range(available), difference))[permutation[index]][
                                        redundancy]]
                                nodes_to_be_deleted.update(list(nx.bfs_tree(new_dependency_graph, root_to_be_deleted, reverse=False)))

                        nodes_to_be_deleted = check_for_upstream_dependencies(new_dependency_graph, list(nodes_to_be_deleted),
                                                                              root_node, verbose)
                        # logging.debug(
                        #     f"[{get_node_name(G, root_node)}] [{current_index + 1}/{all_permutations}] Nodes to be deleted after  checking for upstream dependencies: {nodes_to_be_deleted}") if verbose else None
                        for node_to_be_deleted in nodes_to_be_deleted:
                            new_dependency_graph.remove_node(node_to_be_deleted)
                        index += 1
                elif name.startswith("OR"):
                    root_to_be_deleted = list(new_dependency_graph.successors(node))[permutation[index]]
                    nodes_to_be_deleted = set(nx.bfs_tree(new_dependency_graph, root_to_be_deleted, reverse=False))

                    nodes_to_be_deleted = check_for_upstream_dependencies(new_dependency_graph, list(nodes_to_be_deleted), root_node,
                                                                          verbose)
                    for node_to_be_deleted in nodes_to_be_deleted:
                        new_dependency_graph.remove_node(node_to_be_deleted)
                    index += 1
        graph_list.append(new_dependency_graph)
        node_lists.append(list(nx.bfs_tree(new_dependency_graph, root_node, reverse=False)))
    return graph_list, node_lists


# prune duplicates
def remove_duplicates(graph_list, node_lists):
    unique_graph_list = []
    unique_node_lists = []
    counter = 0
    for graph, node_list in zip(graph_list, node_lists):
        if node_list not in unique_node_lists:
            unique_graph_list.append(graph)
            unique_node_lists.append(node_list)
        else:
            counter += 1
    # logging.info(len(unique_node_lists))
    # logging.info("[{get_node_name(G, root_node)}] Deleted " + str(counter) + " duplicate graphs") if verbose else None
    return unique_graph_list, unique_node_lists


# combine the functions above, so we can resolve the redundancies and create one graph for every combination of equipment
# that realizes the mode
def create_graph_list(dependency_graph, verbose):
    unique_graph_list = {}
    unique_node_lists = {}
    leaf_name_lists = {}
    for root_node in find_root_nodes(dependency_graph):
        # logging.info(get_node_name(G, root_node))
        node_list = list(nx.bfs_tree(dependency_graph, root_node, reverse=False))[::-1]
        if len(node_list) <= 1:
            # logging.warning(
            #     f"[{get_node_name(G, root_node)}] Node list empty for root node {root_node} ({get_node_name(G, root_node)}). Skipping...")
            break
        statespace = create_statespace(dependency_graph, node_list)
        # logging.info(f"[{get_node_name(G, root_node)}] statespace: {statespace}") if verbose else None
        permutations = create_permutations(statespace)
        graph_list, node_lists = create_graphs(dependency_graph, root_node, node_list, permutations, verbose)
        unique_graph_list[root_node], unique_node_lists[root_node] = remove_duplicates(graph_list, node_lists)
        leaf_name_lists[root_node] = []
        for graph in unique_graph_list[root_node]:
            new_dependency_graph = dependency_graph.copy()
            for node in dependency_graph:
                if node not in list(graph):
                    new_dependency_graph.remove_node(node)
            layers = get_layers(new_dependency_graph)
            leaf_name_lists[root_node].append(
                sorted([get_node_name(new_dependency_graph, node) for node in find_leaf_nodes(new_dependency_graph, layers)]))
    return unique_graph_list, unique_node_lists, leaf_name_lists


def get_configuration(dependency_graph, all_equipment):
    configuration = []
    index = 0
    while index < len(all_equipment):
        node_name = all_equipment[index]
        # find out if the component is part of an assembly
        part_of_assembly = False
        for predecessor in list(dependency_graph.predecessors(get_node_id(dependency_graph, node_name))):
            if get_node_name(dependency_graph, predecessor).startswith(">="):
                part_of_assembly = True
                available = len(list(dependency_graph.successors(predecessor)))
                required = int(re.findall(r"\d+", get_node_name(dependency_graph, predecessor))[0])
                configuration.append((required, list(range(index, index + available))))
                index += available
                break
            elif get_node_name(dependency_graph, predecessor).startswith("OR"):
                part_of_assembly = True
                configuration.append((1, list(range(index, index + 2))))
                index += 2
                break
        if not part_of_assembly:
            configuration.append((1, [index]))
            index += 1
    return configuration


def get_configuration_dict(dependency_graph, all_equipment):
    configuration = {}
    index = 0
    while index < len(all_equipment):
        node_name = all_equipment[index]
        # find out if the component is part of an assembly
        part_of_assembly = False
        for predecessor in list(dependency_graph.predecessors(get_node_id(dependency_graph, node_name))):
            if get_node_name(dependency_graph, predecessor).startswith(">="):
                part_of_assembly = True
                assembly_name = get_node_name(dependency_graph, list(dependency_graph.predecessors(predecessor))[0])
                available = len(list(dependency_graph.successors(predecessor)))
                required = int(re.findall(r"\d+", get_node_name(dependency_graph, predecessor))[0])
                configuration[assembly_name] = (required, list(range(index, index + available)))
                index += available
                break
            elif get_node_name(dependency_graph, predecessor).startswith("OR"):
                part_of_assembly = True
                assembly_name = get_node_name(dependency_graph, list(dependency_graph.predecessors(predecessor))[0])
                configuration[assembly_name] = (1, list(range(index, index + 2)))
                index += 2
                break
        if not part_of_assembly:
            configuration[node_name] = (1, [index])
            index += 1
    return configuration


def get_layers(dependency_graph):
    # go through every layer of the tree and look for OR/>= assemblies
    known_nodes = find_root_nodes(dependency_graph)
    layers = [list(set(known_nodes))]
    while True:
        candidates = []
        # next_layer = []
        # node_dict = {known_node: get_node_name(G, known_node) for known_node in known_nodes}
        # logging.debug(f"Known nodes: {node_dict}")
        for root_node in known_nodes:
            candidates.extend(list(nx.bfs_tree(dependency_graph, root_node, depth_limit=1)))
        # node_dict = {candidate: get_node_name(G, candidate) for candidate in candidates}
        # logging.debug(f"Candidates: {node_dict}")
        next_layer = candidates.copy()
        for candidate in candidates:
            if candidate in known_nodes:
                next_layer.remove(candidate)
            for predecessor in list(nx.bfs_tree(dependency_graph, candidate, reverse=True))[1:]:
                if predecessor not in known_nodes:
                    next_layer.remove(candidate)
                    break
        # node_dict = {next_layer_node: get_node_name(G, next_layer_node) for next_layer_node in next_layer}
        # logging.debug(f"Next layer: {node_dict}")
        if not next_layer:
            # logging.debug("Traversed the whole graph")
            return layers
        known_nodes.extend(list(set(next_layer)))
        layers.append(list(set(next_layer)))


def get_configuration_new(dependency_graph, layers, all_equipment):
    configuration = {}
    for layer in layers:
        for node in layer:
            # see if we are below some previous assembly
            valid_assembly = True
            for predecessor in list(nx.bfs_tree(dependency_graph, node, reverse=True))[1:]:
                if get_node_name(dependency_graph, predecessor).startswith(">=") or get_node_name(dependency_graph, predecessor).startswith("OR"):
                    # logging.warning(f"Node {node} will be skipped since it is part of a higher assembly")
                    valid_assembly = False
            if valid_assembly:
                # see if this is a single component
                if node in all_equipment:
                    configuration[get_node_name(dependency_graph, node)] = (1, [[all_equipment.index(node)]])
                elif get_node_name(dependency_graph, node).startswith(">="):
                    # logging.debug(f"Found a greater_than assembly")
                    # get the leaf nodes underneath
                    leaf_nodes = find_leaf_nodes_per_root_node(dependency_graph, node)
                    assembly_name = get_node_name(dependency_graph, list(dependency_graph.predecessors(node))[0])
                    # available = len(list(G.successors(node)))
                    required = int(re.findall(r"\d+", get_node_name(dependency_graph, node))[0])
                    configuration[assembly_name] = (
                        required, [all_equipment.index(leaf_node) for leaf_node in leaf_nodes])
                    # configuration[assembly_name] = (required, [get_node_name(G, leaf_node) for leaf_node in leaf_nodes])
                    # look if there are more assemblies underneath
                    hierarchical_assembly = False
                    for successor in dependency_graph.successors(node):
                        if successor not in all_equipment:
                            hierarchical_assembly = True
                    if hierarchical_assembly:
                        valid_configurations = []
                        for successor in dependency_graph.successors(node):
                            # logging.debug(f"Root node for successor tree: {get_node_name(G, successor)}")
                            sub_dependency_graph = dependency_graph.copy()
                            for temp_node in dependency_graph:
                                if temp_node not in list(nx.bfs_tree(dependency_graph, successor)):
                                    sub_dependency_graph.remove_node(temp_node)
                            if len(list(sub_dependency_graph)) == 1:
                                valid_configurations.append([all_equipment.index(successor)])
                            else:
                                (unique_graph_list, unique_node_lists, leaf_name_lists) = create_graph_list(sub_dependency_graph,
                                                                                                            False)
                                # logging.debug(
                                #     f"Leaf names for root {get_node_name(sub_G, successor)}: {leaf_name_lists[successor]}")
                                for leaf_name_list in leaf_name_lists[successor]:
                                    valid_configurations.append(
                                        [all_equipment.index(get_node_id(sub_dependency_graph, leaf_name)) for leaf_name in
                                         leaf_name_list])

                    if hierarchical_assembly:
                        # logging.debug(f"New configuration for assembly {assembly_name}: {valid_configurations}")
                        configuration[assembly_name] = (1, valid_configurations)
                    else:
                        # logging.debug(
                        # f"New configuration for assembly {assembly_name}: {[[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes]}")
                        configuration[assembly_name] = (
                            required, [[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes])

                elif get_node_name(dependency_graph, node).startswith("OR"):
                    # logging.debug(f"Found an OR assembly")
                    # get the leaf nodes underneath
                    leaf_nodes = find_leaf_nodes_per_root_node(dependency_graph, node)
                    assembly_name = get_node_name(dependency_graph, list(dependency_graph.predecessors(node))[0])
                    # look if there are more assemblies underneath
                    hierarchical_assembly = False
                    for successor in dependency_graph.successors(node):
                        if successor not in all_equipment:
                            hierarchical_assembly = True
                    if hierarchical_assembly:
                        valid_configurations = []
                        for successor in dependency_graph.successors(node):
                            # logging.debug(f"Root node for successor tree: {get_node_name(G, successor)}")
                            sub_dependency_graph = dependency_graph.copy()
                            for temp_node in dependency_graph:
                                if temp_node not in list(nx.bfs_tree(dependency_graph, successor)):
                                    sub_dependency_graph.remove_node(temp_node)
                            if len(list(sub_dependency_graph)) == 1:
                                valid_configurations.append([all_equipment.index(successor)])
                            else:
                                (unique_graph_list, unique_node_lists, leaf_name_lists) = create_graph_list(sub_dependency_graph,
                                                                                                            False)
                                # logging.debug(
                                # f"Leaf names for root {get_node_name(sub_G, successor)}: {leaf_name_lists[successor]}")
                                for leaf_name_list in leaf_name_lists[successor]:
                                    valid_configurations.append(
                                        [all_equipment.index(get_node_id(sub_dependency_graph, leaf_name)) for leaf_name in
                                         leaf_name_list])
                    if hierarchical_assembly:
                        # logging.debug(f"New configuration for assembly {assembly_name}: {valid_configurations}")
                        configuration[assembly_name] = (1, valid_configurations)
                    else:
                        # logging.debug(
                        #     f"New configuration for assembly {assembly_name}: {[[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes]}")
                        configuration[assembly_name] = (
                            1, [[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes])

    return configuration


def get_configuration_all_modes(statistics, parameters):
    # configuration_path_trimmed = configuration_path.split(".")[0]
    graphs = pydot.graph_from_dot_file(parameters["input_file"])
    graph = graphs[0]
    graph.del_node('"\\n\\n\\n"')

    dependency_graph = nx.DiGraph(nx.nx_pydot.from_pydot(graph))

    layers = get_layers(dependency_graph)
    all_equipment = sorted(find_leaf_nodes(dependency_graph, layers))
    configuration_all_modes = {}
    # for mode in find_root_nodes(dependency_graph):
    #     # logging.info(f"Configuration for mode {get_node_name(dependency_graph, mode)}")
    #     sub_graph = dependency_graph.copy()
    #     for node in dependency_graph:
    #         if node not in list(nx.bfs_tree(dependency_graph, mode)):
    #             sub_graph.remove_node(node)
    #     layers = get_layers(sub_graph)
    #     configuration_all_modes[get_node_name(dependency_graph, mode)] = get_configuration_new(sub_graph, layers,
    #                                                                                            all_equipment)

    # statistics["configuration_all_modes"] = configuration_all_modes
    all_equipment_names = [get_node_name(dependency_graph, n) for n in all_equipment]
    all_modes = [get_node_name(dependency_graph, n) for n in find_root_nodes(dependency_graph)]
    statistics["all_equipments"] = all_equipment_names
    statistics["number_of_equipments"] = len(all_equipment_names)
    statistics["all_modes"] = all_modes

    all_actions, all_list_actions, all_actions_cost, action_to_name_mapping, name_to_action_mapping = get_all_actions(
        dependency_graph,
        all_equipment_names, statistics)
    # return configuration_all_modes, all_modes, all_equipment_names
    statistics["all_actions"] = all_actions
    statistics["all_list_actions"] = all_list_actions
    statistics["all_actions_cost"] = all_actions_cost
    statistics["name_to_action_mapping"] = name_to_action_mapping
    statistics["action_to_name_mapping"] = action_to_name_mapping


def get_all_actions(dependency_graph, all_equipment, statistics):
    unique_graph_list, unique_node_lists, leaf_name_lists = create_graph_list(dependency_graph, False)
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
