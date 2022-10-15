# python library of graph analysis functions for parsing the .DOT configurations

import networkx as nx
import itertools
import re
import logging
from functools import reduce
from operator import mul

logging.basicConfig(
    format="[%(levelname)s] %(funcName)s: %(message)s")
# logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

def find_root_nodes(G):
    root_nodes = []
    for node in G:
        if len(list(G.predecessors(node))) == 0:
            root_nodes.append(node)
    return root_nodes


def find_root_node(G):
    for node in G:
        if len(list(G.predecessors(node))) == 0:
            return node


def find_leaf_nodes(G, layers):
    leaf_nodes = []
    for layer in layers:
        leaf_nodes_layer = []
        for node in layer:
            if not list(G.successors(node)):
                leaf_nodes_layer.append(node)
        # convert to names, sort alphabetically, convert back to node IDs
        leaf_node_names_layer = sorted([get_node_name(G, leaf_node) for leaf_node in leaf_nodes_layer])
        # logging.debug(f"Leaf_node_names_layer: {leaf_node_names_layer}")
        # logging.debug(f"Leaf_nodes_layer: {[get_node_id(G, leaf_node_name) for leaf_node_name in leaf_node_names_layer]}")
        leaf_nodes.extend([get_node_id(G, leaf_node_name) for leaf_node_name in leaf_node_names_layer])
    return leaf_nodes


def find_leaf_nodes_per_root_node(G, root_node):
    leaf_nodes = []
    subgraph = nx.bfs_tree(G, root_node)
    for node in subgraph:
        if len(list(subgraph.successors(node))) == 0:
            leaf_nodes.append(node)
    # convert to names, sort alphabetically, convert back to node IDs
    leaf_node_names = sorted([get_node_name(G, leaf_node) for leaf_node in leaf_nodes])
    return [get_node_id(G, leaf_node_name) for leaf_node_name in leaf_node_names]


def get_node_name(G, node):
    attr = G.nodes[node]
    if 'xlabel' in attr:
        return attr['xlabel'].strip('\"')
    else:
        return node


def get_node_names(G):
    return [get_node_name(G, node) for node in list(G.nodes)]


def get_root_node_names(G):
    root_nodes = find_root_nodes(G)
    root_node_names = {}
    for root_node in root_nodes:
        root_node_names[get_node_name(G, root_node)] = root_node
    # logging.info(root_node_names)
    return root_node_names


def get_mode_indices(G):
    roots = sorted([get_node_name(G, root) for root in find_root_nodes(G)])
    mode_indices = {}
    for index, root in enumerate(roots):
        mode_indices[root] = index
    # logging.info(mode_indices)
    return mode_indices


def get_mode_indices_appended(G):
    mode_indices_appended = get_mode_indices(G)
    if "off" not in mode_indices_appended.keys():
        mode_indices_appended["off"] = len(mode_indices_appended)
    return mode_indices_appended


def get_node_id(G, name):
    for node in list(G.nodes):
        attr = G.nodes[node]
        if 'xlabel' in attr:
            if name == attr['xlabel'].strip('\"'):
                return node
    logging.debug(f"Could not find ID for node with name {name}. Returning passed name as an ID")
    return name


def get_dependencies(G, root_node):
    return [get_node_name(G, dependency) for dependency in list(G.successors(root_node))]


def create_statespace(G, node_list, verbose):
    import re
    graph_list = []
    statespace = []
    for node in node_list:
        attr = G.nodes[node]
        if 'xlabel' in attr:
            name = attr['xlabel'].strip('\"')
            if name.startswith(">="):
                # logging.info(attr)
                # logging.info(node)
                # statespace.append(int(re.findall("\d+", name)[0]))
                # variable_size = ((len(list(G.predecessors(node))) * (len(list(G.predecessors(node)))-1)) // 2)
                # statespace.append(variable_size)
                available = len(list(G.successors(node)))
                required = int(re.findall("\d+", name)[0])
                # the length should be equivalent to the binomial coefficient available nCr required
                statespace.append(len(list(itertools.combinations(range(available), available - required))) + 1)
            elif name.startswith("OR"):
                statespace.append(2)
    # if there are no disjunctive assemblies, there is only one state
    if not statespace:
        statespace.append(1)
    return statespace


def create_permutations(statespace, verbose):
    permutations = []
    reversed_statespace = statespace
    reversed_statespace.reverse()
    logging.info(reversed_statespace)
    number_of_permutations = [reversed_statespace[0]]
    for index, variable in enumerate(reversed_statespace[1:]):
        # logging.info(index)
        # logging.info(variable)
        number_of_permutations.append(variable * number_of_permutations[index])
    number_of_permutations.reverse()
    # logging.info(number_of_permutations)
    for i in range(max(number_of_permutations)):
        temp = i
        out_list = []
        # build a list for every state
        for j in range(len(statespace)):
            # logging.info("j: " + str(j))
            if j != (len(statespace) - 1):
                # logging.info(number_of_permutations[j+1])
                quotient = temp // number_of_permutations[j + 1]
                # remainder = temp & number_of_permutations[j+1]
                out_list.append(quotient)
                # logging.info(out_list)
                temp -= quotient * number_of_permutations[j + 1]
            else:  # last element
                out_list.append(temp)
        # logging.info(out_list)
        permutations.append(out_list)
    logging.debug(permutations)
    return permutations


def check_for_upstream_dependencies(G, nodes_to_be_deleted, root_node, predecessor, verbose):
    try:
        for node in nodes_to_be_deleted:
            logging.debug(
                f"[{get_node_name(G, root_node)}] Check node {node} ({get_node_name(G, node)}) that is about to be deleted") if verbose else None
            if len(list(G.predecessors(node))) > 1:
                logging.debug(
                    f"[{get_node_name(G, root_node)}] Node {node} ({get_node_name(G, node)}) has {str(len(list(G.predecessors(node))))} predecessors") if verbose else None
                logging.debug(
                    f"[{get_node_name(G, root_node)}] Therefore, we will not delete: {list(nx.bfs_tree(G, node, reverse=False))}") if verbose else None
                logging.debug(
                    f"[{get_node_name(G, root_node)}] But we will delete the edge from {predecessor} to the node(s) {list(nx.bfs_tree(G, node, reverse=False))}") if verbose else None
                for node_to_keep in list(nx.bfs_tree(G, node, reverse=False)):
                    if node_to_keep in nodes_to_be_deleted:
                        nodes_to_be_deleted.remove(node_to_keep)
                        try:
                            G.remove_edge(predecessor, node_to_keep)
                        except:
                            logging.warning(f"[{get_node_name(G, root_node)}] Could not delete edge. Skipping")
                    else:
                        logging.debug(
                            f"[{get_node_name(G, root_node)}] Node {node} ({get_node_name(G, node)}) is not about to be deleted. Skipping...")
            elif len(list(G.predecessors(node))) == 0:
                logging.warning(
                    f"[{get_node_name(G, root_node)}] That's strange. Node {node} ({get_node_name(G, node)}) has no predecessors") if verbose else None
            else:
                logging.debug(
                    f"[{get_node_name(G, root_node)}] The node has just one predecessor and will be deleted") if verbose else None
            logging.debug(
                f"[{get_node_name(G, root_node)}] New state of node_to_be_deleted: {nodes_to_be_deleted}") if verbose else None
        return nodes_to_be_deleted, G
    except ValueError:
        logging.info(ValueError)


# create n graphs for every permutation
def create_graphs(G, root_node, node_list, permutations, verbose):
    graph_list = []
    node_lists = []
    for permutation in permutations:
        current_index = permutations.index(permutation)
        all_permutations = len(permutations)
        # logging.info(permutation) if verbose else None
        index = 0
        new_G = G.copy()
        for node in G:
            if not node in node_list:
                new_G.remove_node(node)
        # new_G = nx.path_graph(5)
        # new_G = G.__class__()
        # new_G.add_nodes_from(G)
        # new_G.add_edges_from(G.edges)
        for node in node_list:
            # for node in list(nx.bfs_tree(new_G, root_node, reverse=True))[::-1]:
            attr = new_G.nodes[node]
            # logging.info(attr)

            # remove the other options
            if 'xlabel' in attr:
                name = attr['xlabel'].strip('\"')
                if name.startswith(">="):
                    # logging.info(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] [{current_index}/{all_permutations}] We are at disjunctive node {node} ({name}) with predecessor {list(new_G.predecessors(node))}") if verbose else None
                    available = len(list(new_G.successors(node)))
                    required = int(re.findall("\d+", name)[0])
                    difference = available - required
                    # logging.debug(f"The index is {index}, the permutation[index] is {permutation[index]}, the number of combinations is {len(list(itertools.combinations(range(available), available-required)))}")
                    if permutation[index] == len(list(itertools.combinations(range(available), available - required))):
                        # logging.debug("Skipped the +1 configuration")
                        logging.debug(
                            f"[{get_node_name(G, root_node)}] [{current_index + 1}/{all_permutations}] For the assembly {get_node_name(G, list(G.predecessors(node))[0])}, we are in the +1 configuration where all equipment is used")
                        index += 1
                    else:
                        if difference == 1:
                            # logging.info(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] We will delete successor #{permutation[index]} in list {list(new_G.successors(node))}") if verbose else None
                            root_to_be_deleted = list(new_G.successors(node))[permutation[index]]
                            nodes_to_be_deleted = set(nx.bfs_tree(new_G, root_to_be_deleted, reverse=False))
                        elif difference >= 2:
                            # logging.info(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] We will delete {difference} successors in list {list(new_G.successors(node))}") if verbose else None
                            nodes_to_be_deleted = set()
                            for redundancy in range(difference):
                                logging.debug(
                                    f"[{get_node_name(G, root_node)}] [{current_index + 1}/{all_permutations}] Deleting successor #{list(itertools.combinations(range(available), difference))[permutation[index]][redundancy]}") if verbose else None
                                root_to_be_deleted = list(new_G.successors(node))[
                                    list(itertools.combinations(range(available), difference))[permutation[index]][
                                        redundancy]]
                                nodes_to_be_deleted.update(list(nx.bfs_tree(new_G, root_to_be_deleted, reverse=False)))

                        # logging.debug(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] Nodes to be deleted before checking for upstream dependencies: {nodes_to_be_deleted}") if verbose else None
                        nodes_to_be_deleted, new_G = check_for_upstream_dependencies(new_G, list(nodes_to_be_deleted),
                                                                                     root_node, node, verbose)
                        logging.debug(
                            f"[{get_node_name(G, root_node)}] [{current_index + 1}/{all_permutations}] Nodes to be deleted after  checking for upstream dependencies: {nodes_to_be_deleted}") if verbose else None
                        for node_to_be_deleted in nodes_to_be_deleted:
                            new_G.remove_node(node_to_be_deleted)
                        index += 1
                elif name.startswith("OR"):
                    # logging.info(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] We are at disjunctive node {node} ({name}) with predecessor {list(new_G.predecessors(node))}") if verbose else None
                    # logging.info(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] We will delete successor #{permutation[index]} in list {list(new_G.successors(node))}") if verbose else None
                    root_to_be_deleted = list(new_G.successors(node))[permutation[index]]
                    nodes_to_be_deleted = set(nx.bfs_tree(new_G, root_to_be_deleted, reverse=False))

                    # logging.debug(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] Nodes to be deleted before checking for upstream dependencies: {nodes_to_be_deleted}") if verbose else None
                    nodes_to_be_deleted, new_G = check_for_upstream_dependencies(new_G, list(nodes_to_be_deleted),
                                                                                 root_node, node, verbose)
                    # logging.debug(f"[{get_node_name(G, root_node)}] [{current_index}/{all_permutations}] Nodes to be deleted after  checking for upstream dependencies: {nodes_to_be_deleted}") if verbose else None
                    for node_to_be_deleted in nodes_to_be_deleted:
                        new_G.remove_node(node_to_be_deleted)
                    index += 1
        graph_list.append(new_G)
        node_lists.append(list(nx.bfs_tree(new_G, root_node, reverse=False)))
        # node_lists.append(node_list)
    return graph_list, node_lists


# prune duplicates
def remove_duplicates(graph_list, node_lists, root_node, verbose):
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
    logging.info("[{get_node_name(G, root_node)}] Deleted " + str(counter) + " duplicate graphs") if verbose else None
    return unique_graph_list, unique_node_lists


# combine the functions above so we can resolve the redundancies and create one graph for every combination of equipment
# that realizes the mode
def create_graph_list(G, verbose):
    unique_graph_list = {}
    unique_node_lists = {}
    leaf_name_lists = {}
    for root_node in find_root_nodes(G):
        logging.info(get_node_name(G, root_node))
        node_list = list(nx.bfs_tree(G, root_node, reverse=False))[::-1]
        if len(node_list) <= 1:
            logging.warning(
                f"[{get_node_name(G, root_node)}] Node list empty for root node {root_node} ({get_node_name(G, root_node)}). Skipping...")
            break
        statespace = create_statespace(G, node_list, verbose)
        logging.info(f"[{get_node_name(G, root_node)}] statespace: {statespace}") if verbose else None
        permutations = create_permutations(statespace, verbose)
        # logging.info(f"[{get_node_name(G, root_node)}] {len(permutations)} permutations: ") if verbose else None
        # for permutation in permutations:
        #     logging.info(f"[{get_node_name(G, root_node)}] {permutation}") if verbose else None
        graph_list, node_lists = create_graphs(G, root_node, node_list, permutations, verbose)
        unique_graph_list[root_node], unique_node_lists[root_node] = remove_duplicates(graph_list, node_lists,
                                                                                       root_node, verbose)
        leaf_name_lists[root_node] = []
        # logging.debug(f"Unique graph list[{root_node}] = {list(unique_graph_list[root_node][0])}")
        for graph in unique_graph_list[root_node]:
            new_G = G.copy()
            for node in G:
                if not node in list(graph):
                    new_G.remove_node(node)
            # logging.debug(f"Root node: {root_node}, Graph: {new_G}")
            # logging.debug(f"Graph node names: {[get_node_name(new_G, node) for node in new_G]}")
            layers = get_layers(new_G)
            # logging.debug(f"Leaf nodes: {find_leaf_nodes(new_G, layers)}")
            leaf_name_lists[root_node].append(
                sorted([get_node_name(new_G, node) for node in find_leaf_nodes(new_G, layers)]))
    return (unique_graph_list, unique_node_lists, leaf_name_lists)


def get_configuration(G, all_equipment):
    configuration = []
    index = 0
    while index < len(all_equipment):
        node_name = all_equipment[index]
        # find out if the component is part of an assembly
        part_of_assembly = False
        for predecessor in list(G.predecessors(get_node_id(G, node_name))):
            if get_node_name(G, predecessor).startswith(">="):
                part_of_assembly = True
                available = len(list(G.successors(predecessor)))
                required = int(re.findall("\d+", get_node_name(G, predecessor))[0])
                configuration.append((required, list(range(index, index + available))))
                index += available
                break
            elif get_node_name(G, predecessor).startswith("OR"):
                part_of_assembly = True
                configuration.append((1, list(range(index, index + 2))))
                index += 2
                break
        if not part_of_assembly:
            configuration.append((1, [index]))
            index += 1
    return configuration


def get_configuration_dict(G, all_equipment):
    configuration = {}
    index = 0
    while index < len(all_equipment):
        node_name = all_equipment[index]
        # find out if the component is part of an assembly
        part_of_assembly = False
        for predecessor in list(G.predecessors(get_node_id(G, node_name))):
            if get_node_name(G, predecessor).startswith(">="):
                part_of_assembly = True
                assembly_name = get_node_name(G, list(G.predecessors(predecessor))[0])
                available = len(list(G.successors(predecessor)))
                required = int(re.findall("\d+", get_node_name(G, predecessor))[0])
                configuration[assembly_name] = (required, list(range(index, index + available)))
                index += available
                break
            elif get_node_name(G, predecessor).startswith("OR"):
                part_of_assembly = True
                assembly_name = get_node_name(G, list(G.predecessors(predecessor))[0])
                configuration[assembly_name] = (1, list(range(index, index + 2)))
                index += 2
                break
        if not part_of_assembly:
            configuration[node_name] = (1, [index])
            index += 1
    return configuration


def get_layers(G):
    # go through every layer of the tree and look for OR/>= assemblies
    known_nodes = find_root_nodes(G)
    layers = [list(set(known_nodes))]
    while True:
        candidates = []
        next_layer = []
        node_dict = {known_node: get_node_name(G, known_node) for known_node in known_nodes}
        logging.debug(f"Known nodes: {node_dict}")
        for root_node in known_nodes:
            candidates.extend(list(nx.bfs_tree(G, root_node, depth_limit=1)))
        node_dict = {candidate: get_node_name(G, candidate) for candidate in candidates}
        logging.debug(f"Candidates: {node_dict}")
        next_layer = candidates.copy()
        for candidate in candidates:
            if candidate in known_nodes:
                next_layer.remove(candidate)
            for predecessor in list(nx.bfs_tree(G, candidate, reverse=True))[1:]:
                if predecessor not in known_nodes:
                    next_layer.remove(candidate)
                    break
        node_dict = {next_layer_node: get_node_name(G, next_layer_node) for next_layer_node in next_layer}
        logging.debug(f"Next layer: {node_dict}")
        if not next_layer:
            logging.debug("Traversed the whole graph")
            return layers
        known_nodes.extend(list(set(next_layer)))
        layers.append(list(set(next_layer)))

    # logging.info(set(nodes_on_layer))
    # logging.info(set([list(nx.bfs_tree(G, root_node)) for root_node in find_root_nodes(G)]))


def get_configuration_new(G, layers, all_equipment):
    configuration = {}
    for layer in layers:
        for node in layer:
            # see if we are below some previous assembly
            valid_assembly = True
            for predecessor in list(nx.bfs_tree(G, node, reverse=True))[1:]:
                if get_node_name(G, predecessor).startswith(">=") or get_node_name(G, predecessor).startswith("OR"):
                    # logging.warning(f"Node {node} will be skipped since it is part of a higher assembly")
                    valid_assembly = False
            if valid_assembly:
                # see if this is a single component
                if node in all_equipment:
                    configuration[get_node_name(G, node)] = (1, [[all_equipment.index(node)]])
                elif get_node_name(G, node).startswith(">="):
                    logging.debug(f"Found a greater_than assembly")
                    # get the leaf nodes underneath
                    leaf_nodes = find_leaf_nodes_per_root_node(G, node)
                    assembly_name = get_node_name(G, list(G.predecessors(node))[0])
                    available = len(list(G.successors(node)))
                    required = int(re.findall("\d+", get_node_name(G, node))[0])
                    configuration[assembly_name] = (
                    required, [all_equipment.index(leaf_node) for leaf_node in leaf_nodes])
                    # configuration[assembly_name] = (required, [get_node_name(G, leaf_node) for leaf_node in leaf_nodes])
                    # look if there are more assemblies underneath
                    hierarchichal_assembly = False
                    for successor in G.successors(node):
                        if successor not in all_equipment:
                            hierarchichal_assembly = True
                    if hierarchichal_assembly:
                        valid_configurations = []
                        for successor in G.successors(node):
                            logging.debug(f"Root node for successor tree: {get_node_name(G, successor)}")
                            sub_G = G.copy()
                            for node in G:
                                if not node in list(nx.bfs_tree(G, successor)):
                                    sub_G.remove_node(node)
                            if len(list(sub_G)) == 1:
                                valid_configurations.append([all_equipment.index(successor)])
                            else:
                                (unique_graph_list, unique_node_lists, leaf_name_lists) = create_graph_list(sub_G,
                                                                                                            False)
                                logging.debug(
                                    f"Leaf names for root {get_node_name(sub_G, successor)}: {leaf_name_lists[successor]}")
                                for leaf_name_list in leaf_name_lists[successor]:
                                    valid_configurations.append(
                                        [all_equipment.index(get_node_id(sub_G, leaf_name)) for leaf_name in
                                         leaf_name_list])

                        # logging.debug(f"There is more than just leaf nodes in the assembly for root {node}")
                        # valid_configurations = []
                        # temp_configurations = {}
                        # for successor in G.successors(node):
                        #     logging.debug(f"Root node for successor tree: {get_node_name(G, successor)}")
                        #     sub_G = G.copy()
                        #     for node in G:
                        #         if not node in list(nx.bfs_tree(G, successor)):
                        #             sub_G.remove_node(node)
                        #     logging.debug(f"Nodes in the subtree: {list(nx.bfs_tree(G, successor))}")
                        #     logging.debug(f"Nodes in the subtree: {list(sub_G.nodes)}")
                        #     if len(list(sub_G)) == 1:
                        #         logging.debug(f"There is only node {list(sub_G.nodes)[0]} in the tree.")
                        #         temp_configurations[successor] = [[all_equipment.index(list(sub_G.nodes)[0])]]
                        #     else:
                        #         (unique_graph_list, unique_node_lists, leaf_name_lists) = create_graph_list(sub_G, False)
                        #         logging.debug(f"Leaf names for root {get_node_name(sub_G, successor)}: {leaf_name_lists[successor]}")
                        #         temp_configurations[successor] = []
                        #         for leaf_name_list in leaf_name_lists[successor]:
                        #             # logging.debug(f"Add this list to temp_configurations: {[all_equipment.index(get_node_id(sub_G, leaf_name)) for leaf_name in leaf_name_list]}")
                        #             temp_configurations[successor].append([all_equipment.index(get_node_id(sub_G, leaf_name)) for leaf_name in leaf_name_list])
                        # logging.debug(f"temp_configurations: {temp_configurations}")
                    if hierarchichal_assembly:
                        logging.debug(f"New configuration for assembly {assembly_name}: {valid_configurations}")
                        configuration[assembly_name] = (1, valid_configurations)
                        # # build useful lists out of the temp_configurations
                        # logging.debug(f"statespace: {[len(temp_configurations[root_node]) for root_node in temp_configurations]}")
                        # for permutation in create_permutations([len(temp_configurations[root_node]) for root_node in temp_configurations], False):
                        #     new_configuration = []
                        #     for digit, root_node in zip(permutation, temp_configurations):
                        #         logging.debug(f"digit: {digit}, configuration: {temp_configurations[root_node]}")
                        #         new_configuration.extend(temp_configurations[root_node][digit])
                        #     logging.debug(f"new configuration: {new_configuration}")
                        #     valid_configurations.append(new_configuration)
                        # logging.debug(f"New configuration for assembly {assembly_name}: {valid_configurations}")
                        # configuration[assembly_name] = (required, valid_configurations)
                    else:
                        logging.debug(
                            f"New configuration for assembly {assembly_name}: {[[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes]}")
                        configuration[assembly_name] = (
                        required, [[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes])

                elif get_node_name(G, node).startswith("OR"):
                    logging.debug(f"Found an OR assembly")
                    # get the leaf nodes underneath
                    leaf_nodes = find_leaf_nodes_per_root_node(G, node)
                    assembly_name = get_node_name(G, list(G.predecessors(node))[0])
                    # configuration[assembly_name] = (required, [all_equipment.index(leaf_node) for leaf_node in leaf_nodes])
                    # configuration[assembly_name] = (1, [all_equipment.index(leaf_node) for leaf_node in leaf_nodes])
                    # configuration[assembly_name] = (required, [get_node_name(G, leaf_node) for leaf_node in leaf_nodes])
                    # look if there are more assemblies underneath
                    hierarchichal_assembly = False
                    for successor in G.successors(node):
                        if successor not in all_equipment:
                            hierarchichal_assembly = True
                    if hierarchichal_assembly:
                        valid_configurations = []
                        for successor in G.successors(node):
                            logging.debug(f"Root node for successor tree: {get_node_name(G, successor)}")
                            sub_G = G.copy()
                            for node in G:
                                if not node in list(nx.bfs_tree(G, successor)):
                                    sub_G.remove_node(node)
                            if len(list(sub_G)) == 1:
                                valid_configurations.append([all_equipment.index(successor)])
                            else:
                                (unique_graph_list, unique_node_lists, leaf_name_lists) = create_graph_list(sub_G,
                                                                                                            False)
                                logging.debug(
                                    f"Leaf names for root {get_node_name(sub_G, successor)}: {leaf_name_lists[successor]}")
                                for leaf_name_list in leaf_name_lists[successor]:
                                    valid_configurations.append(
                                        [all_equipment.index(get_node_id(sub_G, leaf_name)) for leaf_name in
                                         leaf_name_list])
                    if hierarchichal_assembly:
                        logging.debug(f"New configuration for assembly {assembly_name}: {valid_configurations}")
                        configuration[assembly_name] = (1, valid_configurations)
                    else:
                        logging.debug(
                            f"New configuration for assembly {assembly_name}: {[[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes]}")
                        configuration[assembly_name] = (
                        1, [[all_equipment.index(leaf_node)] for leaf_node in leaf_nodes])

    return configuration


def check_isolability(all_equipment, leaf_name_lists):
    all_equipment_set = set(all_equipment)

    all_leaf_name_lists = []
    for root_node in leaf_name_lists:
        for configuration in leaf_name_lists[root_node]:
            all_leaf_name_lists.append(configuration)

    alternative_sets = {}

    for equipment in all_equipment:
        alternative_set = set()
        alternative_configurations = [leaf_name_list for leaf_name_list in all_leaf_name_lists if
                                      equipment not in leaf_name_list]
        for configuration in alternative_configurations:
            for element in configuration:
                alternative_set.add(element)
        alternative_sets[equipment] = alternative_set

    non_isolable = []
    for equipment in alternative_sets:
        # print(equipment)
        alternative_sets[equipment].add(equipment)
        if alternative_sets[equipment] == all_equipment_set:
            logging.info("Equipment " + equipment + " is isolatable.")
        else:
            logging.info("Equipment " + equipment + " is not isolatable.")
            non_isolable.append(equipment)
    return non_isolable


def check_recoverability(G, all_equipment, leaf_name_lists):
    non_recoverable = []
    for mode in leaf_name_lists:
        mode_available = True
        for equipment in all_equipment:
            mode_available_per_equipment = False
            for configuration in leaf_name_lists[mode]:
                if equipment not in configuration:
                    logging.info("The mode " + get_node_name(G, mode) + " is available if " + equipment + " has a fault.")
                    mode_available_per_equipment = True
                    break
            if not mode_available_per_equipment:
                logging.info("The mode " + get_node_name(G, mode) + " is not available if " + equipment + " has a fault.")
                mode_available = False
        logging.info("The fault recoverability for mode " + get_node_name(G, mode) + " is " + str(mode_available) + "\n\n")
        if not mode_available:
            non_recoverable.append(mode)
    return non_recoverable

def examine_successor(G, node, equipment_fault_probabilities):
    # logging.info(f"Examining successor {get_node_name(G, node)} ({node})")
    if len(list(G.successors(node))) == 0:
        # We abort the recursive cycle
        fault_probability = equipment_fault_probabilities[get_node_name(G, node)]
    else:
        if get_node_name(G, node).startswith(">="):
            # sub assembly fails for num_available-num_required+1 faults
            required = int(re.findall("\d+", get_node_name(G, node))[0])
            # logging.info(f"Node {get_node_name(G, node)} ({node}) requires {required} operational children out of {len(list(G.successors(node)))}")
            successor_reliabilities = [1 - examine_successor(G, successor, equipment_fault_probabilities) for successor in G.successors(node)]
            # logging.info(f"successor_reliabilities: {successor_reliabilities}")
            fault_probability_combinations = [1 - reduce(mul, combination, 1) for combination in itertools.combinations(successor_reliabilities, required)]
            # logging.info(f"fault_probability_combinations: {sorted(fault_probability_combinations)}")
            fault_probability = reduce(mul, sorted(fault_probability_combinations)[:len(list(G.successors(node)))-required+1], 1)
        elif get_node_name(G, node).startswith("OR"):
            # sub assembly fails if all members fail
            fault_probability = reduce(mul, [examine_successor(G, successor, equipment_fault_probabilities) for successor in G.successors(node)], 1)
        else:
            # sub assembly fails if one of the members fails
            successor_reliabilities = [1-examine_successor(G, successor, equipment_fault_probabilities) for successor in G.successors(node)]
            reliability = reduce(mul, successor_reliabilities, 1)
            fault_probability = 1 - reliability
    # logging.info(f"Fault probability of {get_node_name(G, node)} ({node}) is {fault_probability:.6}")
    return fault_probability
