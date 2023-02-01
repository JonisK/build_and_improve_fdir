# python library of graph analysis functions for parsing the .DOT configurations

import networkx as nx
import itertools
import re
import logging
import concurrent.futures
import time
from functools import reduce
from operator import mul

logging.basicConfig(
    format="[%(levelname)s] %(funcName)s: %(message)s")


# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)


class Permutation_Fetcher():
    def __init__(self, all_permutations):
        self.all_permutations = all_permutations
        self.graph_list = []
        self.node_lists = []
        self.computed_permutations = 0

    def add_permutation(self, root_node, graph, node_list):
        self.graph_list.append(graph)
        self.node_lists.append(node_list)
        self.computed_permutations += 1
        logging.info(f"[{get_node_name(graph, root_node)}] [{self.computed_permutations}/{self.all_permutations}], "
                     f"{self.computed_permutations / self.all_permutations:.1%}")

    def add_dummy(self):
        self.computed_permutations += 1

    def get_permutations(self):
        return self.computed_permutations

    def get_done(self):
        return self.computed_permutations == self.all_permutations

    def get_graph_lists(self):
        return self.graph_list, self.node_lists


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


def get_effects(G, node):
    effects = []
    attr = G.nodes[node]
    if 'xlabel' in attr:
        name = attr['xlabel'].strip('\"')
        lines = name.split('\n')
        for line in lines:
            if '=' in line:
                effects.append(line)  # [variable] = value
    return effects


def find_isolated_nodes(G):
    isolated_nodes = []
    for node in G:
        if len(list(G.predecessors(node))) == 0 and len(list(G.successors(node))) == 0:
            isolated_nodes.append(node)
    return isolated_nodes


def is_component(G, node):
    attr = G.nodes[node]
    if 'xlabel' in attr:
        name = attr['xlabel'].strip('\"')
        if not '=' in name:
            return True
    return False


def is_guard(G, node):
    attr = G.nodes[node]
    if 'xlabel' in attr:
        name = attr['xlabel'].strip('\"')
        if '=' in name:
            return True
    return False


def find_leaf_nodes(G, layers=None, root_node=None, type='all'):
    # if the user specifies a root node, limit the graph and recompute the layers
    if root_node:
        subgraph = nx.bfs_tree(G, root_node)
        layers = get_layers(subgraph)
    else:
        subgraph = G
        if not layers:
            layers = get_layers(G)

    leaf_nodes = []
    for layer in layers:
        leaf_nodes_layer = []
        for node in layer:
            if not list(G.successors(node)):
                if type == 'all' \
                        or (type == 'components' and is_component(G, node)) \
                        or (type == 'guards' and is_guard(G, node)):
                    leaf_nodes_layer.append(node)
        # convert to names, sort alphabetically, convert back to node IDs
        leaf_node_names_layer = sorted([get_node_name(G, leaf_node) for leaf_node in leaf_nodes_layer])
        # logging.debug(f"Leaf_node_names_layer: {leaf_node_names_layer}")
        # logging.debug(f"Leaf_nodes_layer: {[get_node_id(G, leaf_node_name) for leaf_node_name in leaf_node_names_layer]}")
        leaf_nodes.extend([get_node_id(G, leaf_node_name) for leaf_node_name in leaf_node_names_layer])
    return leaf_nodes


def get_node_name(G, node):
    attr = G.nodes[node]
    if 'xlabel' in attr:
        # only return the first line, do not include quotes
        return attr['xlabel'].strip('\"').split('\n')[0]
    else:
        # if the node has no name tag, use its ID
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


def get_assembly_name(G, node):
    # get the name of the predecessor because this indicates the function of the assembly
    assembly_name = get_node_name(G, list(G.predecessors(node))[0])
    # an example name would be e.g. reaction_wheels_>=3
    return assembly_name + f"_{get_node_name(G, node).strip('>=')}"


def create_configuration_space(G, layers):
    configurations = {}
    configuration_space = {}
    number_of_permutations = 1
    assembly_names = {}
    for layer in layers:
        for node in layer:
            # for node in node_list:
            attr = G.nodes[node]
            if 'xlabel' in attr:
                name = attr['xlabel'].strip('\"')
                if name.startswith(">=") or name.startswith("OR"):
                    logging.debug(f"Found disjunctive node {name}")
                    # assembly_names[node] = get_assembly_name(G, node)
                    available = len(list(G.successors(node)))
                    if name.startswith(">="):
                        required = int(re.findall(r"\d+", name)[0])
                        # configurations is a list that describes all successors that shall be part of the assembly per configuration
                        # the size should be equivalent to the binomial coefficient available nCr required
                        configuration_list = list(itertools.combinations(range(available), required))
                        # we add 1 to also include a configuration where the whole >= assembly is in use
                        configuration_list.append(tuple(range(available)))
                    else:
                        required = 1
                        # the size is equivalent to the number of successors
                        configuration_list = list(itertools.combinations(range(available), required))
                    configurations[node] = configuration_list
                    configuration_space[node] = len(configuration_list)
                    # configuration_space[assembly_names[node]] = len(configuration_list)
                    number_of_permutations *= len(configuration_list)
    return configurations, configuration_space, number_of_permutations


def create_permutations(configurations):
    permutations = []
    for assembly in configurations:
        logging.debug(f"{assembly=}")
        if not permutations:
            # create the first entry
            logging.debug(f"The permutation list is empty so we create the first one as a copy of {assembly}.")
            for configuration in configurations[assembly]:
                permutations.append({assembly: configuration})
        else:
            # multiplicate the number of permutations with the number of configurations in this assembly
            logging.debug(f"Multiplicate {assembly} with the current permutation list.")
            new_permutations = []
            # Every permutation is a dict containing assemblies as key and their configuration as value
            for permutation in permutations:
                for configuration in configurations[assembly]:
                    # For every possible configuration that an assembly can have, we copy
                    # the existing permutation and add the new configuration to it
                    new_permutation = permutation.copy()
                    new_permutation[assembly] = configuration
                    new_permutations.append(new_permutation)
            permutations = new_permutations
        logging.debug(f"{permutations=}")
    return permutations


def get_predecessors_outside_removal_range(G, node, nodes_to_be_deleted):
    predecessors_outside_removal_range = []
    for predecessor in G.predecessors(node):
        if predecessor not in nodes_to_be_deleted:
            predecessors_outside_removal_range.append(predecessor)
    logging.debug(f"With {node=} and {nodes_to_be_deleted=}, we get {predecessors_outside_removal_range}")
    return predecessors_outside_removal_range


def get_predecessors_inside_removal_range(G, node, nodes_to_be_deleted):
    predecessors_inside_removal_range = []
    for predecessor in G.predecessors(node):
        if predecessor in nodes_to_be_deleted:
            predecessors_inside_removal_range.append(predecessor)
    logging.debug(f"With {node=} and {nodes_to_be_deleted=}, we get {predecessors_inside_removal_range}")
    return predecessors_inside_removal_range


def check_for_upstream_dependencies(G, nodes_to_be_deleted, root_node, predecessor):
    try:
        for node in nodes_to_be_deleted:
            logging.debug(
                f"[{get_node_name(G, root_node)}] Check node {node} ({get_node_name(G, node)}) "
                f"that is about to be deleted")
            if len(list(G.predecessors(node))) > 1 and get_predecessors_outside_removal_range(G, node,
                                                                                              nodes_to_be_deleted):
                # if there are nodes that we want to delete but that other branches depend on, we will
                # not delete those but delete the connecting edges so we can show that our branch does
                # not depend on them
                logging.debug(
                    f"[{get_node_name(G, root_node)}] Node {node} ({get_node_name(G, node)}) "
                    f"has {str(len(list(G.predecessors(node))))} predecessors and some of "
                    f"them are outside the range of nodes we intend to delete")
                logging.debug(
                    f"[{get_node_name(G, root_node)}] Therefore, we will not delete: "
                    f"{list(nx.bfs_tree(G, node, reverse=False))}")
                logging.debug(
                    f"[{get_node_name(G, root_node)}] But we will delete the edge from "
                    f"{get_predecessors_inside_removal_range(G, node, nodes_to_be_deleted)} "
                    f"to the node(s) {list(nx.bfs_tree(G, node, reverse=False))}. "
                    f"And we will also delete the edge from {predecessor} "
                    f"to the node(s) {list(nx.bfs_tree(G, node, reverse=False))}")
                for node_to_keep in list(nx.bfs_tree(G, node, reverse=False)):
                    if node_to_keep in nodes_to_be_deleted:
                        nodes_to_be_deleted.remove(node_to_keep)
                        # remove edges between the nodes we want to keep and the nodes that will be deleted
                        for predecessor_inside_removal_range in get_predecessors_inside_removal_range(G, node,
                                                                                                      nodes_to_be_deleted):
                            try:
                                G.remove_edge(predessor_inside_removal_range, node_to_keep)
                            except:
                                logging.warning((
                                    f"[{get_node_name(G, root_node)}] Could not find edge from "
                                    f"{predecessor_inside_removal_range} ({get_node_name(G, predecessor_inside_removal_range)}) "
                                    f"to {node_to_keep} ({get_node_name(G, node_to_keep)}). Skipping"))
                        # remove edge from the predecessor, i.e. the disjunctive node, to the node we want to keep
                        try:
                            G.remove_edge(predecessor, node_to_keep)
                        except:
                            logging.warning((
                                f"[{get_node_name(G, root_node)}] Could not find edge from "
                                f"{predecessor} ({get_node_name(G, predecessor)}) "
                                f"to {node_to_keep} ({get_node_name(G, node_to_keep)}). Skipping"))
                    else:
                        logging.debug((
                            f"[{get_node_name(G, root_node)}] Node {node} ({get_node_name(G, node)}) "
                            f"is not about to be deleted. Skipping..."))
            elif len(list(G.predecessors(node))) > 1 and not predecessors_outside_removal_range(G, node,
                                                                                                nodes_to_be_deleted):
                logging.debug((
                    f"[{get_node_name(G, root_node)}] Node {node} ({get_node_name(G, node)}) "
                    f"has {str(len(list(G.predecessors(node))))} predecessors but they all "
                    f"lie in the range of nodes we are about to delete"))
                logging.debug(
                    f"[{get_node_name(G, root_node)}] Therefore, we will delete: "
                    f"{list(nx.bfs_tree(G, node, reverse=False))}")
            elif len(list(G.predecessors(node))) == 0:
                logging.warning((
                    f"[{get_node_name(G, root_node)}] That's strange. Node {node} ({get_node_name(G, node)}) "
                    f"has no predecessors"))
            else:
                logging.debug((
                    f"[{get_node_name(G, root_node)}] The node has just one predecessor and will "
                    f"be deleted"))
            logging.debug((
                f"[{get_node_name(G, root_node)}] New state of node_to_be_deleted: {nodes_to_be_deleted}"))
        return nodes_to_be_deleted, G
    except ValueError:
        logging.info(ValueError)

    # go through the graph and delete all disjunctive nodes
    inv_subgraph = subgraph.copy()
    for assembly in configurations:
        inv_subgraph.remove_node(assembly)
    # then search for the remaining sub-graph accessible from the root
    # the nodes of this subgraph are not affected by the configurations
    invariant_nodes = set(nx.bfs_tree(inv_subgraph, root_node, reverse=False))


def create_graph(G, root_node, node_list, invariant_nodes, configuration_space, permutation, current_index,
                 all_permutations, fetcher):
    logging.debug(f"Create graphs for {permutation}")

    # determine the set of nodes that we want to keep as children of the disjunctive nodes
    nodes_to_keep = set()
    nodes_to_delete = set()
    # go from top to bottom so the shadowing check will be effective
    # for assembly in reversed(permutation):
    for assembly in permutation:
        nodes_to_keep_per_assembly = set()
        nodes_to_delete_per_assembly = set()
        # add the disjunctive node
        # logging.debug(f"Adding assembly {assembly} to nodes_to_keep")
        nodes_to_keep_per_assembly |= {assembly}
        # add all its successors that are active in this configuration to nodes_to_keep
        # add all inactive successors to nodes_to_delete
        for index, successor in enumerate(G.successors(assembly)):
            # determine the successor tree
            node_list_successor = nx.bfs_tree(G, successor, reverse=False)
            # if the successor tree includes more disjunctive nodes, cut off the tree there
            inv_successor_graph = G.copy()
            for node in G:
                if not node in node_list_successor:
                    inv_successor_graph.remove_node(node)
            for assembly_2 in permutation:
                if assembly_2 in inv_successor_graph:
                    inv_successor_graph.remove_node(assembly_2)

            if index in permutation[assembly]:
                # logging.debug(f"Adding the successors of {list(G.successors(assembly))[index]} to nodes_to_keep")
                # logging.debug(f"Add {set(nx.bfs_tree(G, successor, reverse=False))} for {index=}, {successor=} to nodes_to_keep_per_assembly")
                # add the node list of this successor tree to the set of nodes that shall be kept
                nodes_to_keep_per_assembly |= set(nx.bfs_tree(inv_successor_graph, successor, reverse=False))
                logging.debug(
                    f"{assembly=}, {successor=}, added to nodes_to_keep: {set(nx.bfs_tree(inv_successor_graph, successor, reverse=False))}")
            else:
                # logging.debug(f"Add {set(nx.bfs_tree(G, successor, reverse=False))} for {index=}, {successor=} to nodes_to_delete_per_assembly")
                nodes_to_delete_per_assembly |= set(nx.bfs_tree(inv_successor_graph, successor, reverse=False))
                logging.debug(
                    f"{assembly=}, {successor=}, added to nodes_to_delete: {set(nx.bfs_tree(inv_successor_graph, successor, reverse=False))}")
            # in case of inter-dependencies, make sure that the nodes_to_delete will
            # not cause the removal of nodes required for the selected branch(es)
            nodes_to_delete_per_assembly -= nodes_to_keep_per_assembly
            nodes_to_delete_per_assembly -= nodes_to_keep
        logging.debug(f"{nodes_to_keep_per_assembly=}")
        logging.debug(f"{nodes_to_delete_per_assembly=}")
        # check if the nodes_to_delete cuts into the set of invariant_nodes
        while any([node in invariant_nodes for node in nodes_to_delete_per_assembly]):
            logging.debug(f"nodes_to_delete_per_assembly violates the invariant_nodes")
            for node in nodes_to_delete_per_assembly:
                if node in invariant_nodes:
                    logging.debug(f"Removing {set(nx.bfs_tree(G, node))} from the nodes_to_delete_per_assembly")
                    nodes_to_delete_per_assembly -= set(nx.bfs_tree(G, node))
                    break
        # check if the assembly is shadowed by an earlier assembly
        # if not nodes_to_keep_per_assembly.issubset(nodes_to_delete):
        if not list(G.predecessors(assembly))[0] in nodes_to_delete:
            logging.debug(
                f"[{get_node_name(G, root_node)}] Assembly {assembly} is not shadowed by earlier assemblies and therefore added")
            nodes_to_keep |= nodes_to_keep_per_assembly
            nodes_to_keep -= nodes_to_delete_per_assembly
            nodes_to_delete |= nodes_to_delete_per_assembly
        else:
            logging.debug(
                f"[{get_node_name(G, root_node)}] Assembly {assembly} is shadowed by earlier assemblies and therefore not added")
    logging.debug(f"[{get_node_name(G, root_node)}] {nodes_to_keep=}")

    # the nodes remaining in the graph are the union of the invariant_nodes and node_to_keep
    new_node_list = nodes_to_keep | invariant_nodes
    logging.debug(f"[{get_node_name(G, root_node)}] {new_node_list=}")

    new_G = G.copy()
    for node in G:
        if not node in new_node_list:
            new_G.remove_node(node)

    logging.debug(f"Finished graph for {permutation}. Adding it to the list")
    fetcher.add_permutation(root_node, new_G, list(nx.bfs_tree(new_G, root_node, reverse=False)))

    # new_G = G.copy()
    # for node in G:
    #     if not node in node_list:
    #         new_G.remove_node(node)
    # for node in node_list:
    #     # for node in list(nx.bfs_tree(new_G, root_node, reverse=True))[::-1]:
    #     attr = new_G.nodes[node]
    #     # logging.info(attr)
    #
    #     # remove the other options
    #     if 'xlabel' in attr:
    #         name = attr['xlabel'].strip('\"')
    #         # if we are at a disjunctive node, find the configuration for it
    #         if name.startswith(">=") or name.startswith("OR"):
    #             logging.debug(f"We are at disjunctive node {node} and about "
    #                           f"to apply configuration {permutation[node]}")
    #             # find the nodes we want to delete in the permutation
    #             candidates = list(new_G.successors(node))
    #             roots_to_be_deleted = candidates.copy()
    #             for index in permutation[node]:
    #                 roots_to_be_deleted.remove(candidates[index])
    #             logging.debug(f"We want to delete the successors {roots_to_be_deleted} and their children")
    #
    #             nodes_to_be_deleted = set()
    #             for root_to_be_deleted in roots_to_be_deleted:
    #                 nodes_to_be_deleted.update(list(nx.bfs_tree(new_G, root_to_be_deleted, reverse=False)))
    #             # find the nodes that
    #             logging.debug(
    #                 f"[{get_node_name(G, root_node)}] "
    #                 f"[{current_index + 1}/{all_permutations}] "
    #                 f"Nodes to be deleted before checking for upstream "
    #                 f"dependencies: {nodes_to_be_deleted}")
    #             nodes_to_be_deleted, new_G = check_for_upstream_dependencies(
    #                 new_G, list(nodes_to_be_deleted), root_node, node)
    #             logging.debug(
    #                 f"[{get_node_name(G, root_node)}] "
    #                 f"[{current_index + 1}/{all_permutations}] "
    #                 f"Nodes to be deleted after  checking for upstream "
    #                 f"dependencies: {nodes_to_be_deleted}")
    #             for node_to_be_deleted in nodes_to_be_deleted:
    #                 new_G.remove_node(node_to_be_deleted)
    # logging.info(f"Finished graph for {permutation}. Adding it to the list")
    # fetcher.add_permutation(new_G, list(nx.bfs_tree(new_G, root_node, reverse=False)))


# create n graphs for every permutation
def create_graphs(G, root_node, node_list, invariant_nodes, configuration_space, permutations, threading):
    number_of_permutations = len(permutations)
    fetcher = Permutation_Fetcher(number_of_permutations)
    logging.debug(f"Created fetcher object")
    if threading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for permutation in permutations:
                current_index = permutations.index(permutation)
                # launch a thread that shall check the permutation
                executor.submit(create_graph, G, root_node, node_list, invariant_nodes, configuration_space,
                                permutation,
                                current_index, number_of_permutations, fetcher)
        timeout = 6  # seconds
        while not fetcher.get_done() and timeout > 0:
            logging.info("Waiting for all threads to finish")
            time.sleep(2)
            timeout -= 2
    else:
        for permutation in permutations:
            current_index = permutations.index(permutation)
            create_graph(G, root_node, node_list, invariant_nodes, configuration_space, permutation,
                         current_index, number_of_permutations, fetcher)
    graph_list, node_lists = fetcher.get_graph_lists()
    return graph_list, node_lists


# prune duplicates
def remove_duplicates(graph_list, node_lists, permutations, root_node):
    unique_graph_list = []
    unique_node_lists = []
    configuration_list = []
    counter = 0
    for graph, node_list, permutation in zip(graph_list, node_lists, permutations):
        if node_list not in unique_node_lists:
            unique_graph_list.append(graph)
            unique_node_lists.append(node_list)
            configuration_list.append(permutation)
        else:
            counter += 1
    logging.info(f"[{get_node_name(graph_list[0], root_node)}] Deleted {counter} duplicate graphs")
    return unique_graph_list, unique_node_lists, configuration_list


class Graph_List_Fetcher():
    def __init__(self, number_of_root_nodes):
        self.unique_graph_list = {}
        self.unique_node_lists = {}
        self.leaf_name_lists = {}
        self.configuration_list = {}
        self.configuration_space = {}
        self.number_of_root_nodes = number_of_root_nodes
        self.number_of_computed_root_nodes = 0

    def add_list(self,
                 root_node,
                 unique_graph_list_per_root_node,
                 unique_node_lists_per_root_node,
                 leaf_name_lists_per_root_node,
                 configuration_list_per_root_node,
                 configuration_space_per_root_node):
        self.unique_graph_list[root_node] = unique_graph_list_per_root_node
        self.unique_node_lists[root_node] = unique_node_lists_per_root_node
        self.leaf_name_lists[root_node] = leaf_name_lists_per_root_node
        self.configuration_list[root_node] = configuration_list_per_root_node
        self.configuration_space[root_node] = configuration_space_per_root_node

        self.number_of_computed_root_nodes += 1
        logging.info(f"Completed creating graphs for root node "
                     f"{get_node_name(unique_graph_list_per_root_node[0], root_node)} - "
                     f"Overall progress: "
                     f"{self.number_of_computed_root_nodes / self.number_of_root_nodes:.1%}")

    def skip(self):
        self.number_of_computed_root_nodes += 1

    def get_done(self):
        return self.number_of_computed_root_nodes == self.number_of_root_nodes

    def get_computed_root_nodes(self):
        return self.number_of_computed_root_nodes

    def get_graph_lists(self):
        return self.unique_graph_list, self.unique_node_lists, self.leaf_name_lists, self.configuration_list, self.configuration_space


def subtract_list(minuend, subtrahend):
    return [element for element in minuend if not element in subtrahend]


def create_graph_list_per_root_node(G, root_node, list_fetcher, threading):
    logging.info(f"Start analyzing root node {get_node_name(G, root_node)}")
    node_list = list(nx.bfs_tree(G, root_node, reverse=False))[::-1]
    subgraph = G.copy()
    for node in G:
        if not node in node_list:
            subgraph.remove_node(node)
    layers = get_layers(subgraph)
    logging.debug(f"[{get_node_name(subgraph, root_node)}] {layers=}")
    logging.debug(f"{node_list=}")
    if len(node_list) <= 1:
        logging.warning(
            f"[{get_node_name(G, root_node)}] Node list empty for root node {root_node} ({get_node_name(subgraph, root_node)}). Skipping...")
        list_fetcher.skip(root_node)

    configurations, configuration_space_per_root_node, number_of_permutations = create_configuration_space(subgraph,
                                                                                                           layers)
    logging.debug(f"[{get_node_name(subgraph, root_node)}] Configurations: {configurations}")
    logging.debug(f"[{get_node_name(subgraph, root_node)}] Configuration space: {configuration_space_per_root_node}")
    permutations = create_permutations(configurations)
    # permutations = [{'N58': (0,), 'N26': (0,), 'N71': (0,)}]
    logging.debug(f"{len(permutations)=}")

    # determine the set of nodes not affected by the configurations

    # invariant_nodes = set(node_list)
    # for assembly in configurations:
    #     # take the intersection of all nodes not part of a specific assembly
    #     # to get the nodes which are never affected by any assembly (at least for this root node)
    #     logging.debug(f"[{get_node_name(G, root_node)}] Remove subgraph from invariant_nodes: {list(nx.bfs_tree(G, assembly, reverse=False))}")
    #     invariant_nodes &= set(subtract_list(node_list, list(nx.bfs_tree(G, assembly, reverse=False))))
    #     logging.debug(f"[{get_node_name(G, root_node)}] After update: {invariant_nodes=}")

    # go through the graph and delete all disjunctive nodes
    inv_subgraph = subgraph.copy()
    for assembly in configurations:
        inv_subgraph.remove_node(assembly)
    # then search for the remaining sub-graph accessible from the root
    # the nodes of this subgraph are not affected by the configurations
    invariant_nodes = set(nx.bfs_tree(inv_subgraph, root_node, reverse=False))

    logging.debug(f"[{get_node_name(subgraph, root_node)}] {invariant_nodes=}")

    graph_list, node_lists = create_graphs(subgraph, root_node, node_list, invariant_nodes,
                                           configuration_space_per_root_node, permutations, threading)
    unique_graph_list_per_root_node, unique_node_lists_per_root_node, configuration_list_per_root_node = remove_duplicates(
        graph_list,
        node_lists,
        permutations,
        root_node)
    leaf_name_lists_per_root_node = []
    for unique_graph, permutation_node_list in zip(unique_graph_list_per_root_node, unique_node_lists_per_root_node):
        permutation_graph = subgraph.copy()
        for node in subgraph:
            if not node in permutation_node_list:
                permutation_graph.remove_node(node)
        # logging.debug(f"Root node: {root_node}, Graph: {new_G}")
        # logging.debug(f"Graph node names: {[get_node_name(new_G, node) for node in new_G]}")
        # logging.debug(f"Leaf nodes: {find_leaf_nodes(new_G)}")
        leaf_name_lists_per_root_node.append(
            sorted([get_node_name(subgraph, node) for node in find_leaf_nodes(unique_graph, type='components')]))
    list_fetcher.add_list(root_node,
                          unique_graph_list_per_root_node,
                          unique_node_lists_per_root_node,
                          leaf_name_lists_per_root_node,
                          configuration_list_per_root_node,
                          configuration_space_per_root_node)


# combine the functions above so we can resolve the redundancies and create one graph for every combination of equipment
# that realizes the mode
def create_graph_list(G, threading=False):
    root_nodes = find_root_nodes(G)
    list_fetcher = Graph_List_Fetcher(len(root_nodes))
    logging.debug(f"Created fetcher object")
    if threading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as list_executor:
            for root_node in root_nodes:
                # launch a thread that shall check the mode
                list_executor.submit(create_graph_list_per_root_node, G, root_node, list_fetcher, threading)
        timeout = 6  # seconds
        while not list_fetcher.get_done() and timeout > 0:
            logging.info("Waiting for all threads to finish")
            logging.debug(
                f"Progress so far: {list_fetcher.get_computed_root_nodes()} of {len(root_nodes)} root nodes completed")
            logging.debug(f"{root_nodes=}")
            time.sleep(2)
            timeout -= 2
    else:
        for root_node in root_nodes:
            create_graph_list_per_root_node(G, root_node, list_fetcher, threading)

    unique_graph_list, unique_node_lists, leaf_name_lists, configuration_list, configuration_space = list_fetcher.get_graph_lists()
    return unique_graph_list, unique_node_lists, leaf_name_lists, configuration_list, configuration_space


def get_layers(G):
    # go through every layer of the tree and look for OR/>= assemblies
    known_nodes = find_root_nodes(G)
    layers = [list(set(known_nodes))]
    while True:
        candidates = []
        next_layer = []
        node_dict = {known_node: get_node_name(G, known_node) for known_node in known_nodes}
        # logging.debug(f"Known nodes: {node_dict}")
        for root_node in known_nodes:
            candidates.extend(list(nx.bfs_tree(G, root_node, depth_limit=1)))
        node_dict = {candidate: get_node_name(G, candidate) for candidate in candidates}
        # logging.debug(f"Candidates: {node_dict}")
        next_layer = candidates.copy()
        for candidate in candidates:
            if candidate in known_nodes:
                next_layer.remove(candidate)
            for predecessor in list(nx.bfs_tree(G, candidate, reverse=True))[1:]:
                if predecessor not in known_nodes:
                    next_layer.remove(candidate)
                    break
        node_dict = {next_layer_node: get_node_name(G, next_layer_node) for next_layer_node in next_layer}
        # logging.debug(f"Next layer: {node_dict}")
        if not next_layer:
            logging.debug("All layers found. Traversed the whole graph")
            return layers
        known_nodes.extend(list(set(next_layer)))
        layers.append(list(set(next_layer)))

    # logging.info(set(nodes_on_layer))
    # logging.info(set([list(nx.bfs_tree(G, root_node)) for root_node in find_root_nodes(G)]))


def check_isolability(all_equipment, leaf_name_lists, number_of_faults):
    if number_of_faults == 1:
        plural = False
    else:
        plural = True

    all_leaf_name_lists = []
    for root_node in leaf_name_lists:
        for configuration in leaf_name_lists[root_node]:
            all_leaf_name_lists.append(configuration)
    all_equipment_set = set(all_equipment)

    isolable_combinations = []
    non_isolable_combinations = []
    for components in itertools.combinations(all_equipment, number_of_faults):
        alternative_set = set()
        for leaf_name_list in all_leaf_name_lists:
            if not set(components).issubset(set(leaf_name_list)):
                for element in leaf_name_list:
                    alternative_set.add(element)
        for component in components:
            alternative_set.add(component)
        if alternative_set == all_equipment_set:
            isolable_combinations.append(components)
            logging.debug(f"Fault{'s' if plural else ''} in component{'s' if plural else ''} "
                          f"{', '.join(components)} {'are' if plural else 'is'} isolable.")
        else:
            non_isolable_combinations.append(components)
            logging.debug(f"Fault{'s' if plural else ''} in component{'s' if plural else ''} "
                          f"{', '.join(components)} {'are' if plural else 'is'} not isolable.")

    non_isolable = set([component for combination in non_isolable_combinations for component in combination])
    isolable = set(all_equipment) - non_isolable
    for equipment in all_equipment:
        if equipment in non_isolable:
            logging.info("Equipment " + equipment + " is not isolable.")
        else:
            logging.info("Equipment " + equipment + " is isolable.")
    return sorted(isolable), sorted(non_isolable)


def check_recoverability(G, all_equipment, leaf_name_lists, number_of_faults):
    if number_of_faults == 1:
        plural = False
    else:
        plural = True

    recoverable = []
    non_recoverable = []
    for mode in leaf_name_lists:
        mode_available = True
        for components in itertools.combinations(all_equipment, number_of_faults):
            mode_available_per_combination = False
            for leaf_name_list in leaf_name_lists[mode]:
                if reduce(mul, [component not in leaf_name_list for component in components]):
                    logging.debug(
                        f"The mode {get_node_name(G, mode)} is available if {components} {'have' if plural else 'has'} a fault.")
                    mode_available_per_combination = True
                    break
            if not mode_available_per_combination:
                logging.info(
                    f"The mode {get_node_name(G, mode)} is not available if {components} {'have' if plural else 'has'} a fault.")
                mode_available = False
        logging.info(f"The fault recoverability for mode {get_node_name(G, mode)} is {mode_available}\n\n")
        if mode_available:
            recoverable.append(mode)
        else:
            non_recoverable.append(mode)
    return sorted(recoverable), sorted(non_recoverable)


def get_fault_probability(G, node, equipment_fault_probabilities):
    # logging.info(f"Examining successor {get_node_name(G, node)} ({node})")
    if len(list(G.successors(node))) == 0:
        # We abort the recursive cycle
        fault_probability = equipment_fault_probabilities[get_node_name(G, node)]
    else:
        if get_node_name(G, node).startswith(">="):
            # sub assembly fails for num_available-num_required+1 faults
            required = int(re.findall(r"\d+", get_node_name(G, node))[0])
            # logging.info(f"Node {get_node_name(G, node)} ({node}) requires {required} operational children out of {len(list(G.successors(node)))}")
            successor_reliabilities = [1 - get_fault_probability(G, successor, equipment_fault_probabilities) for
                                       successor
                                       in exclude_guards(G, G.successors(node))]
            # logging.info(f"successor_reliabilities: {successor_reliabilities}")
            fault_probability_combinations = [1 - reduce(mul, combination, 1) for combination in
                                              itertools.combinations(successor_reliabilities, required)]
            # logging.info(f"fault_probability_combinations: {sorted(fault_probability_combinations)}")
            fault_probability = reduce(mul, sorted(fault_probability_combinations)[
                                            :len(list(exclude_guards(G, G.successors(node)))) - required + 1], 1)
        elif get_node_name(G, node).startswith("OR"):
            # sub assembly fails if all members fail
            fault_probability = reduce(mul,
                                       [get_fault_probability(G, successor, equipment_fault_probabilities) for successor
                                        in
                                        exclude_guards(G, G.successors(node))], 1)
        else:
            # sub assembly fails if one of the members fails
            successor_reliabilities = [1 - get_fault_probability(G, successor, equipment_fault_probabilities) for
                                       successor
                                       in exclude_guards(G, G.successors(node))]
            reliability = reduce(mul, successor_reliabilities, 1)
            fault_probability = 1 - reliability
    # logging.info(f"Fault probability of {get_node_name(G, node)} ({node}) is {fault_probability:.6}")
    return fault_probability


def exclude_guards(graph, nodes):
    return_list = []
    for node in nodes:
        if len(list(graph.successors(node))):
            # no leaf so no need to remove the node
            return_list.append(node)
        else:
            if not '=' in get_node_name(graph, node):
                # exclude guard leaves
                return_list.append(node)
    return return_list
