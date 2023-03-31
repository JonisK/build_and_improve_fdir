# python library of graph analysis functions for parsing the .DOT configurations

import networkx as nx
import itertools
import re
import logging
import concurrent.futures
import time
from functools import reduce
from operator import mul


class PermutationFetcher():
    def __init__(self, all_permutations):
        self.all_permutations = all_permutations
        self.graph_list = []
        self.node_lists = []
        self.computed_permutations = 0

    def add_permutation(self, root_node, graph, node_list):
        self.graph_list.append(graph)
        self.node_lists.append(node_list)
        self.computed_permutations += 1

    def get_done(self):
        return self.computed_permutations == self.all_permutations

    def get_graph_lists(self):
        return self.graph_list, self.node_lists


class GraphListFetcher():
    def __init__(self, number_of_root_nodes):
        self.unique_graph_list = {}
        self.unique_node_lists = {}
        self.component_lists = {}
        self.configuration_list = {}
        self.configuration_space = {}
        self.number_of_root_nodes = number_of_root_nodes
        self.number_of_computed_root_nodes = 0

    def add_list(self, root_node, unique_graph_list_mode, unique_node_lists_mode,
                 component_lists_mode, configuration_list_mode, configuration_space_mode):
        self.unique_graph_list[root_node] = unique_graph_list_mode
        self.unique_node_lists[root_node] = unique_node_lists_mode
        self.component_lists[root_node] = component_lists_mode
        self.configuration_list[root_node] = configuration_list_mode
        self.configuration_space[root_node] = configuration_space_mode

        self.number_of_computed_root_nodes += 1
        logging.info(f"[{get_node_name(unique_graph_list_mode[0], root_node)}] Completed - "
                     f"Overall progress: "
                     f"{self.number_of_computed_root_nodes / self.number_of_root_nodes:.1%}")

    def skip(self):
        self.number_of_computed_root_nodes += 1

    def get_done(self):
        return self.number_of_computed_root_nodes == self.number_of_root_nodes

    def get_graph_lists(self):
        return self.unique_graph_list, self.unique_node_lists, self.component_lists, \
            self.configuration_list, self.configuration_space


def get_node_name(graph, node):
    attr = graph.nodes[node]
    if 'xlabel' in attr:
        # only return the first line, do not include quotes
        return attr['xlabel'].strip('\"').split('\n')[0]
    else:
        return node  # if the node has no name tag, return its ID


def get_node_id(graph, name):
    for node in list(graph.nodes):
        attr = graph.nodes[node]
        if 'xlabel' in attr:
            if name == attr['xlabel'].strip('\"').split('\n')[0]:
                return node
    logging.warning(f"Node ID for {name=} not found.")
    return name  # return name in case the node ID could not be found


def get_root_node_names(graph):
    root_nodes = find_root_nodes(graph)
    root_node_names = {}
    for root_node in root_nodes:
        root_node_names[get_node_name(graph, root_node)] = root_node
    # logging.info(root_node_names)
    return root_node_names


def get_subgraph(graph, node_list):
    # Copy the graph and remove all nodes not belonging to node_list to yield a subgraph
    subgraph = graph.copy()
    for node in graph:
        if node not in node_list:  # graph must contain all nodes specified in node_list
            subgraph.remove_node(node)
    return subgraph


def get_mode_indices(graph):
    roots = sorted([get_node_name(graph, root) for root in find_root_nodes(graph)])
    mode_indices = {}
    for index, root in enumerate(roots):
        mode_indices[root] = index
    return mode_indices


def get_mode_indices_appended(graph):
    mode_indices_appended = get_mode_indices(graph)
    if "off" not in mode_indices_appended.keys():
        mode_indices_appended["off"] = len(mode_indices_appended)
    return mode_indices_appended


def get_effects(graph, node):
    effects = []
    attr = graph.nodes[node]
    if 'xlabel' in attr:
        name = attr['xlabel'].strip('\"')
        lines = name.split('\n')
        for line in lines:
            if '=' in line:
                effects.append(line)  # [variable] = value
    return effects


def is_component(graph, node):
    attr = graph.nodes[node]
    if 'xlabel' in attr:
        name = attr['xlabel'].strip('\"')
        if not '=' in name:
            return True
    return False


def is_guard(graph, node):
    attr = graph.nodes[node]
    if 'xlabel' in attr:
        name = attr['xlabel'].strip('\"')
        if '=' in name:
            return True
    return False


def find_leaf_nodes(graph, layers=None, root_node=None, type='all'):
    # if the user specifies a root node, limit the graph and recompute the layers
    if root_node:
        subgraph = nx.bfs_tree(graph, root_node)
        layers = get_layers(subgraph)
    else:
        if not layers:
            layers = get_layers(graph)

    leaf_nodes = []
    for layer in layers:
        leaf_nodes_layer = []
        for node in layer:
            if not list(graph.successors(node)):
                if type == 'all' \
                        or (type == 'components' and is_component(graph, node)) \
                        or (type == 'guards' and is_guard(graph, node)):
                    leaf_nodes_layer.append(node)
        # convert to names, sort alphabetically, convert back to node IDs
        leaf_node_names_layer = sorted([get_node_name(graph, leaf_node)
                                        for leaf_node in leaf_nodes_layer])
        leaf_nodes.extend([get_node_id(graph, leaf_node_name)
                           for leaf_node_name in leaf_node_names_layer])
    return leaf_nodes


def find_isolated_nodes(graph):
    isolated_nodes = []
    for node in graph:
        if len(list(graph.predecessors(node))) == 0 and len(list(graph.successors(node))) == 0:
            isolated_nodes.append(node)
    return isolated_nodes


def find_root_nodes(graph):
    root_nodes = []
    for node in graph:
        if len(list(graph.predecessors(node))) == 0:
            root_nodes.append(node)
    return root_nodes


def get_layers(graph):
    # Go through every layer of the tree and look for OR/>= assemblies
    known_nodes = find_root_nodes(graph)
    layers = [sorted(known_nodes)]  # the first layer contains all root nodes
    while True:
        candidates = []
        for root_node in known_nodes:  # find all direct children of every node know so far
            candidates.extend(list(nx.bfs_tree(graph, root_node, depth_limit=1)))
        next_layer = candidates.copy()
        for candidate in candidates:
            if candidate in known_nodes:
                # Do not add children that have been added over a shorter path already
                next_layer.remove(candidate)
            predecessors = list(nx.bfs_tree(graph, candidate, reverse=True))  # all ancestors
            for predecessor in predecessors[1:]:  # start from 1 to skip the candidate itself
                # If not all ancestors are known yet (i.e. multiple ancestors on different layers),
                # remove the candidate for now. It will be added once the last ancestor has been
                # added to known_nodes.
                if predecessor not in known_nodes:
                    next_layer.remove(candidate)
                    break
        if not next_layer:  # if no candidate remains, the layers list is complete
            return layers
        # If not finished, add the remaining candidates and start another iteration a.k.a. layer
        known_nodes.extend(list(set(next_layer)))  # convert to set to remove duplicates
        layers.append(sorted(list(set(next_layer))))


def create_configuration_space(graph, layers):
    configurations = {}  # valid combinations for each assembly
    configuration_space = {}  # number of combinations for each assembly
    number_of_permutations = 1  # number of permutations in the whole graph
    for layer in layers:  # follow the hierarchy of the model by following the layers list
        for node in layer:
            name = get_node_name(graph, node)
            if name.startswith(">=") or name.startswith("OR"):
                available = len(list(graph.successors(node)))
                if name.startswith(">="):  # k-out-of-n assembly
                    required = int(re.findall(r"\d+", name)[0])
                    # configuration_list is a list of tuples that describes all successors that
                    # shall be part of the assembly per configuration. The size should be equivalent
                    # to the binomial coefficient: available nCr required
                    configuration_list = list(itertools.combinations(range(available), required))
                    # We add 1 combination to also include a configuration where all components in a
                    # k-out-of-n assembly are in use
                    configuration_list.append(tuple(range(available)))
                else:  # OR assembly
                    required = 1
                    # The size is equivalent to the number of successors
                    configuration_list = list(itertools.combinations(range(available), required))
                configurations[node] = configuration_list
                configuration_space[node] = len(configuration_list)
                number_of_permutations *= len(configuration_list)
    return configurations, configuration_space, number_of_permutations


def create_permutations(configurations):  # compute all valid configurations for the graph
    # Every permutation is a dict containing assemblies as keys and their configurations as values
    permutations = []
    for assembly in configurations:
        if not permutations:  # create the first entry
            # The permutation list is empty, so we create the first one as a copy of the current
            # assembly
            for configuration in configurations[assembly]:
                permutations.append({assembly: configuration})
        else:
            # Multiplicate the number of permutations found so far with the number of configurations
            # in this assembly by using a nested for-loop and replacing permutations with
            # new_permutations
            new_permutations = []
            for permutation in permutations:
                for configuration in configurations[assembly]:
                    # For every possible configuration that an assembly can have, we copy
                    # the existing permutation and add the new configuration to it
                    new_permutation = permutation.copy()
                    new_permutation[assembly] = configuration
                    new_permutations.append(new_permutation)
            permutations = new_permutations
    return permutations


def remove_duplicates(graph_list, node_lists, permutations, root_node):  # prune duplicates
    unique_graph_list = []
    unique_node_lists = []
    configuration_list = []
    counter = 0
    for graph, node_list, permutation in zip(graph_list, node_lists, permutations):
        # Check if the graph is known already. We assume that the node_list suffices to identify
        # the graph. This might not cover systems where components are used for multiple functions
        if node_list not in unique_node_lists:
            unique_graph_list.append(graph)
            unique_node_lists.append(node_list)
            configuration_list.append(permutation)
        else:  # if the graph is known already, do not add it to the list of unique graphs
            counter += 1
    if counter:
        logging.info(f"[{get_node_name(graph_list[0], root_node)}] Deleted {counter} duplicate "
                     f"graphs")
    return unique_graph_list, unique_node_lists, configuration_list


# Configure the graph according to the permutation and prune all non-required nodes
def create_graph(graph, root_node, invariant_nodes, permutation, fetcher):
    # Determine the set of nodes that we want to keep as children of the disjunctive nodes
    nodes_to_keep = set()
    nodes_to_delete = set()
    # Go from top to bottom so the shadowing check will be effective for assembly in
    # reversed(permutation):
    for assembly in permutation:
        nodes_to_keep_per_assembly = set()
        nodes_to_delete_per_assembly = set()

        nodes_to_keep_per_assembly |= {assembly}  # Add the disjunctive node
        # Add all its successors that are active in this configuration to nodes_to_keep
        # add all inactive successors to nodes_to_delete
        for index, successor in enumerate(graph.successors(assembly)):  # look at all successors
            # Graph containing all nodes reachable from the successor
            inv_successor_graph = get_subgraph(graph, nx.bfs_tree(graph, successor))
            # Cut off the successor tree at locations where more disjunctive nodes follow
            for assembly_2 in permutation:  # variable name assembly is already in use
                if assembly_2 in inv_successor_graph:
                    inv_successor_graph.remove_node(assembly_2)

            if index in permutation[assembly]:  # check if the current successor is in permutation
                # Add the node list of this successor tree to the set of nodes that shall be kept
                nodes_to_keep_per_assembly |= set(nx.bfs_tree(inv_successor_graph, successor))
            else:
                # Add the node list of this successor tree to the set of nodes to be deleted
                nodes_to_delete_per_assembly |= set(nx.bfs_tree(inv_successor_graph, successor))

            # In case of interdependencies, make sure that the nodes_to_delete will
            # not cause the removal of nodes required for the selected branch(es)
            nodes_to_delete_per_assembly -= nodes_to_keep_per_assembly
            nodes_to_delete_per_assembly -= nodes_to_keep

        # Check if the nodes_to_delete cuts into the set of invariant_nodes
        while any([node in invariant_nodes for node in nodes_to_delete_per_assembly]):
            for node in nodes_to_delete_per_assembly:
                if node in invariant_nodes:
                    nodes_to_delete_per_assembly -= set(nx.bfs_tree(graph, node))
                    break

        # Check if the assembly is shadowed by an earlier assembly
        if list(graph.predecessors(assembly))[0] in nodes_to_delete:
            # If the assembly is shadowed by a higher one, we will not update the nodes_to_keep and
            # nodes_to_delete sets
            logging.debug(f"[{get_node_name(graph, root_node)}] Assembly {assembly} is shadowed by "
                          f"earlier assemblies and therefore not added")
        else:
            # If the assembly is not shadowed, we complete the cycle by updating nodes_to_keep and
            # nodes_to_delete with the newly built sets for this assembly
            nodes_to_keep |= nodes_to_keep_per_assembly
            nodes_to_keep -= nodes_to_delete_per_assembly
            nodes_to_delete |= nodes_to_delete_per_assembly

    # The nodes remaining in the graph are the union of the invariant_nodes and node_to_keep
    new_node_list = nodes_to_keep | invariant_nodes

    # Get the graph that connects the components required in this configuration to root_node
    new_graph = get_subgraph(graph, new_node_list)

    fetcher.add_permutation(root_node, new_graph, new_node_list)


# Distribute the analysis of all permutations over multiple threads
def create_graphs_mode(graph, root_node, node_list, invariant_nodes, configuration_space,
                       permutations, threading):
    number_of_permutations = len(permutations)
    fetcher = PermutationFetcher(number_of_permutations)
    logging.debug(f"Created fetcher object")
    if threading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            for permutation in permutations:
                # launch a thread that shall check the permutation
                executor.submit(create_graph, graph, root_node, invariant_nodes, permutation,
                                fetcher)
        timeout = 30  # seconds
        while not fetcher.get_done() and timeout > 0:
            logging.info("Waiting for all permutation threads to finish")
            time.sleep(2)
            timeout -= 2
    else:
        for permutation in permutations:
            create_graph(graph, root_node, invariant_nodes, permutation, fetcher)
    graph_list, node_lists = fetcher.get_graph_lists()
    return graph_list, node_lists


# Determine the subgraph and invariant nodes for every mode. Get all permutations and call
# create_graphs_mode for analyzing every configuration
def create_graph_list_mode(main_graph, root_node, list_fetcher, threading):
    logging.info(f"[{get_node_name(main_graph, root_node)}] Start analysis")

    # Get the subgraph for this root node containing all nodes that are reachable from root_node
    node_list = list(nx.bfs_tree(main_graph, root_node))
    if len(node_list) <= 1:
        logging.warning(f"[{get_node_name(main_graph, root_node)}] Node list empty for root node "
                        f"{root_node} ({get_node_name(main_graph, root_node)}). Skipping...")
        list_fetcher.skip(root_node)
    subgraph = get_subgraph(main_graph, node_list)
    layers = get_layers(subgraph)

    # Analyze the disjunctive assemblies and get all possible configurations for each of them
    configurations, configuration_space_mode, number_of_permutations \
        = create_configuration_space(subgraph, layers)
    # Compile all combinations of configurations for all assemblies. Each permutation will yield
    # one feasible configuration of the whole mode graph
    permutations = create_permutations(configurations)

    # Determine the set of nodes not affected by the configurations
    inv_subgraph = subgraph.copy()
    for assembly in configurations: 
        inv_subgraph.remove_node(assembly)  # go through the graph and delete all disjunctive nodes
    # Then search for the remaining sub-graph accessible from the root. The nodes of this subgraph 
    # are not affected by the configurations
    invariant_nodes = set(nx.bfs_tree(inv_subgraph, root_node))

    logging.debug(f"[{get_node_name(subgraph, root_node)}] {invariant_nodes=}")

    # create_graphs_mode will look at each configuration determined by the permutations
    graph_list, node_lists = create_graphs_mode(subgraph, root_node, node_list, invariant_nodes,
                                                configuration_space_mode, permutations, threading)
    # Remove duplicate configurations, i.e. where one assembly shadowed another
    unique_graph_list_mode, unique_node_lists_mode, configuration_list_mode \
        = remove_duplicates(graph_list, node_lists, permutations, root_node)
    # Generate the component_lists which is useful for checking fault isolability and tolerance
    component_lists_mode = [sorted([get_node_name(graph, node)
                                    for node in find_leaf_nodes(graph, type='components')])
                            for graph in unique_graph_list_mode]  # nested list comprehension
    list_fetcher.add_list(root_node, unique_graph_list_mode, unique_node_lists_mode,
                          component_lists_mode, configuration_list_mode, configuration_space_mode)


# Distribute the analysis of the individual modes over multiple threads
def create_graph_list(main_graph, threading=False):
    root_nodes = find_root_nodes(main_graph)  # modes equal root nodes
    # list_fetcher will collect the results of the individual threads
    list_fetcher = GraphListFetcher(len(root_nodes))
    # Threading allows for faster execution if multiple modes exist in the graph
    if threading:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for root_node in root_nodes:
                # launch a thread that checks the mode
                executor.submit(create_graph_list_mode, main_graph, root_node, 
                                list_fetcher, threading)
        timeout = 30  # seconds
        while not list_fetcher.get_done() and timeout > 0:
            logging.info("Waiting for all mode threads to finish")
            time.sleep(2)
            timeout -= 2
    else:
        for root_node in root_nodes:
            create_graph_list_mode(main_graph, root_node, list_fetcher, threading)

    # Retrieve the results of the individual threads from list_fetcher
    unique_graph_list, unique_node_lists, component_lists, configuration_list, configuration_space \
        = list_fetcher.get_graph_lists()
    return unique_graph_list, unique_node_lists, component_lists, configuration_list, \
        configuration_space


def check_isolability(all_equipment, component_lists, number_of_faults):
    if number_of_faults == 1:
        plural = False
    else:
        plural = True

    all_component_lists = []
    for root_node in component_lists:
        for configuration in component_lists[root_node]:
            all_component_lists.append(configuration)
    all_equipment_set = set(all_equipment)

    isolable_combinations = []
    non_isolable_combinations = []
    for components in itertools.combinations(all_equipment, number_of_faults):
        alternative_set = set()
        for component_list in all_component_lists:
            if not set(components).issubset(set(component_list)):
                for element in component_list:
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

    non_isolable = set([component for combination in non_isolable_combinations
                        for component in combination])
    isolable = set(all_equipment) - non_isolable
    for equipment in all_equipment:
        if equipment in non_isolable:
            logging.info("Equipment " + equipment + " is not isolable.")
        else:
            logging.info("Equipment " + equipment + " is isolable.")
    return sorted(isolable), sorted(non_isolable)


def check_recoverability(main_graph, all_equipment, component_lists, number_of_faults):
    if number_of_faults == 1:
        plural = False
    else:
        plural = True

    recoverable = []
    non_recoverable = []
    for mode in component_lists:
        mode_available = True
        for components in itertools.combinations(all_equipment, number_of_faults):
            mode_available_per_combination = False
            for component_list in component_lists[mode]:
                if reduce(mul, [component not in component_list for component in components]):
                    logging.debug(f"The mode {get_node_name(main_graph, mode)} is available if "
                                  f"{components} {'have' if plural else 'has'} a fault.")
                    mode_available_per_combination = True
                    break
            if not mode_available_per_combination:
                logging.info(f"The mode {get_node_name(main_graph, mode)} is not available if "
                             f"{components} {'have' if plural else 'has'} a fault.")
                mode_available = False
        logging.info(f"The fault recoverability for mode {get_node_name(main_graph, mode)} is "
                     f"{mode_available}")
        if mode_available:
            recoverable.append(mode)
        else:
            non_recoverable.append(mode)
    return sorted(recoverable), sorted(non_recoverable)


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


def get_fault_probability(graph, node, equipment_fault_probabilities):
    if len(list(graph.successors(node))) == 0:
        # We abort the recursive cycle
        fault_probability = equipment_fault_probabilities[get_node_name(graph, node)]
    else:
        if get_node_name(graph, node).startswith(">="):  # k-out-of-n assembly
            # sub assembly fails for num_available-num_required+1 faults
            required = int(re.findall(r"\d+", get_node_name(graph, node))[0])
            successor_reliabilities = \
                [1 - get_fault_probability(graph, successor, equipment_fault_probabilities)
                 for successor in exclude_guards(graph, graph.successors(node))]
            fault_probability_combinations = \
                [1 - reduce(mul, combination, 1)
                 for combination in itertools.combinations(successor_reliabilities, required)]
            surplus = len(list(exclude_guards(graph, graph.successors(node)))) - required + 1
            fault_probability = reduce(mul, sorted(fault_probability_combinations)[:surplus], 1)
        elif get_node_name(graph, node).startswith("OR"):  # OR assembly
            # sub assembly fails if all members fail
            successor_fault_probabilities = \
                [get_fault_probability(graph, successor, equipment_fault_probabilities)
                 for successor in exclude_guards(graph, graph.successors(node))]
            fault_probability = reduce(mul, successor_fault_probabilities, 1)
        else:  # AND assembly
            # sub assembly fails if one of the members fails
            successor_reliabilities = \
                [1 - get_fault_probability(graph, successor, equipment_fault_probabilities)
                 for successor in exclude_guards(graph, graph.successors(node))]
            reliability = reduce(mul, successor_reliabilities, 1)
            fault_probability = 1 - reliability
    return fault_probability
