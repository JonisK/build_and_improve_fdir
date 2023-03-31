from ctypes import CDLL, c_float
import logging
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from graph_analysis.graph_analysis import get_node_id


class Strategy_Helper():
    def __init__(self, G, leaf_name_lists, action_names, filename):
        self.fault_isolation_graph = nx.DiGraph()

        self.G = G
        self.leaf_name_lists = leaf_name_lists
        self.action_names = action_names

        lib_object = CDLL(filename)
        self.function_object = lib_object.classify
        self.function_object.restype = c_float

    def execute_isolation_strategy(self, current_state, guess):
        state_array = c_float * (len(current_state) + 1)
        # state_array = c_float * len(current_state)
        state_list = state_array()
        for index, component in enumerate(current_state):
            if current_state[component] == 'suspicious':
                state_list[index] = 1
            else:
                state_list[index] = 0
        state_list[len(current_state)] = guess
        # logging.info(f"Result for input {list(state_list)}: {int(self.function_object(state_list))}")
        # logging.info(self.action_names[int(self.function_object(state_list))])

        return self.action_names[int(self.function_object(state_list))]

    def add_initial_node_to_graph(self, initial_mode_and_configuration):
        self.fault_isolation_graph.add_node(initial_mode_and_configuration)

    def add_edge_to_graph(self, previous_mode_and_configuration, next_mode_and_configuration, label):
        self.fault_isolation_graph.add_edge(previous_mode_and_configuration, next_mode_and_configuration, label=label)

    def get_graph(self):
        return self.fault_isolation_graph


def get_action_names(input_strategy):
    logging.info(f"Read action names from {input_strategy}")
    action_names = []

    with open(input_strategy) as source:
        lines = source.readlines()
        for line in lines:
            current_action = line.split(':')[1].strip()
            if not current_action in action_names:
                action_names.append(current_action)

    return action_names


def propagate_state(configuration_index, strategy_helper, all_equipment, previous_mode_and_configuration, previous_state, guess, depth):
    if depth < 5:
        depth += 1
        [previous_mode, configuration_number] = previous_mode_and_configuration.split('_', 1)
        # logging.info(f"{previous_mode=}, {configuration_number=}")
        root_node = get_node_id(strategy_helper.G, previous_mode)
        if configuration_index:
            index = configuration_index[root_node][configuration_number]
        else:
            index = int(configuration_number)
        used_components = strategy_helper.leaf_name_lists[root_node][index]

        positive_outcome = previous_state.copy()
        # negative_outcome = previous_state.copy()

        for component in previous_state:
            if component in used_components:
                positive_outcome[component] = 'available'
            else:
                pass
                # negative_outcome[component] = 'available'

        if list(positive_outcome.values()).count('suspicious') == 1:
            # logging.info(f"[{previous_mode_and_configuration} Positive {guess}] Fault isolation done with state {positive_outcome}")
            faulty_component = [key for key, value in positive_outcome.items() if value == 'suspicious'][0]
            strategy_helper.add_edge_to_graph(previous_mode_and_configuration, f"Fault in {faulty_component}", label=f'positive {guess}')
        elif not positive_outcome == previous_state:
            next_mode_and_configuration = strategy_helper.execute_isolation_strategy(positive_outcome, guess)
            strategy_helper.add_edge_to_graph(previous_mode_and_configuration, next_mode_and_configuration, label=f'positive {guess}')
            # logging.info(
            #     f"[{previous_mode_and_configuration} Positive {guess}] Next mode is {next_mode_and_configuration} with state {positive_outcome}")
            propagate_state(configuration_index, strategy_helper, all_equipment, next_mode_and_configuration, positive_outcome, guess, depth)
        else:
            pass
            # logging.warning(f"[{previous_mode_and_configuration} Positive {guess}] Deadlock at state {previous_state}.")

        # else:
        #     next_modes_and_configurations = set()
        #     for next_guess in range(guess, len(previous_state)):
        #         next_modes_and_configurations.add(strategy_helper.execute_isolation_strategy(positive_outcome, next_guess))
        #     for next_mode_and_configuration in next_modes_and_configurations:
        #         if (positive_outcome == previous_state) or (next_mode_and_configuration == previous_mode_and_configuration):
        #             # try again with a different guess
        #             if guess < len(previous_state) - 1:
        #                 propagate_state(configuration_index, strategy_helper, all_equipment, next_mode_and_configuration, positive_outcome, guess + 1)
        #             else:
        #                 logging.warning(f"[{previous_mode_and_configuration} Positive {guess}] Deadlock at state {previous_state}.")
        #         else:
        #             strategy_helper.add_edge_to_graph(previous_mode_and_configuration, next_mode_and_configuration, label='positive')
        #             logging.info(
        #                 f"[{previous_mode_and_configuration} Positive {guess}] Next mode is {next_mode_and_configuration} with state {positive_outcome}")
        #             propagate_state(configuration_index, strategy_helper, all_equipment, next_mode_and_configuration, positive_outcome, 0)

        # if list(negative_outcome.values()).count('suspicious') == 1:
        #     logging.info(f"[{previous_mode_and_configuration} Negative] Fault isolation done with state {negative_outcome}")
        #     faulty_component = [key for key, value in negative_outcome.items() if value == 'suspicious'][0]
        #     strategy_helper.add_edge_to_graph(previous_mode_and_configuration, f"Fault in {faulty_component}", label='negative')
        # elif not negative_outcome == previous_state:
        #     next_mode_and_configuration = strategy_helper.execute_isolation_strategy(negative_outcome, guess)
        #     strategy_helper.add_edge_to_graph(previous_mode_and_configuration, next_mode_and_configuration, label='negative')
        #     logging.info(
        #         f"[{previous_mode_and_configuration} Negative] Next mode is {next_mode_and_configuration} with state {negative_outcome}")
        #     propagate_state(configuration_index, strategy_helper, all_equipment, next_mode_and_configuration, negative_outcome, guess)
        # else:
        #     logging.warning(f"[{previous_mode_and_configuration} Negative] Deadlock at state {previous_state}.")

        # Negative outcome. No change to the state but we will change the guess.
        next_modes_and_configurations = set()
        guess_dict = {}
        for component in used_components:
            next_guess = all_equipment.index(component)
            # if not next_guess == guess:
            mode_and_configuration = strategy_helper.execute_isolation_strategy(previous_state, next_guess)

            if not mode_and_configuration == previous_mode_and_configuration:
                [candidate_mode, candidate_configuration_number] = mode_and_configuration.split('_', 1)
                # logging.info(f"{candidate_mode=}, {candidate_configuration_number=}")
                root_node = get_node_id(strategy_helper.G, candidate_mode)
                index = configuration_index[root_node][candidate_configuration_number]
                used_components = strategy_helper.leaf_name_lists[root_node][index]
                candidate_positive_outcome = previous_state.copy()
                for component in previous_state:
                    if component in used_components:
                        candidate_positive_outcome[component] = 'available'

                if not candidate_positive_outcome == previous_state:
                    guess_dict[mode_and_configuration] = next_guess
                    next_modes_and_configurations.add(mode_and_configuration)
        for next_mode_and_configuration in next_modes_and_configurations:
            guess = guess_dict[next_mode_and_configuration]
            strategy_helper.add_edge_to_graph(previous_mode_and_configuration, next_mode_and_configuration,
                                              label=f'negative {guess}')
            # logging.info(f"[{previous_mode_and_configuration} Negative {guess}] Next mode is {next_mode_and_configuration} with state {previous_state}")
            propagate_state(configuration_index, strategy_helper, all_equipment, next_mode_and_configuration,
                            previous_state, guess, depth)
    else:
        pass
        # logging.warning(f"[{previous_mode_and_configuration} {guess}] Reached maximum depth. Aborting")


def get_strategy_graph(G,
                       leaf_name_lists,
                       configuration_index,
                       tree_filename,
                       strategy_filename,
                       initial_state,
                       all_equipment,
                       graph_filename):
    action_names = get_action_names(strategy_filename)
    strategy_helper = Strategy_Helper(G, leaf_name_lists, action_names, tree_filename)
    # initial_state = {component: 'suspicious' for component in all_equipment}

    initial_mode_and_configuration = strategy_helper.execute_isolation_strategy(initial_state, 0)
    strategy_helper.add_initial_node_to_graph(initial_mode_and_configuration)

    # propagate_state will recursively call itself until having explored the whole tree
    propagate_state(configuration_index, strategy_helper, all_equipment, initial_mode_and_configuration, initial_state, 0, 0)

    # after the tree has been completed, we fetch it and write it to a .dot file
    fault_isolation_graph = strategy_helper.get_graph()
    write_dot(fault_isolation_graph, graph_filename)