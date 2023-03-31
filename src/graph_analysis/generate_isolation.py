import json
import os
from graph_analysis.graph_analysis import get_node_name


class StrategyWriter:
    state_number = 0

    def __init__(self, states_filename, strategy_filename):
        self.strategy = open(strategy_filename, 'w')
        self.states = open(states_filename, 'w')

    def write_header(self, all_equipment):
        # write header
        header_states = "("
        for equipment in all_equipment[:-1]:
            header_states += equipment + ","
        header_states += all_equipment[-1] + ")"
        print(header_states, file=self.states)

    def write_action(self, state, action):
        # strategy file
        line_strategy = str(self.state_number) + ":" + action
        print(line_strategy, file=self.strategy)
        # states file
        line_states = str(self.state_number) + ":" + state
        print(line_states, file=self.states)
        self.state_number += 1

    def close(self):
        self.strategy.close()
        self.states.close()


def get_configuration_lists(component_lists, all_equipment):
    configuration_lists = {}
    for mode in component_lists:
        temp_list = []
        for index, configuration in enumerate(component_lists[mode]):
            # generate configuration
            indices = [0] * 23
            # print(indices)
            for inner_index, equipment in enumerate(all_equipment):
                if equipment in configuration:
                    indices[inner_index] = 1
                else:
                    indices[inner_index] = 0
            temp_list.append(indices)
        configuration_lists[mode] = temp_list
    return configuration_lists


def and_list(list1, list2):
    output_list = []
    for (left, right) in zip(list1, list2):
        if left and right:
            output_list.append(1)
        else:
            output_list.append(0)
    return output_list


def find_best_configuration_weights(G, configuration_lists, suspicious_equipment_list, mode_times, writer, verbose):
    print("Configuration: " + str(suspicious_equipment_list)) if verbose else None
    print(len(suspicious_equipment_list)) if verbose else None
    print("Number of suspects: " + str(sum(suspicious_equipment_list))) if verbose else None
    ideal_difference = sum(suspicious_equipment_list) / 2.0
    best_cost = 1000.0
    best_mode = "none"
    best_configuration = -1
    # make the binary tree for this configuration
    # find the optimum configuration for bisecting the array of suspicious components
    for mode in configuration_lists:
        # mode = 'N9'
        for index, configuration_list in enumerate(configuration_lists[mode]):
            if configuration_list != suspicious_equipment_list:
                # print("Investigating configuration #" + str(index) + ": " + str(configuration_list))
                intersection_length = sum(and_list(configuration_list, suspicious_equipment_list))
                intersection_length_deviation = abs(intersection_length-ideal_difference)
                # print("Intersection length: " + str(intersection_length))
                total_cost = 1.0 * intersection_length_deviation + 0.0001 * mode_times[mode]
                if total_cost < best_cost:
                    # found a new best fit
                    # best_difference = abs(intersection_length-ideal_difference)
                    # print("New best fit with cost of " + str(total_cost))
                    best_mode = get_node_name(G, mode)
                    best_cost = total_cost
                    best_configuration = index
                    best_configuration_list = configuration_list
    if best_mode == "none":
        print("Search for a subsequent mode unsuccessful") if verbose else None
    else:
        print("We select mode " + best_mode + ", configuration #" + str(best_configuration) + " with a cost of " + str(best_cost) + " \n\tfor permutation " + str(suspicious_equipment_list)) if verbose else None
    writer.write_action(str(suspicious_equipment_list).replace('[', '(').replace(']', ')').replace(' ', ''),
                        best_mode + '_' + str(best_configuration))
    return best_configuration_list


def trim_suspects(suspicious_equipment_list, best_configuration_list, outcome, verbose):
    print("Trim " + outcome) if verbose else None
    print(suspicious_equipment_list) if verbose else None
    print(best_configuration_list) if verbose else None
    trimmed_configuration = []
    for (suspicious_equipment_bit, best_configuration_bit) in zip(suspicious_equipment_list, best_configuration_list):
        if suspicious_equipment_bit:
            if best_configuration_bit:
                if outcome == 'positive':
                    trimmed_configuration.append(0)
                else:
                    trimmed_configuration.append(1)
            else:
                if outcome == 'positive':
                    trimmed_configuration.append(1)
                else:
                    trimmed_configuration.append(0)
        else:
            trimmed_configuration.append(0)
    print(trimmed_configuration) if verbose else None
    return trimmed_configuration


def traverse_binary_tree_weights(G, configuration_lists, suspicious_equipment_list, mode_times, writer, verbose):
    best_configuration_list = find_best_configuration_weights(G,
                                                              configuration_lists,
                                                              suspicious_equipment_list,
                                                              mode_times,
                                                              writer,
                                                              verbose)
    for outcome in ['positive', 'negative']:
        # the next configurations to be considered are the outcomes of best_configuration_list
        new_suspicious_equipment_list = trim_suspects(suspicious_equipment_list,
                                                      best_configuration_list,
                                                      outcome,
                                                      verbose)
        if new_suspicious_equipment_list == suspicious_equipment_list:
            print("We lack a suitable mode to bisect the suspicious equipment list")
            break
        if sum(new_suspicious_equipment_list) == 1:
            # We are done. The faulty equipment has been found.
            print("Done")  # if verbose else None
        elif sum(new_suspicious_equipment_list) == 0:
            # This is not possible
            print("Hit a dead-end") if verbose else None
        else:
            # Keep searching
            traverse_binary_tree_weights(G, configuration_lists, new_suspicious_equipment_list, mode_times, writer, verbose)


def generate_config_json(all_equipment, filename):
    config = {"x_column_types": {"categorical": []},
              "y_column_types": {},
              "x_column_names": list(all_equipment),
              "x_category_names": {}
              }
    config["x_column_types"]["categorical"] = list(range(len(all_equipment)))
    for equipment in all_equipment:
        config["x_category_names"][equipment] = ["not_available", "available"]

    with open(filename, "w") as text_file:
        print(json.dumps(config, indent=4), file=text_file)


def run_dtcontrol(mode_switcher_strategy_filename, verbose):
    command = "dtcontrol --input " + mode_switcher_strategy_filename + \
               " --use-preset avg --rerun --benchmark-file benchmark.json"
    if not verbose:
        command += " > /dev/null 2>&1"
    os.system(command)
