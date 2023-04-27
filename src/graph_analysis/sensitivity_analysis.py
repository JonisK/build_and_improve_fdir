import numpy as np
from graph_analysis.graph_analysis import get_node_id, get_fault_probability
import pandas as pd
from to_precision import to_precision
import logging

sig_figures = 4


def get_mode_gradients(G, equipment_fault_probabilities, mode_costs):
    # row_index_to_equipment_name = {key: index for index, key in enumerate(equipment_fault_probabilities)}
    row_index_to_equipment_name = {index: key for index, key in enumerate(equipment_fault_probabilities)}
    column_index_to_mode_name = {index: key for index, key in enumerate(mode_costs)}

    mode_gradients = np.zeros([len(equipment_fault_probabilities), len(mode_costs)])
    for row_index, row in enumerate(mode_gradients):
        for column_index, element in enumerate(row):
            node_id = get_node_id(G, column_index_to_mode_name[column_index])
            # node_id = get_node_id(G, row_index_to_equipment_name[row_index])
            modified_fault_probabilities = equipment_fault_probabilities.copy()
            modified_fault_probabilities[row_index_to_equipment_name[row_index]] *= 10
            # partial derivative of the fault probability per mode and component
            gradient = (get_fault_probability(G, node_id, modified_fault_probabilities) - get_fault_probability(G, node_id, equipment_fault_probabilities)) / get_fault_probability(G, node_id, equipment_fault_probabilities)
            mode_gradients[row_index, column_index] = gradient
            # increased_mode_cost[row_index, column_index] = get_fault_probability(G, node_id, modified_fault_probabilities)
    return mode_gradients


def get_sensitivity_analysis(G, equipment_fault_probabilities, mode_costs):
    mode_gradients = get_mode_gradients(G, equipment_fault_probabilities, mode_costs)
    row_index_to_equipment_name = {index: key for index, key in enumerate(equipment_fault_probabilities)}
    column_index_to_mode_name = {index: key for index, key in enumerate(mode_costs)}

    message = "Sensitivity matrix\n------------------\n"
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 20, 'display.width',
                           1000):  # more options can be specified also
        df = pd.DataFrame(mode_gradients,
                          columns=mode_costs.keys(),
                          index=equipment_fault_probabilities.keys())
        message += str(df.apply(lambda x: [to_precision(y, sig_figures, preserve_integer=True) for y in x]))
        message += "\n\n"

    for index, column in enumerate(mode_gradients.T):
        message += f"Mode {column_index_to_mode_name[index]} is most sensitive to the component {row_index_to_equipment_name[column.argmax()]}\n"
    message += "\n"

    for index, row in enumerate(mode_gradients):
        message += f"Component {row_index_to_equipment_name[index]} has most effect on mode {column_index_to_mode_name[row.argmax()]}\n"

    return message



def get_uncertainty_interval(G, equipment_fault_probabilities, equipment_fault_probabilities_lower_bound,
                             equipment_fault_probabilities_upper_bound, mode_costs):
    row_index_to_equipment_name = {index: key for index, key in enumerate(equipment_fault_probabilities)}
    column_index_to_mode_name = {index: key for index, key in enumerate(mode_costs)}

    best_case = np.zeros([1, len(mode_costs)])
    worst_case = np.zeros([1, len(mode_costs)])
    uncertainty_interval = np.zeros([len(equipment_fault_probabilities_lower_bound), len(mode_costs)])

    for row_index, row in enumerate(uncertainty_interval):
        for column_index, element in enumerate(row):
            # logging.debug(f"Mode name: {column_index_to_mode_name[column_index]}")
            # logging.debug(f"Component name: {row_index_to_equipment_name[row_index]}")
            node_id = get_node_id(G, column_index_to_mode_name[column_index])
            if row_index == 0:
                best_case[0, column_index] = get_fault_probability(G, node_id,
                                                                   equipment_fault_probabilities_lower_bound)
                worst_case[0, column_index] = get_fault_probability(G, node_id,
                                                                    equipment_fault_probabilities_upper_bound)
            # Examine best case for this component
            these_fault_probabilities = equipment_fault_probabilities.copy()
            these_fault_probabilities[row_index_to_equipment_name[row_index]] = \
                equipment_fault_probabilities_lower_bound[row_index_to_equipment_name[row_index]]
            best_case_per_component = get_fault_probability(G, node_id, these_fault_probabilities)
            # logging.debug(f"Best case fault probability: {best_case_per_component}")
            # Examine worst case
            these_fault_probabilities[row_index_to_equipment_name[row_index]] = \
                equipment_fault_probabilities_upper_bound[row_index_to_equipment_name[row_index]]
            worst_case_per_component = get_fault_probability(G, node_id, these_fault_probabilities)
            # logging.debug(f"Worst case fault probability: {worst_case_per_component}")
            # Baseline fault probability for normalization
            baseline = get_fault_probability(G, node_id, equipment_fault_probabilities)
            # Normalized uncertainty interval
            uncertainty_interval[row_index, column_index] = (worst_case_per_component - best_case_per_component) / baseline

    return uncertainty_interval, best_case, worst_case


# def get_best_case_probabilities(G, equipment_fault_probabilities_lower_bound, mode_costs):
#     row_index_to_equipment_name = {index: key for index, key in enumerate(equipment_fault_probabilities_lower_bound)}
#     column_index_to_mode_name = {index: key for index, key in enumerate(mode_costs)}


def get_uncertainty_propagation(G, equipment_fault_probabilities, equipment_fault_probabilities_lower_bound,
                                equipment_fault_probabilities_upper_bound, mode_costs):
    row_index_to_equipment_name = {index: key for index, key in enumerate(equipment_fault_probabilities_lower_bound)}
    column_index_to_mode_name = {index: key for index, key in enumerate(mode_costs)}

    uncertainty_interval, best_case, worst_case = get_uncertainty_interval(G, equipment_fault_probabilities,
                                                                           equipment_fault_probabilities_lower_bound,
                                                                           equipment_fault_probabilities_upper_bound,
                                                                           mode_costs)

    message = "Uncertainty interval per component\n----------------------------------\n"
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 20, 'display.width',
                           1000):  # more options can be specified also
        df = pd.DataFrame(uncertainty_interval,
                          columns=mode_costs.keys(),
                          index=equipment_fault_probabilities_lower_bound.keys())
        message += str(df.apply(lambda x: [to_precision(y, sig_figures, preserve_integer=True) for y in x]))
        message += "\n\n"

    for index, column in enumerate(uncertainty_interval.T):
        message += f"Mode {column_index_to_mode_name[index]}'s uncertainty interval mostly depends on {row_index_to_equipment_name[column.argmax()]}'s uncertainty interval\n"
    message += "\n"

    for index, row in enumerate(uncertainty_interval):
        message += f"Component {row_index_to_equipment_name[index]}'s uncertainty interval has most effect on mode {column_index_to_mode_name[row.argmax()]}'s uncertainty interval\n"
    message += "\n\n"

    message += "Best case\n---------\n"
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 20, 'display.width',
                           1000):  # more options can be specified also
        df = pd.DataFrame(best_case,
                          columns=mode_costs.keys())
        message += str(df.apply(lambda x: [to_precision(y, sig_figures, preserve_integer=True) for y in x]))
        message += "\n\n"

    message += "Worst case\n----------\n"
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 20, 'display.width',
                           1000):  # more options can be specified also
        df = pd.DataFrame(worst_case,
                          columns=mode_costs.keys())
        message += str(df.apply(lambda x: [to_precision(y, sig_figures, preserve_integer=True) for y in x]))

    return message
