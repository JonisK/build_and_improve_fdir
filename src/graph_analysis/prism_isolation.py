import subprocess
import logging
import re
from functools import reduce
from graph_analysis.graph_analysis import \
    get_node_name, find_leaf_nodes, get_fault_probability, get_effects, \
    get_assembly_name


class Variable_Handler():
    def __init__(self):
        self.state_variables = {}
        self.configurations = {}

    def convert_to_prism_configuration(self, configuration, value):
        if not configuration in self.configurations:
            self.configurations[configuration] = []
        if not value in self.configurations[configuration]:
            self.configurations[configuration].append(value)
        # logging.debug(f"{self.configurations=}")
        return self.configurations[configuration].index(value)

    def convert_to_prism_guard(self, guard):
        guard_list = []
        for condition in guard.split('&'):
            # get text before the equal sign
            split_condition = condition.split('=')
            # variable = regex.findall(r"(?<=\=\s*)\w+", condition)[0]
            variable = split_condition[0].strip()
            # get text after the equal sign
            # value = regex.findall(r"\w+(?=\s*\=)", condition)[0]
            value = split_condition[1].strip()
            # check if the key already exists in the state_variables
            if variable in self.state_variables:
                # check if the value is known already
                if not value in self.state_variables[variable]:
                    # add the value to state_variables
                    self.state_variables[variable].append(value)
            else:
                # add the variable to state_variables
                self.state_variables[variable] = ['uninitialized', value]

            # get the index of the value
            index = self.state_variables[variable].index(value)

            guard_list.append(f"{variable}={index}")
        return " & ".join(guard_list)

    def convert_to_prism_outcome(self, effects):
        outcome_list = []
        for effect in effects:
            # get text before the equal sign
            split_effect = effect.split('=')
            # variable = regex.findall(r"(?<=\=\s*)\w+", effect)[0]
            variable = split_effect[0].strip()
            # get text after the equal sign
            # value = regex.findall(r"\w+(?=\s*\=)", effect)[0]
            value = split_effect[1].strip()
            # check if the key already exists in the state_variables
            if variable in self.state_variables:
                # check if the value is known already
                if not value in self.state_variables[variable]:
                    # add the value to state_variables
                    self.state_variables[variable].append(value)
            else:
                # add the variable to state_variables
                self.state_variables[variable] = ['uninitialized', value]

            # get the index of the value
            index = self.state_variables[variable].index(value)
            outcome_list.append(f"({variable}\'={index})")
        return " & ".join(outcome_list)

    def convert_to_prism_declaration(self, graph, all_equipment, include_configurations=False, debug=False):
        declaration_string = "  // equipment states\n"
        declaration_string += "  // 0=available, 1=suspicious\n"
        for component in all_equipment:
            declaration_string += f"  {component}: [0..1]{' init 1' if debug else ''};\n"
        if self.state_variables:
            declaration_string += "\n  // planning states\n"
        for variable in self.state_variables:
            variable_comments = [f"{index}={value}" for index, value in enumerate(self.state_variables[variable])]
            declaration_string += f"  // {', '.join(variable_comments)}\n"
            declaration_string += f"  {variable}: [0..{len(self.state_variables[variable]) - 1}]{' init 0' if debug else ''};\n"
        declaration_string += "\n  // configurations\n"
        for assembly in self.configurations:
            logging.debug(f"{assembly=}: {self.configurations[assembly]}")
            assembly_comments = [f"{index}={value}" for index, value in enumerate(self.configurations[assembly])]
            declaration_string += f"  // {get_assembly_name(graph, assembly)}: {', '.join(assembly_comments)}\n"
            assembly_comments = [f"({index},)={get_node_name(graph, successor)}" for index, successor in
                                 enumerate(graph.successors(assembly))]
            # assembly_comments = [f"{value}={list(G.successors(assembly))[value]}" for value in self.configurations[assembly]]
            declaration_string += f"  //   {', '.join(assembly_comments)}\n"
            if include_configurations:
                declaration_string += f"  {get_assembly_name(graph, assembly)}: [0..{len(self.configurations[assembly]) - 1}]{' init 0' if debug else ''};\n"
        return declaration_string


# ilen() is written by Al Hoo, published on stackoverflow
# https://stackoverflow.com/questions/19182188
def ilen(iterable):
    return reduce(lambda sum, element: sum + 1, iterable, 0)


def get_action(G,
               root_node,
               unique_graph_per_root_node,
               leaf_name_list_per_root_node,
               configuration_list_per_root_node,
               variable_handler,
               equipment_fault_probabilities,
               mode_costs,
               include_configurations=False):
    action_strings = ""
    cost_strings = ""
    for unique_graph, configuration, leaf_name_list in \
            zip(unique_graph_per_root_node,
                configuration_list_per_root_node,
                leaf_name_list_per_root_node):
        logging.debug(f"[{get_node_name(G, root_node)}] "
                      f"Write action for {configuration=}")
        logging.debug(f"{leaf_name_list=}")
        # name
        action_string = f"  [{get_node_name(G, root_node)}"
        for assembly in configuration:
            action_string += f"_{variable_handler.convert_to_prism_configuration(assembly, configuration[assembly])}"
        action_string += "] "
        cost_strings += action_string + f"true: {mode_costs[get_node_name(G, root_node)]};\n"
        # guard
        guards = []
        # configuration
        if include_configurations:
            for assembly in configuration:
                guards.append(f"{get_assembly_name(G, assembly)}="
                              f"{variable_handler.convert_to_prism_configuration(assembly, configuration[assembly])}")
        # logical guards
        logical_guards = [get_node_name(G, node) for node in find_leaf_nodes(G, root_node=root_node, type='guards')]
        guards += [variable_handler.convert_to_prism_guard(guard) for guard in logical_guards]
        if guards:
            action_string += " & ".join(guards)
        else:
            action_string += "true"
        action_string += "\n    -> "
        # probability of success
        fault_probability = get_fault_probability(unique_graph, root_node, equipment_fault_probabilities)
        action_string += f"{1 - fault_probability:.4f}: "
        # positive outcome, components
        positive_outcomes = []
        for component in leaf_name_list:
            positive_outcomes.append(f"({component}\'=0)")
        # positive outcome, variable changes
        positive_outcomes.append(variable_handler.convert_to_prism_outcome(get_effects(G, root_node)))
        # if there are no effects, we might add an empty string that needs to be filtered
        if ilen(filter(None, positive_outcomes)):
            action_string += " & ".join(filter(None, positive_outcomes))
        else:
            action_string += "true"
        action_string += "\n    + "
        # probability of failure
        action_string += f"{fault_probability:.4f}: "
        # negative outcome
        action_string += "true;\n"
        action_strings += action_string
    return action_strings, cost_strings


def get_cost(G, costs):
    cost_string = "rewards \"total_cost\"\n"
    for cost in costs:
        cost_string += cost
    cost_string += "endrewards\n"
    return cost_string


def get_labels(G, all_equipment, component_to_be_isolated):
    label_string = f"label \"isolation_complete_{component_to_be_isolated}\" = "
    component_strings = []
    for component in all_equipment:
        component_strings.append(f"{component}={'1' if component == component_to_be_isolated else '0'}")
    label_string += " & ".join(component_strings)
    label_string += ";"
    return label_string


def get_init_string(G):
    init_string = "init\n"
    component_strings = []
    for component in find_leaf_nodes(G, type='components'):
        component_strings.append(f"{get_node_name(G, component)}=1")
    init_string += " & ".join(component_strings)
    init_string += "\nendinit\n"
    return init_string


def generate_prism_model(filename, G, all_equipment, unique_graph_list, leaf_name_lists, configuration_list,
                         equipment_fault_probabilities, mode_costs, debug=False):
    with open(filename, 'w') as prism_file:
        variable_handler = Variable_Handler()
        # first generate the actions so the variable handler knows the number of prism states needed
        actions = []
        costs = []
        for root_node in unique_graph_list:
            logging.info(f"Generate action for node {get_node_name(G, root_node)}")
            action, cost = get_action(G, root_node, unique_graph_list[root_node], leaf_name_lists[root_node],
                                      configuration_list[root_node], variable_handler, equipment_fault_probabilities,
                                      mode_costs)
            actions.append(action)
            costs.append(cost)
        # generate initialization
        print("mdp\n\nmodule sat\n", file=prism_file)
        # declare variables for components, logical states, and configurations
        print(variable_handler.convert_to_prism_declaration(G, all_equipment, include_configurations=False, debug=debug),
              file=prism_file)
        print("\n", file=prism_file)
        # actions
        for action in actions:
            print(action, file=prism_file)
        print("endmodule\n", file=prism_file)
        # rewards
        print(get_cost(G, costs), file=prism_file)
        # labels
        for component in all_equipment:
            print(get_labels(G, all_equipment, component), file=prism_file)
        print("", file=prism_file)
        # init
        if not debug:
            print(get_init_string(G), file=prism_file)


def generate_props(filename, all_equipment):
    with open(f"{filename[:-6]}.props", 'w') as props_file:
        for component in all_equipment:
            print(f"\"{component}\": Rmin=? [ F \"isolation_complete_{component}\" ]", file=props_file)


def run_prism(filename, all_equipment):
    isolability = {}
    isolation_cost = {}
    base_path = "/home/jonis/git/build_and_improve_fdir/"
    prism_path = "prism/bin/prism"
    pattern = r'Result: \S*'
    for component in all_equipment:
        logging.info(f"Check isolability for {component}")
        args = f"{base_path + prism_path} {filename} {filename[:-6]}.props -prop {component} -explicit -javamaxmem 50g"
        result = subprocess.run(args.split(" "), stdout=subprocess.PIPE, text=True)
        prism_result = re.findall(pattern, result.stdout)
        if prism_result:
            logging.debug(f"{component} - {prism_result[0]}")
            first_number = re.findall("\d+.\d+", prism_result[0])
            if first_number:
                isolation_cost[component] = float(first_number[0])
                isolability[component] = True
            else:
                isolation_cost[component] = float('inf')
                isolability[component] = False
        else:
            isolation_cost[component] = float('inf')
            isolability[component] = False
            logging.error(f"{component} - Error!")
            logging.error(result.stdout)
    logging.info(f"Check for isolability done")
    return isolability, isolation_cost
