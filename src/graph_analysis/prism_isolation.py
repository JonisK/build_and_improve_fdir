import subprocess
import logging
import re
from functools import reduce
from to_precision import to_precision
from graph_analysis.graph_analysis import \
    get_node_name, find_leaf_nodes, get_fault_probability, get_effects, \
    get_assembly_name


class Variable_Handler():
    def __init__(self):
        self.state_variables = {}
        self.configurations = {}
        self.configuration_index = {}

    def add_configuration(self, root_node, configuration_string):
        if root_node in self.configuration_index:
            self.configuration_index[root_node][configuration_string] = len(self.configuration_index[root_node])
        else:
            self.configuration_index[root_node] = {}
            self.configuration_index[root_node][configuration_string] = 0

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

    def convert_to_prism_declaration(self, graph, all_equipment, hidden_variable, include_configurations=False, debug=False):
        declaration_string = "  // equipment states\n"
        declaration_string += "  // 0=available, 1=suspicious\n"
        for component in all_equipment:
            declaration_string += f"  {component}: [0..1]{' init 1' if debug else ''};\n"
        if hidden_variable:
            declaration_string += f"  faulty_component: [0..{len(all_equipment) - 1}]{' init 0' if debug else ''};\n"
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

    def get_configuration_index(self):
        return self.configuration_index


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
               hidden_variable,
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
        configuration_numbers = []
        for assembly in configuration:
            configuration_numbers.append(
                str(variable_handler.convert_to_prism_configuration(assembly, configuration[assembly])))
        action_string += f"_{'_'.join(configuration_numbers)}"
        variable_handler.add_configuration(root_node, '_'.join(configuration_numbers))
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
        # block action if one of the utilized components is faulty
        if hidden_variable:
            all_equipment = list(equipment_fault_probabilities.keys())
            guards += [f"faulty_component!={all_equipment.index(component)}" for component in leaf_name_list]
        if guards:
            action_string += " & ".join(guards)
        else:
            action_string += "true"
        action_string += "\n    -> "
        # probability of success
        fault_probability = get_fault_probability(unique_graph, root_node, equipment_fault_probabilities)
        action_string += f"{to_precision(1-fault_probability, 30, notation='std')}: "
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
        # if not hidden_variable:
        #     for component in leaf_name_list:
        #         action_string += "\n    + "
        #         # probability of failure
        #         action_string += f"{to_precision(fault_probability/len(leaf_name_list), 30, notation='std')}: "
        #         # negative outcome
        #         action_string += f"(faulty_component'={all_equipment.index(component)})"
        #     action_string += ";\n"
        # else:
        #     action_string += "\n    + "
        #     # probability of failure
        #     action_string += f"{to_precision(fault_probability, 30, notation='std')}: "
        #     # negative outcome
        #     action_string += "true;\n"
        action_string += "\n    + "
        # probability of failure
        action_string += f"{to_precision(fault_probability, 30, notation='std')}: "
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


def get_labels(G, all_equipment, hidden_variable, component_to_be_isolated="any"):
    if component_to_be_isolated == "any":
        label_string = "label \"isolation_complete\" =\n      "
        sub_strings = []
        for faulty_component in all_equipment:
            component_strings = []
            for checked_component in all_equipment:
                component_strings.append(f"{checked_component}={'1' if checked_component == faulty_component else '0'}")
            if hidden_variable:
                component_strings.append(f"faulty_component={all_equipment.index(faulty_component)}")
            sub_strings.append(f"({' & '.join(component_strings)})")
        label_string += "\n    | ".join(sub_strings)
        label_string += ";"
    else:
        label_string = f"label \"isolation_complete_{component_to_be_isolated}\" = "
        component_strings = []
        for component in all_equipment:
            component_strings.append(f"{component}={'1' if component == component_to_be_isolated else '0'}")
        label_string += " & ".join(component_strings)
        label_string += ";"
    return label_string


def get_init_string(G, leaf_name_lists, all_equipment, hidden_variable):
    # init_string = "init\n"
    # component_strings = []
    # for component in find_leaf_nodes(G, type='components'):
    #     component_strings.append(f"{get_node_name(G, component)}=1")
    # init_string += " & ".join(component_strings)
    # init_string += "\nendinit\n"

    # init_string = "init\n  true\nendinit\n"

    init_string = "init\n"
    config_strings = []
    for mode in leaf_name_lists:
        for leaf_name_list in leaf_name_lists[mode]:
            init_per_config = []
            for component in all_equipment:
                init_per_config.append(f"{component}={'1' if component in leaf_name_list else '0'}")
            if hidden_variable:
                for faulty_component in leaf_name_list:
                    config_strings.append(f"{' & '.join(init_per_config)} & faulty_component={all_equipment.index(faulty_component)} // {get_node_name(G, mode)}")
            else:
                config_strings.append(f"{' & '.join(init_per_config)} // {get_node_name(G, mode)}")
    init_string += "\n  | ".join(config_strings)
    init_string += "\nendinit\n"

    return init_string


def generate_prism_model(base_directory, filename, G, all_equipment, unique_graph_list, leaf_name_lists, configuration_list,
                         equipment_fault_probabilities, mode_costs, hidden_variable, debug=False):
    trimmed_filename = filename.split('/')[-1].split('.')[0]
    work_directory = base_directory + "/temp/"
    if debug:
        prism_filename = f"{work_directory + trimmed_filename}_debug.prism"
    else:
        prism_filename = f"{work_directory + trimmed_filename}.prism"

    with open(prism_filename, 'w') as prism_file:
        variable_handler = Variable_Handler()
        # first generate the actions so the variable handler knows the number of prism states needed
        actions = []
        costs = []
        for root_node in unique_graph_list:
            logging.info(f"Generate actions for mode {get_node_name(G, root_node)}")
            action, cost = get_action(G, root_node, unique_graph_list[root_node], leaf_name_lists[root_node],
                                      configuration_list[root_node], variable_handler, equipment_fault_probabilities,
                                      mode_costs, hidden_variable)
            actions.append(action)
            costs.append(cost)
        # generate initialization
        print("mdp\n\nmodule sat\n", file=prism_file)
        # declare variables for components, logical states, and configurations
        print(
            variable_handler.convert_to_prism_declaration(G, all_equipment, hidden_variable, include_configurations=False, debug=debug),
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
            print(get_labels(G, all_equipment, hidden_variable, component), file=prism_file)
        print(get_labels(G, all_equipment, hidden_variable, "any"), file=prism_file)
        print("", file=prism_file)
        # init
        if not debug:
            print(get_init_string(G, leaf_name_lists, all_equipment, hidden_variable), file=prism_file)
    logging.info(f"Generated prism model {prism_filename}")
    return variable_handler.get_configuration_index()


def get_configuration_index(G, unique_graph_list, leaf_name_lists, configuration_list,
                            equipment_fault_probabilities, mode_costs):
    variable_handler = Variable_Handler()
    for root_node in unique_graph_list:
        get_action(G, root_node, unique_graph_list[root_node], leaf_name_lists[root_node],
                   configuration_list[root_node], variable_handler, equipment_fault_probabilities,
                   mode_costs, hidden_variable=False)
    return variable_handler.get_configuration_index()


def generate_props(base_directory, filename, all_equipment):
    trimmed_filename = filename.split('/')[-1].split('.')[0]
    work_directory = base_directory + "temp/"
    with open(f"{work_directory + trimmed_filename}.props", 'w') as props_file:
        for component in all_equipment:
            print(f"\"{component}\": Rmin=? [ F \"isolation_complete_{component}\" ]", file=props_file)
        print(f"\"any\": Rmin=? [ F \"isolation_complete\" ]", file=props_file)
        print(f"\"sparse\": Pmax=? [F \"isolation_complete\"]", file=props_file)
    logging.info(f"Generated props file {work_directory + trimmed_filename}.props")


def run_prism(base_directory, filename, all_equipment, components="all"):
    if components == "all":
        isolability = {}
        isolation_cost = {}
        for component in all_equipment:
            logging.info(f"Check isolability for {component}")
            isolability[component], isolation_cost[component] = run_prism_helper(base_directory, filename, component)
            logging.info(f"Result for {component}: {isolability[component]}, Cost: {isolation_cost[component]}")
    elif components == "any":
        logging.info(f"Check isolability for any component")
        isolability, isolation_cost = run_prism_helper(base_directory, filename, "any")
        logging.info(f"Result for any component: {isolability}, Cost: {isolation_cost}")
    logging.info(f"Check for isolability done")
    return isolability, isolation_cost


def run_prism_helper(base_directory, filename, component):
    trimmed_filename = filename.split('/')[-1].split('.')[0]
    work_directory = base_directory + "temp/"
    prism_path = "prism/bin/prism"
    pattern = r'Result: \S*'
    args = f"{base_directory + prism_path} {work_directory + trimmed_filename}.prism {work_directory + trimmed_filename}.props -prop {component} -explicit -javamaxmem 50g"
    # logging.info(f"Command: prism {trimmed_filename}.prism {trimmed_filename}.props -prop {component} -explicit -javamaxmem 50g")
    result = subprocess.run(args.split(" "), stdout=subprocess.PIPE, text=True)
    # logging.info(result.stdout)
    prism_result = re.findall(pattern, result.stdout)
    if prism_result:
        logging.debug(f"{component} - {prism_result[0]}")
        first_number = re.findall("\d+.\d+", prism_result[0])
        if first_number:
            isolation_cost = float(first_number[0])
            isolability = True
        else:
            isolation_cost = float('inf')
            isolability = False
    else:
        isolation_cost = float('inf')
        isolability = False
        logging.error(f"{component} - Error!")
        logging.error(result.stdout)
    return isolability, isolation_cost


def export_strategy(base_directory, filename, shell, component=None):
    trimmed_filename = filename.split('/')[-1].split('.')[0]
    if component:
        trimmed_filename_component = filename.split('/')[-1].split('.')[0] + "_" + component
        property = component
    else:
        property = "sparse"
    work_directory = base_directory + "temp/"
    prism_path = "prism/bin/prism"
    args = f"{base_directory + prism_path} {work_directory + trimmed_filename}.prism {work_directory + trimmed_filename}.props -prop {property} -explicit -javamaxmem 50g -exportstrat {work_directory}strategy_{trimmed_filename_component}.prism:type=actions -exportstates {work_directory}strategy_{trimmed_filename_component}_states.prism"
    logging.info(f"Command: {args}")
    # result = subprocess.run(args.split(" "), stdout=subprocess.PIPE, text=True)
    shell(args + "\n")
    logging.info(f"Generated strategy file {work_directory}strategy_{trimmed_filename}.prism")
    # return result
