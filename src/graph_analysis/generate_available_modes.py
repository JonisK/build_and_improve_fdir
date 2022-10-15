from graph_analysis.graph_analysis import find_leaf_nodes, get_node_name, get_layers


def get_guards(G, unique_graph_list, verbose):
    guards = {}
    for feature in unique_graph_list:
        guard = ""
        for configuration_index, configuration in enumerate(unique_graph_list[feature]):
            layers = get_layers(configuration)
            leaves = sorted(find_leaf_nodes(configuration, layers))
            guard += "("
            for leaf_index, leaf in enumerate(leaves):
                guard += get_node_name(G, leaf)
                if leaf_index < len(leaves)-1:
                    guard += " && "
            guard += ")"
            if configuration_index < len(unique_graph_list[feature])-1:
                guard += " || "
        print(guard) if verbose else None
        guards[feature] = guard
    return guards


def get_initialization(unique_graph_list, all_equipment, verbose):
    initialization = """#include <stdio.h>
#include <stdbool.h>

void available(const unsigned char x[], unsigned char* y);

void available(const unsigned char x[], unsigned char* y) {"""
    # for feature in unique_graph_list:
    #     initialization += "    unsigned char " + get_node_name(G, feature) + " = false;\n"
    initialization += "\n    // read the equipment status\n"
    for index, item in enumerate(all_equipment):
        initialization += "    unsigned char " + item + " = x[" + str(index) + "];\n"

    initialization += "\n    // initialize all " + str(len(unique_graph_list)) + " output fields with false\n"
    initialization += "    for (int i = 0; i<" + str(len(unique_graph_list)) + "; ++i) {\n"
    initialization += "        y[i] = false;\n"
    initialization += "    }\n"
    print(initialization) if verbose else None
    return initialization


def get_branches(G, guards, verbose):
    branches = "    // ---\n    // determine which functions are available based on the available eqipment\n    // ---\n\n"
    for index, guard in enumerate(guards):
        branches += "    // " + get_node_name(G, guard) + "\n"
        branches += "    if ( " + guards[guard] + " ) //NOLINT//\n    {\n"
        # branches += "        " + get_node_name(G, guard) + " = true;\n"
        branches += "        y[" + str(index) + "] = true;\n"
        branches += "    }\n"
    print(branches) if verbose else None
    return branches


def generate_available_modes(G, unique_graph_list, filename, verbose):
    guards = get_guards(G, unique_graph_list, verbose)
    all_equipment = sorted([get_node_name(G, node) for node in find_leaf_nodes(G, get_layers(G))])
    initialization = get_initialization(unique_graph_list, all_equipment, verbose)
    branches = get_branches(G, guards, verbose)
    end = "}"
    code = initialization + branches + end
    # print(code)
    with open(filename, "w") as text_file:
        print(code, file=text_file)
