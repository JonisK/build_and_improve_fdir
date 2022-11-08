#!/usr/bin/python3.9

import time
import sys
import os
import shutil
import networkx as nx
import pydot
from graph_analysis.graph_analysis import create_graph_list, get_layers, \
    get_node_name, find_leaf_nodes, find_isolated_nodes, get_mode_indices, \
    get_mode_indices_appended
from graph_analysis.generate_available_modes import generate_available_modes
from graph_analysis.generate_mode_switcher import generate_mode_switcher
from graph_analysis.generate_config_json import generate_config_json
from graph_analysis.run_prism import run_prism
from graph_analysis.run_dtcontrol import run_dtcontrol
from graph_analysis.generate_actions_list import create_actions_list
from graph_analysis.generate_reconfigure import generate_reconfigure

directory = sys.argv[1]
filename = sys.argv[2]

graphs = pydot.graph_from_dot_file(filename)
graph = graphs[0]
G = nx.DiGraph(nx.nx_pydot.from_pydot(graph))
if len(find_isolated_nodes(G)) > 0:
    print(
        f"Found {len(find_isolated_nodes(G))} isolated nodes: {find_isolated_nodes(G)}. Removing them.")
    for node in find_isolated_nodes(G):
        G.remove_node(node)
else:
    print(f"No isolated nodes found")

layers = get_layers(G)
all_equipment = sorted([get_node_name(G, node) for node in find_leaf_nodes(G, layers)])
(unique_graph_list, unique_node_lists, leaf_name_lists) = create_graph_list(G, verbose=False)

start_time = time.time()
verbose = False
directory_name = directory + "/recovery_" + filename.split('/')[-1].split('.')[0] + "/"
if os.path.exists(directory_name):
    shutil.rmtree(directory_name)
os.makedirs(directory_name)
available_modes_filename = "available_modes.c"
print("Generate " + available_modes_filename)
generate_available_modes(G, unique_graph_list, directory_name + available_modes_filename, verbose)

mode_switcher_filename = "mode_switcher_" + filename.split('/')[-1].split('.')[0] + ".prism"
print("Generate " + mode_switcher_filename)
with open(directory_name + "mode_switcher.props", "w") as text_file:
    print('Pmax=? [ F "mode_selected" ]\n', file=text_file)
with open(directory_name + "mode_switcher_printall_init.props", "w") as text_file:
    print('// Print values of all states, print max probability of reaching the target\n'
          + 'filter(printall, Pmax=? [ F "mode_selected" ], "init")\n', file=text_file)
generate_mode_switcher(get_mode_indices(G), get_mode_indices_appended(G),
                       directory_name + mode_switcher_filename)

print("Model-checking with PRISM")
prism_path = directory + "/../../prism/bin/prism"
mode_switcher_strategy_filename = "strategy_" + mode_switcher_filename
mode_switcher_properties_filename = "mode_switcher" + ".props"
command = run_prism(prism_path, directory_name + mode_switcher_filename,
                    directory_name + mode_switcher_properties_filename,
                    directory_name + mode_switcher_strategy_filename, verbose)
os.system(command)

print("Generate config JSON")
mode_switcher_config_filename = "strategy_" + mode_switcher_filename.split(".")[0] + "_config.json"
generate_config_json(get_mode_indices(G), get_mode_indices_appended(G),
                     directory_name + mode_switcher_config_filename)

print("Run dtControl and move decision tree")
command = run_dtcontrol(directory_name + mode_switcher_strategy_filename, verbose)
os.system(command)

reconfigure_filename = "reconfigure.c"
print("Generate " + reconfigure_filename)
actions_list = create_actions_list(directory_name + mode_switcher_strategy_filename)
print(actions_list)
generate_reconfigure(G, actions_list, get_mode_indices(G), get_mode_indices_appended(G),
                     directory_name + reconfigure_filename)

print("This model took " + str(time.time() - start_time) + "s")

os.system(f'''echo $(echo "scale=5; 100* $({prism_path} {directory_name + mode_switcher_filename} {directory_name}mode_switcher_printall_init.props | grep -c "=1.0") / $({prism_path} {directory_name + mode_switcher_filename} {directory_name}/mode_switcher.props | grep -oP '\d+(?= initial\))')" | bc)"% solvable of all initial states"''')
