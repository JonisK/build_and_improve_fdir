import logging
import logging.handlers
import os
import pathlib
import random
import re
import string
import sys
import time

import networkx as nx
from tqdm import tqdm
import argparse

import evaluate_prism_strat
from base import get_configuration_all_modes, no_possible_successors, get_fault_probabilities, remove_unnecessary_nodes, \
    list_to_int
from evaluate_mcts_strategy import evaluate_mcts_strategy
from expand import add_edge, mcts_expand, add_state
from export import export_graph, export_mcts_strategy, export_prism_file
from selec import mcts_select
from simulations import mcts_simulate
from naive import evaluate_naive
from trim import mcts_trim


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not os.path.isdir("./temp/logs"):
        os.mkdir("./temp/logs")
    log_filename = "./temp/logs/mcts.log"
    should_roll_over = os.path.isfile(log_filename)
    handler = logging.handlers.RotatingFileHandler(log_filename, mode='w', backupCount=100)
    if should_roll_over:  # log already exists, roll over!
        handler.doRollover()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def is_not_finished(nodes_to_explore):
    if len(nodes_to_explore) == 0:
        return False
    else:
        return True


def mcts(mcts_graph, mcts_data, statistics, parameters, state):
    total_sim_iter = 0
    iteration_number = 0
    while is_not_finished(statistics["nodes_to_explore"]):
        iteration_number += 1

        if parameters["successors_to_keep"] == 0:
            i = random.randrange(len(statistics["nodes_to_explore"]))
            selected_state = statistics["nodes_to_explore"][i]
            if parameters["debug"]:
                logging.debug("Expanding node: " + str(selected_state))
            if no_possible_successors(statistics, selected_state):
                statistics["nodes_to_explore"].remove(selected_state)
                continue
            mcts_expand(mcts_graph, mcts_data, statistics, parameters, selected_state)
            statistics["nodes_explored"].append(selected_state)
            statistics["nodes_to_explore"].remove(selected_state)
            mcts_graph = mcts_trim(mcts_graph, mcts_data, statistics, parameters, selected_state)

        else:
            selected_state, path, action_path = mcts_select(mcts_graph, statistics, state)
            if parameters["debug"]:
                logging.debug("Expanding node: " + str(selected_state))
            if no_possible_successors(statistics, selected_state):
                statistics["nodes_to_explore"].remove(selected_state)
                continue
            num_new_successors = mcts_expand(mcts_graph, mcts_data, statistics, parameters, selected_state)
            max_sim_round = parameters["simulations_for_each_children"] * num_new_successors

            if parameters["debug"]:
                logging.debug("Number of simulations: " + str(max_sim_round))

            num_sim = mcts_simulate(mcts_data, statistics, parameters, selected_state,
                                    path, action_path, max_sim_round)

            statistics["nodes_explored"].append(selected_state)
            statistics["nodes_to_explore"].remove(selected_state)
            statistics["total_simulations"] += num_sim

            mcts_graph = mcts_trim(mcts_graph, mcts_data, statistics, parameters, selected_state)
            total_sim_iter += num_sim
    return mcts_graph, total_sim_iter


def get_state_from_file(statistics, filename):
    f = open(filename, "r")
    text = f.read()
    x = re.search(r"\[(.*)]", text)
    state = x.group(1).split(", ")
    for i in range(len(state)):
        if state[i] == "1":
            state[i] = 1
        elif state[i] == "0":
            state[i] = 0
    return list_to_int(statistics, state)


def mcts_outer(parameters):
    # initialize the mcts graph
    mcts_graph = nx.DiGraph()
    root_node = 0
    mcts_graph.add_node(root_node)
    mcts_data = {}

    # initialize some statistics to keep track of
    statistics = {'total_simulations': 0, 'rounds': 0, 'nodes_to_explore': [], 'nodes_explored': [],
                  "available_actions": {}, "name_to_action_mapping": {}, "action_to_name_mapping": {},
                  "int_to_list_mapping": {}}

    # noinspection PyBroadException
    try:
        parse_cost(statistics, parameters)
    except:
        print("Error in input file:", parameters["cost_file"], "in the mode_costs part")

    get_configuration_all_modes(statistics, parameters)

    # noinspection PyBroadException
    try:
        parse_equipment(statistics, parameters)
    except:
        print("Syntax error in input file:", parameters["cost_file"], "in the equipment_fault_probabilities part")

    print("Starting MCTS...")
    time.sleep(0.01)
    if parameters["initial_state_file"] != "":
        state = get_state_from_file(statistics, parameters["initial_state_file"])
        if parameters["debug"]:
            logging.debug("-------------------------------------------------------------------------------------------")
            logging.debug("-------------------------------------------------------------------------------------------")
            logging.debug("Initial state: " + str(state) + "\n")
        add_state(mcts_graph, mcts_data, statistics, state)
        add_edge(mcts_graph, root_node, state)
        statistics["nodes_to_explore"].append(state)
        mcts_graph, num_current_round_sim = mcts(mcts_graph, mcts_data, statistics, parameters, state)
        statistics["total_simulations"] += num_current_round_sim
        statistics["rounds"] += 1
        if parameters["debug"]:
            logging.debug("Total simulations: " + str(statistics["total_simulations"]))
            logging.debug("finished mcts from state: " + str(state))
            logging.debug(
                "-------------------------------------------------------------------------------------------\n\n")
    else:
        for state in tqdm(statistics["all_actions"]):
            if parameters["debug"]:
                logging.debug(
                    "-------------------------------------------------------------------------------------------")
                logging.debug(
                    "-------------------------------------------------------------------------------------------")
                logging.debug("Initial state: " + str(state) + "\n")
            add_state(mcts_graph, mcts_data, statistics, state)
            add_edge(mcts_graph, root_node, state)
            statistics["nodes_to_explore"].append(state)
            mcts_graph, num_current_round_sim = mcts(mcts_graph, mcts_data, statistics, parameters, state)
            statistics["total_simulations"] += num_current_round_sim
            statistics["rounds"] += 1
            if parameters["debug"]:
                logging.debug("Total simulations: " + str(statistics["total_simulations"]))
                logging.debug("finished mcts from state: " + str(state))
                logging.debug(
                    "-------------------------------------------------------------------------------------------\n\n")

    if parameters["debug"]:
        logging.debug("MCTS data: " + str(mcts_data) + "\n\n")

    print("done\n")

    return mcts_graph, mcts_data, statistics


def arguments(parameters):
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('input',
                           metavar='input',
                           action='store',
                           type=str,
                           help='dot file describing the dependency graph')
    my_parser.add_argument('--modecosts',
                           required=True,
                           action='store',
                           type=str,
                           help='txt file describing cost of all the modes')
    my_parser.add_argument('--equipfailprobs',
                           required=True,
                           action='store',
                           type=str,
                           help='txt file describing equipment fail probabilities')
    my_parser.add_argument('--successorstokeep',
                           action='store',
                           type=int,
                           help='Specify how many successors to keep after trimming')
    my_parser.add_argument('--simulationsize',
                           action='store',
                           type=int,
                           help='Simulation to do from each children for its evaluation')
    my_parser.add_argument('--samplingtype',
                           action='store',
                           type=int,
                           help='Sampling type for simulations (0 or 1): 0 for sampling the next successor based on '
                                'current probability and 1 for sampling a defect and then isolating the fault')
    my_parser.add_argument('--outputdir',
                           action='store',
                           type=str,
                           help='directory to store output')
    my_parser.add_argument('-d',
                           '--debug',
                           action='store_true',
                           help='enable debug mode')
    my_parser.add_argument('--mctsstrat',
                           action='store_true',
                           help='evaluate mcts strategy')
    my_parser.add_argument('--evaluatenaive',
                           action='store_true',
                           help='evaluate naive strategy')
    my_parser.add_argument('--initialstatefile',
                           action='store',
                           help='give initial state as input')

    args = my_parser.parse_args()

    parameters["input_file"] = args.input
    if args.modecosts is not None:
        parameters["cost_file"] = args.modecosts
    if args.equipfailprobs is not None:
        parameters["equipment_fail_probabilities_file"] = args.equipfailprobs

    if args.successorstokeep is not None:
        parameters["successors_to_keep"] = args.successorstokeep

    if args.simulationsize is not None:
        parameters["simulations_for_each_children"] = args.simulationsize

    if args.samplingtype is not None:
        parameters["sampling_type"] = args.samplingtype

    if args.outputdir is not None:
        parameters["strategy_file"] = args.outputdir + "/strategy.txt"
        parameters["output_dot_file"] = args.outputdir + "/mcts_graph.dot"
        if not os.path.isdir(args.outputdir):
            os.mkdir(args.outputdir)
    elif not os.path.exists('temp'):
        os.mkdir('temp')

    if args.debug is not None:
        parameters["debug"] = args.debug
    if args.mctsstrat is not None:
        parameters["mcts_strategy"] = args.mctsstrat
        if args.mctsstrat:
            parameters["successors_to_keep"] = 1
    if args.evaluatenaive is not None:
        parameters["evaluate_naive"] = args.evaluatenaive
    if args.initialstatefile is not None:
        parameters["initial_state_file"] = args.initialstatefile
    else:
        parameters["initial_state_file"] = ""

    if not os.path.isfile(parameters["input_file"]) or not os.path.isfile(
            parameters["cost_file"]) or not os.path.isfile(parameters["equipment_fail_probabilities_file"]):
        print('Input file(s) does not exist')
        sys.exit()


# noinspection DuplicatedCode
def parse_cost(statistics, parameters):
    f = open(parameters["cost_file"], "r")
    lines = f.readlines()
    mode_costs = {}
    for line in lines:
        item = re.search(r"([a-zA-Z0-9_-]*)\s*:\s*([0-9.]*)", line)
        if item is not None:
            mode_costs[item.group(1)] = float(item.group(2))
    statistics["mode_costs"] = mode_costs
    f.close()


# noinspection DuplicatedCode
def parse_equipment(statistics, parameters):
    f = open(parameters["equipment_fail_probabilities_file"], "r")
    lines = f.readlines()
    equipment_fail_probabilities = {}
    for line in lines:
        item = re.search(r"([a-zA-Z0-9_-]*)\s*:\s*([0-9.]*)", line)
        if item is not None:
            equipment_fail_probabilities[item.group(1)] = float(item.group(2))
    statistics["equipment_fail_probabilities"] = get_fault_probabilities(statistics, equipment_fail_probabilities)
    f.close()


def main():
    # initialize the parameters used
    # sampling type 0: sample the next successor based on distribution
    # 1: sample a defect and find successor according to that defect
    path_of_src = str(pathlib.Path(__file__).parent.parent.resolve())
    parameters = {"successors_to_keep": 10, "simulations_for_each_children": 200, "sampling_type": 0, "debug": False,
                  "output_graph": True, "output_dot_file": path_of_src + "/temp/mcts_graph.dot",
                  "strategy_file": path_of_src + "temp/strategy.txt"}

    # process arguments
    arguments(parameters)

    # setup debut logging
    if parameters["debug"]:
        setup_logging()

    start_time_mcts = time.time()
    graph, data, stats = mcts_outer(parameters)
    remove_unnecessary_nodes(graph)
    end_time_mcts = time.time()

    strategy = {}
    if parameters["mcts_strategy"]:
        strategy = export_mcts_strategy(graph, data, stats, parameters)
        evaluate_mcts_strategy(data, stats)
    if parameters["evaluate_naive"]:
        evaluate_naive(stats)

    # print stats
    print("Total simulations:", stats["total_simulations"])
    print("Graph size: ", len(graph.nodes) - 1)
    print("Graph transitions: ", len(graph.edges) - len(stats["all_modes"]))

    start_time_prism = 0
    end_time_prism = 0
    if not parameters["mcts_strategy"]:
        prism_filename = path_of_src + "/temp/model"
        start_time_prism = time.time()
        prism_state_to_state_mapping = export_prism_file(graph, parameters, stats, prism_filename)
        end_time_prism = time.time()
        strategy, output_state_to_prism_state = evaluate_prism_strat.generate_prism_strat(parameters, stats, prism_state_to_state_mapping)
        evaluate_prism_strat.evaluate_prism_strategy(parameters, stats, prism_state_to_state_mapping, strategy)

    # Exporting stuff
    if parameters["output_graph"]:
        remove_unnecessary_nodes(graph)
        export_graph(graph, stats, strategy, parameters["output_dot_file"])

    print("Time taken: ", round(end_time_mcts - start_time_mcts + end_time_prism - start_time_prism, 2), "s")


if __name__ == "__main__":
    main()
