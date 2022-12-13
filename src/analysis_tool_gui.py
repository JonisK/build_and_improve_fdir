#!/usr/bin/env python3.9
#
# Copyright (c) 2022 Jonis Kiesbye, Kush Grover
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from graph_analysis.generate_config_json import generate_config_json_isolation
from graph_analysis.graph_analysis import create_graph_list, get_layers, \
    get_node_name, find_root_nodes, find_leaf_nodes, check_isolability, \
    check_recoverability, get_root_node_names, get_fault_probability, \
    get_node_id, find_isolated_nodes
import graph_analysis.prism_isolation

import pydot
import networkx as nx
import xdot
import re
import textwrap
import logging
import os

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Vte', '2.91')
from gi.repository import GLib
from gi.repository import Gtk, Vte


class MyHandler(logging.Handler):
    def __init__(self, log_output):
        logging.Handler.__init__(self)
        self.log_output = log_output

    def handle(self, record):
        message_with_linebreaks = '\n'.join(l for l in textwrap.wrap(str(self.format(record)), width=81))

        self.log_output.set_editable(True)
        self.log_output.get_buffer().insert_at_cursor(message_with_linebreaks + "\n")
        self.log_output.set_editable(False)


class MainWindow(Gtk.Window):
    do_not_prune = False

    def __init__(self):
        self.directory = os.getcwd() + "/../"
        self.base_directory = os.getcwd() + "/../"
        self.filename = ""
        self.filename_fault_probs = ""
        self.filename_mode_costs = ""
        self.filename_initial_state = ""
        self.G = None

        self.analysis_done = False
        self.check_isolability_done = False
        self.check_recoverability_done = False

        self.non_isolable = []
        self.non_recoverable = []
        self.num_non_isolable = 0
        self.num_non_recoverable = 0

        self.all_equipment = []
        self.leaf_name_lists = {}

        Gtk.Window.__init__(self, title="Analysis Tool")
        self.set_border_width(10)
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        paned.set_position(650)
        self.add(paned)

        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        grid.set_column_homogeneous(True)

        self.button_import = Gtk.Button(label="Import Graph")
        self.button_import.connect("clicked", self.on_open)
        grid.attach(self.button_import, 0, 2, 1, 1)
        self.button_analyze = Gtk.Button(label="Analyze Graph")
        self.button_analyze.set_sensitive(False)
        self.button_analyze.connect("clicked", self.on_analyze)
        grid.attach_next_to(self.button_analyze, self.button_import, Gtk.PositionType.RIGHT, 1, 1)

        self.graph_stats = Gtk.Label()
        self.graph_stats.set_xalign(0)  # left-aligned
        self.graph_stats.set_markup(
            f"<b><big>No graph selected</big></b>\n"
            + f" - ? modes\n"
            + f" - ? components\n"
            + f" - ? to ? configurations per mode\n")
        grid.attach(self.graph_stats, 0, 3, 2, 1)

        self.number_of_faults_label = Gtk.Label(label="Number of faults: ")
        grid.attach(self.number_of_faults_label, 0, 4, 1, 1)
        self.number_of_faults_entry = Gtk.Entry()
        self.number_of_faults_entry.set_text("1")
        self.number_of_faults_entry.connect("activate", self.reset_check_buttons)
        grid.attach_next_to(self.number_of_faults_entry, self.number_of_faults_label, Gtk.PositionType.RIGHT, 1, 1)

        self.children_to_keep_label = Gtk.Label(label="Actions to keep: ")
        grid.attach(self.children_to_keep_label, 0, 5, 1, 1)
        self.children_to_keep_entry = Gtk.Entry()
        self.children_to_keep_entry.set_text("2")
        grid.attach_next_to(self.children_to_keep_entry, self.children_to_keep_label, Gtk.PositionType.RIGHT, 1, 1)

        self.simulations_per_node_label = Gtk.Label(label="Simulations per node: ")
        grid.attach(self.simulations_per_node_label, 0, 6, 1, 1)
        self.simulations_per_node_entry = Gtk.Entry()
        self.simulations_per_node_entry.set_text("10")
        grid.attach_next_to(self.simulations_per_node_entry, self.simulations_per_node_label, Gtk.PositionType.RIGHT, 1, 1)

        self.button_check_isolation = Gtk.Button(label="Check Isolation")
        self.button_check_isolation.connect("clicked", self.check_isolation)
        self.button_check_isolation.set_sensitive(False)
        grid.attach(self.button_check_isolation, 0, 7, 1, 1)
        self.button_build_isolation = Gtk.Button(label="Build Isolation")
        self.button_build_isolation.set_sensitive(False)
        self.button_build_isolation.connect("clicked", self.build_prune_and_compress)
        grid.attach_next_to(self.button_build_isolation, self.button_check_isolation, Gtk.PositionType.RIGHT, 1, 1)
        self.button_export_isolation = Gtk.Button(label="Export PRISM")
        # self.button_export_isolation.set_sensitive(False)
        self.button_export_isolation.connect("clicked", self.export_isolation)
        grid.attach(self.button_export_isolation, 0, 8, 1, 1)
        self.button_run_isolation = Gtk.Button(label="Run PRISM")
        # self.button_run_isolation.set_sensitive(False)
        self.button_run_isolation.connect("clicked", self.run_isolation)
        grid.attach_next_to(self.button_run_isolation, self.button_export_isolation, Gtk.PositionType.RIGHT, 1, 1)
        self.isolation_info = Gtk.Label()
        self.isolation_info.set_xalign(0)  # left-aligned
        self.isolation_info.set_markup("<b><big>Isolation info</big></b>\n"
                                       + " - ? components can be isolated\n"
                                       + " - ? components cannot be isolated\n")
        grid.attach(self.isolation_info, 0, 9, 2, 1)

        self.button_check_recovery = Gtk.Button(label="Check Recovery")
        self.button_check_recovery.connect("clicked", self.check_recovery)
        self.button_check_recovery.set_sensitive(False)
        grid.attach(self.button_check_recovery, 0, 10, 1, 1)
        self.button_build_recovery = Gtk.Button(label="Build Recovery")
        self.button_build_recovery.connect("clicked", self.build_recovery)
        self.button_build_recovery.set_sensitive(False)
        grid.attach_next_to(self.button_build_recovery, self.button_check_recovery, Gtk.PositionType.RIGHT, 1, 1)
        self.button_export_recovery = Gtk.Button(label="Export PRISM")
        self.button_export_recovery.connect("clicked", self.export_recovery)
        # self.button_export_recovery.set_sensitive(False)
        grid.attach(self.button_export_recovery, 0, 11, 1, 1)
        self.button_run_recovery = Gtk.Button(label="Build Recovery")
        self.button_run_recovery.connect("clicked", self.run_recovery)
        # self.button_run_recovery.set_sensitive(False)
        grid.attach_next_to(self.button_run_recovery, self.button_export_recovery, Gtk.PositionType.RIGHT, 1, 1)
        self.recovery_info = Gtk.Label()
        self.recovery_info.set_xalign(0)  # left-aligned
        self.recovery_info.set_markup("<b><big>Recovery info</big></b>\n"
                                      + " - ? modes are fault-tolerant\n"
                                      + " - ? modes are not fault-tolerant\n")
        grid.attach(self.recovery_info, 0, 12, 2, 1)

        self.notebook = Gtk.Notebook()
        # First page, xdot view of the graph
        self.page1 = xdot.DotWidget()
        self.notebook.append_page(child=self.page1, tab_label=Gtk.Label(label='Show graph'))

        # Second page, enter fault probabilities
        self.fault_probabilities_text = Gtk.TextView()
        self.fault_probabilities_text.set_editable(True)

        fault_probabilities_scroller = Gtk.ScrolledWindow()
        fault_probabilities_scroller.set_border_width(10)
        fault_probabilities_scroller.set_hexpand(True)
        fault_probabilities_scroller.set_vexpand(True)
        fault_probabilities_scroller.add(self.fault_probabilities_text)
        fault_probabilities_scroller_box = Gtk.Box()
        fault_probabilities_scroller_box.add(fault_probabilities_scroller)

        fault_probabilities_save_button = Gtk.Button(label="Save to File")
        fault_probabilities_save_button.connect('clicked', self.write_probabilities)
        self.page2 = Gtk.Grid()
        self.page2.set_border_width(10)
        self.page2.attach(fault_probabilities_save_button, 0, 0, 1, 1)
        self.page2.attach(fault_probabilities_scroller_box, 0, 1, 1, 10)
        self.notebook.append_page(child=self.page2, tab_label=Gtk.Label(label='Fault Probabilities'))

        # Third page, enter mode costs
        self.mode_costs_text = Gtk.TextView()
        self.mode_costs_text.set_editable(True)
        mode_costs_scroller = Gtk.ScrolledWindow()
        mode_costs_scroller.set_border_width(10)
        mode_costs_scroller.set_hexpand(True)
        mode_costs_scroller.set_vexpand(True)
        mode_costs_scroller.add(self.mode_costs_text)
        mode_costs_scroller_box = Gtk.Box()
        mode_costs_scroller_box.add(mode_costs_scroller)

        mode_costs_save_button = Gtk.Button(label="Save to File")
        mode_costs_save_button.connect('clicked', self.write_costs)
        self.page3 = Gtk.Grid()
        self.page3.set_border_width(10)
        self.page3.attach(mode_costs_save_button, 0, 0, 1, 1)
        self.page3.attach(mode_costs_scroller_box, 0, 1, 1, 10)

        self.notebook.append_page(child=self.page3, tab_label=Gtk.Label(label='Mode Costs'))

        # Fourth page, show weakness report
        self.page4 = Gtk.Box()
        self.page4.set_border_width(10)
        self.report_text = Gtk.TextView()
        self.report_text.set_editable(False)
        self.get_report_initial()
        report_scroller = Gtk.ScrolledWindow()
        report_scroller.set_border_width(10)
        report_scroller.set_hexpand(True)
        report_scroller.set_vexpand(True)
        report_scroller.add(self.report_text)
        self.page4.add(report_scroller)
        self.notebook.append_page(child=self.page4, tab_label=Gtk.Label(label='Weakness Report'))

        # Fifth page, enter equipment state
        self.page5 = Gtk.Box()
        self.page5.set_border_width(10)

        self.grid = Gtk.Grid()
        self.grid.set_column_homogeneous(True)
        self.grid.set_row_homogeneous(True)
        self.page5.add(self.grid)

        self.states_liststore = Gtk.ListStore(str)
        self.states_liststore.append(['suspicious'])
        self.states_liststore.append(['available'])

        self.initialize_liststore(self.all_equipment)
        self.scrollable_treelist = self.initialize_treelist(self.states_liststore,
                                                            self.on_combo_changed)
        self.grid.attach(self.scrollable_treelist, 0, 0, 4, 9)

        # export button
        export_button = Gtk.Button(label="Export")
        export_button.connect("clicked", self.export_action)
        self.grid.attach(export_button, 0, 9, 1, 1)

        # build isolation per state button
        build_isolation_per_state_button = Gtk.Button(label="Build")
        # build_isolation_per_state_button.connect("clicked", self.prune_graph_with_initial_state)
        self.grid.attach_next_to(build_isolation_per_state_button, export_button, Gtk.PositionType.RIGHT, 1, 1)
        self.notebook.append_page(child=self.page5, tab_label=Gtk.Label(label='Enter State'))

        # Sixth page, xdot view of the isolation graph
        self.page6 = xdot.DotWidget()
        self.notebook.append_page(child=self.page6, tab_label=Gtk.Label(label='Isolation graph'))

        grid.attach(self.notebook, 2, 1, 5, 12)
        paned.add1(grid)

        # Terminal and Python Log
        self.terminal_notebook = Gtk.Notebook()

        # Terminal view
        self.terminal = Vte.Terminal()
        self.pty = Vte.Pty.new_sync(Vte.PtyFlags.DEFAULT)
        self.terminal.set_pty(self.pty)
        self.pty.spawn_async(
            self.base_directory,
            ["/bin/sh"],
            None,
            GLib.SpawnFlags.DO_NOT_REAP_CHILD,
            None,
            None,
            -1,
            None,
            self.ready
        )
        self.terminal.set_cursor_blink_mode(Vte.CursorBlinkMode.OFF)

        # a scroll window is required for the terminal
        self.scroller = Gtk.ScrolledWindow()
        self.scroller.set_hexpand(True)
        self.scroller.set_vexpand(True)
        self.scroller.add(self.terminal)

        self.terminal_notebook.append_page(child=self.scroller, tab_label=Gtk.Label(label='Terminal'))

        # Python Log
        self.log_output = Gtk.TextView()
        self.log_output.set_editable(False)

        self.log_scroller = Gtk.ScrolledWindow()
        # self.log_scroller.connect('size-allocate', self.scroll_down)
        self.log_scroller.set_hexpand(True)
        self.log_scroller.set_vexpand(True)
        self.log_scroller.add(self.log_output)

        self.terminal_notebook.append_page(child=self.log_scroller, tab_label=Gtk.Label(label='Log'))

        paned.add2(self.terminal_notebook)

    def on_analyze(self, action):
        self.button_analyze.set_sensitive(False)
        self.terminal_notebook.set_current_page(1)
        self.analyze_graph(self.G)
        self.get_graph_stats(self.filename, self.G)
        self.scroll_down2()

    def on_open(self, action):
        chooser = Gtk.FileChooserDialog(parent=self,
                                        title="Open Architecture as .dot Graph",
                                        action=Gtk.FileChooserAction.OPEN)
        chooser.add_buttons(Gtk.STOCK_CANCEL,
                            Gtk.ResponseType.CANCEL,
                            Gtk.STOCK_OPEN,
                            Gtk.ResponseType.OK)
        chooser.set_default_response(Gtk.ResponseType.OK)
        chooser.set_current_folder(self.directory)
        filter = Gtk.FileFilter()
        filter.set_name(".dot files")
        # filter.add_pattern("*.gv")
        filter.add_pattern("*.dot")
        chooser.add_filter(filter)
        filter = Gtk.FileFilter()
        filter.set_name("All files")
        filter.add_pattern("*")
        chooser.add_filter(filter)
        if chooser.run() == Gtk.ResponseType.OK:
            self.filename = chooser.get_filename()
            self.directory = chooser.get_current_folder()
            chooser.destroy()
            self.filename_fault_probs = self.filename.split(".")[0] + "_fault_probabilities.txt"
            self.filename_mode_costs = self.filename.split(".")[0] + "_mode_costs.txt"
            self.filename_initial_state = self.filename.split(".")[0] + "_initial_state.txt"

            # Reset buttons
            self.button_analyze.set_sensitive(True)
            self.button_check_isolation.set_sensitive(False)
            self.button_build_isolation.set_sensitive(False)
            self.button_check_recovery.set_sensitive(False)
            self.button_build_recovery.set_sensitive(False)
            self.get_report_initial()
            self.analysis_done = False
            self.check_isolability_done = False
            self.check_recoverability_done = False

            self.open_file(self.filename, self.page1)
            self.page1.zoom_to_fit()
            self.get_graph_stats_filename(self.filename)
            self.import_graph(self.filename)
            self.terminal_notebook.set_current_page(1)
            self.get_graph_stats_initial(self.filename, self.G)
            self.read_probabilities()
            self.read_costs()
            self.update_enter_state(self.all_equipment)
        else:
            chooser.destroy()

    def open_file(self, filename, page):
        try:
            fp = open(filename, 'rb')
            page.set_dotcode(fp.read(), filename)
            fp.close()
        except IOError as ex:
            self.error_dialog(str(ex))

    def import_graph(self, filename):
        logging.basicConfig(
            format="[%(levelname)s] %(funcName)s: %(message)s")
        logging.getLogger().setLevel(logging.INFO)

        logging.info("Reading from graph " + filename)

        graphs = pydot.graph_from_dot_file(filename)
        graph = graphs[0]
        self.G = nx.DiGraph(nx.nx_pydot.from_pydot(graph))
        if len(find_isolated_nodes(self.G)) > 0:
            logging.warning(
                f"Found {len(find_isolated_nodes(self.G))} isolated nodes: {find_isolated_nodes(self.G)}. Removing them.")
            for node in find_isolated_nodes(self.G):
                self.G.remove_node(node)
        else:
            logging.info(f"No isolated nodes found")

        layers = get_layers(self.G)
        self.all_equipment = sorted([get_node_name(self.G, node) for node in find_leaf_nodes(self.G, layers)])
        logging.info(f"All equipment: {[(index, component) for index, component in enumerate(self.all_equipment)]}")
        self.scroll_down2()

    def get_graph_stats_filename(self, filename):
        self.graph_stats.set_markup(
            f"<b><big>Selected graph: {filename.split('/')[-1]}</big></b>\n"
            + f" - ? modes\n"
            + f" - ? components\n"
            + f" - ? to ? configurations per mode\n")

    def get_graph_stats_initial(self, filename, G):
        layers = get_layers(G)
        self.graph_stats.set_markup(
            f"<b><big>Selected graph: {filename.split('/')[-1]}</big></b>\n"
            + f" - {len(find_root_nodes(G))} modes\n"
            + f" - {len(find_leaf_nodes(G, layers))} components\n"
            + f" - ? to ? configurations per mode\n")

    def get_graph_stats(self, filename, G):
        layers = get_layers(G)
        num_configs = [len(self.leaf_name_lists[this_list]) for this_list in self.leaf_name_lists]
        self.graph_stats.set_markup(
            f"<b><big>Selected graph: {filename.split('/')[-1]}</big></b>\n"
            + f" - {len(find_root_nodes(G))} modes\n"
            + f" - {len(find_leaf_nodes(G, layers))} components\n"
            + f" - {min(num_configs)} to {max(num_configs)} configurations per mode\n")

    def analyze_graph(self, G):
        logging.info("Analyze the configuration graph")
        self.unique_graph_list, unique_node_lists, self.leaf_name_lists, \
            self.configuration_list, configuration_space = \
            create_graph_list(G, threading=False)

        # set button states
        self.button_check_isolation.set_sensitive(True)
        self.button_build_isolation.set_sensitive(True)
        self.button_check_recovery.set_sensitive(True)
        self.button_build_recovery.set_sensitive(True)
        self.analysis_done = True
        self.get_report()
        self.scroll_down2()

    def reset_check_buttons(self, widget):
        if self.analysis_done:
            self.button_check_isolation.set_sensitive(True)
            self.button_check_recovery.set_sensitive(True)

    def check_isolation(self, button):
        self.button_check_isolation.set_sensitive(False)
        self.terminal_notebook.set_current_page(1)
        logging.info("Checking isolation")
        self.isolable, self.non_isolable = check_isolability(self.all_equipment,
                                                             self.leaf_name_lists,
                                                             int(self.number_of_faults_entry.get_text()))
        self.num_non_isolable = len(self.non_isolable)
        num_isolable = len(self.all_equipment) - self.num_non_isolable
        self.isolation_info.set_markup(
            f"<b><big>Isolation info</big></b>\n"
            + f" - {num_isolable} components ({100*(num_isolable/len(self.all_equipment)):.2f}%) can be isolated\n"
            + f" - {self.num_non_isolable} components cannot be isolated\n")
        self.check_isolability_done = True
        self.get_report()
        self.scroll_down2()

    def build_prune_and_compress(self, button):
        self.button_build_isolation.set_sensitive(False)
        self.terminal_notebook.set_current_page(0)
        self.prune_graph(button)
        self.feed_input(f'\n')
        generate_config_json_isolation(
            self.all_equipment,
            self.base_directory + "/temp/",
            self.base_directory + "/temp/prism_strategy_config.json")
        strategy_name = 'temp/prism_strategy.prism'
        self.feed_input(f'dtcontrol --input {strategy_name} --use-preset avg --benchmark-file benchmark.json --rerun\n')

    def prune_graph(self, button):
        self.feed_input(f'python3 src/mcts.py '
            f'--modecosts {self.filename_mode_costs} '
            f'--equipfailprobs {self.filename_fault_probs} '
            f'--successorstokeep {self.children_to_keep_entry.get_text()} '
            f'--simulationsize {self.simulations_per_node_entry.get_text()} '
            f'{self.filename}\n')

    def prune_graph_with_initial_state(self, button):
        self.feed_input(f'python3 src/mcts.py '
            f'--modecosts {self.filename_mode_costs} '
            f'--equipfailprobs {self.filename_fault_probs} '
            f'--successorstokeep {self.children_to_keep_entry.get_text()} '
            f'--simulationsize {self.simulations_per_node_entry.get_text()} '
            f'--initialstate '
            f'{self.filename}\n')
        self.open_file(self.filename, self.page6)
    # def prune_graph_with_initial_state(self, button):
    #     self.terminal_notebook.set_current_page(1)
    #
    #     configuration_lists = get_configuration_lists(self.leaf_name_lists, self.all_equipment)
    #
    #     strategy_filename = self.directory + "/../../temp/strategy_initial.prism"
    #     states_filename = self.directory + "/../../temp/strategy_initial_states.prism"
    #     writer = StrategyWriter(states_filename, strategy_filename)
    #     writer.write_header(self.all_equipment)
    #
    #     equipment_state = [1 if self.equipment_liststore[state][1] == "available" else 0 \
    #                        for state in range(len(self.all_equipment))]
    #
    #     traverse_binary_tree_weights(self.G, configuration_lists, equipment_state, self.get_costs(), writer, verbose=False)
    #     writer.close()

    def ready(self, pty, task):
        pass

    def feed_input(self, text):
        text = bytearray(text, "utf-8")
        self.terminal.feed_child(text)

    def display_input(self, text):
        self.log_output.set_editable(True)
        self.log_output.get_buffer().insert_at_cursor(text)
        self.log_output.set_editable(False)

    def export_isolation(self, button):
        graph_analysis.prism_isolation.generate_prism_model(
            self.base_directory + "temp/" + "isolation_model.prism",
            self.G,
            self.all_equipment,
            self.unique_graph_list,
            self.leaf_name_lists,
            self.configuration_list,
            self.get_probabilities(),
            self.get_costs(),
            debug=False)

        graph_analysis.prism_isolation.generate_props(
            self.base_directory + "temp/" + "isolation_model.prism",
            self.all_equipment)
            
    def run_isolation(self, button):
        isolability, isolation_cost = graph_analysis.prism_isolation.run_prism(
            self.base_directory + "temp/" + "isolation_model.prism",
            self.all_equipment)

    def check_recovery(self, button):
        self.button_check_recovery.set_sensitive(False)
        self.terminal_notebook.set_current_page(1)
        logging.info("Checking recovery")
        self.recoverable, self.non_recoverable = check_recoverability(
            self.G,
            self.all_equipment,
            self.leaf_name_lists,
            int(self.number_of_faults_entry.get_text()))
        self.num_non_recoverable = len(self.non_recoverable)
        num_recoverable = len(self.leaf_name_lists) - self.num_non_recoverable
        self.recovery_info.set_markup(
            f"<b><big>Recovery info</big></b>\n"
            + f" - {num_recoverable} modes ({100*(num_recoverable/len(self.leaf_name_lists)):.2f}%) are fault-tolerant\n"
            + f" - {self.num_non_recoverable} modes are not fault-tolerant\n")
        self.check_recoverability_done = True
        self.get_report()
        self.scroll_down2()

    def build_recovery(self, button):
        self.button_build_recovery.set_sensitive(False)
        self.terminal_notebook.set_current_page(0)
        self.feed_input(f"python3 src/build_recovery.py {self.base_directory} {self.directory} {self.filename}\n")

    def export_recovery(self, button):
        pass

    def run_recovery(self, button):
        pass

    def read_probabilities(self):
        try:
            with open(self.filename_fault_probs, 'r') as file_ref:
                file_content = file_ref.read()
                logging.info(f"Read fault probabilities from {self.filename_fault_probs}")
        except FileNotFoundError:
            file_content = self.generate_fault_probs()
            logging.warning(f"File {self.filename_fault_probs} doesn't exist yet")
        finally:
            self.fault_probabilities_text.get_buffer().set_text(file_content, len(file_content))

    def generate_fault_probs(self):
        string_list = [component + ": 0.0" for component in self.all_equipment]
        logging.info(f"Generated fault probabilites template: {', '.join(string_list)}")
        return ",\n".join(string_list)

    def write_probabilities(self, button):
        with open(self.filename_fault_probs, 'w') as file_ref:
            file_ref.write(self.fault_probabilities_text.get_buffer().get_text(
                self.fault_probabilities_text.get_buffer().get_start_iter(),
                self.fault_probabilities_text.get_buffer().get_end_iter(), False))

    def get_probabilities(self):
        probabilities_text = self.fault_probabilities_text.get_buffer().get_text(
            self.fault_probabilities_text.get_buffer().get_start_iter(),
            self.fault_probabilities_text.get_buffer().get_end_iter(), False)
        return {line.split(":")[0]: float(line.split(":")[1]) for line in probabilities_text.split(",\n")}

    def read_costs(self):
        try:
            with open(self.filename_mode_costs, 'r') as file_ref:
                file_content = file_ref.read()
                logging.info(f"Read mode costs from {self.filename_mode_costs}")
        except FileNotFoundError:
            file_content = self.generate_mode_costs()
            logging.warning(f"File {self.filename_mode_costs} doesn't exist yet")
        finally:
            self.mode_costs_text.get_buffer().set_text(file_content, len(file_content))

    def get_costs(self):
        mode_costs = {}
        print(self.mode_costs_text.get_buffer().get_text(
            self.mode_costs_text.get_buffer().get_start_iter(),
            self.mode_costs_text.get_buffer().get_end_iter(), False))
        for line in self.mode_costs_text.get_buffer().get_text(
                self.mode_costs_text.get_buffer().get_start_iter(),
                self.mode_costs_text.get_buffer().get_end_iter(), False).split("\n"):
            print(line)
            mode_costs[re.findall("\\w+(?=:)", line)[0]] = float(re.findall("\\d+.\\d+", line)[0])
        # print(mode_costs)
        return mode_costs

    def generate_mode_costs(self):
        string_list = [mode + ": 0.0" for mode in get_root_node_names(self.G)]
        logging.info(f"Generated mode costs template: {', '.join(string_list)}")
        return ",\n".join(string_list)

    def write_costs(self, button):
        with open(self.filename_mode_costs, 'w') as file_ref:
            file_ref.write(self.mode_costs_text.get_buffer().get_text(
                self.mode_costs_text.get_buffer().get_start_iter(),
                self.mode_costs_text.get_buffer().get_end_iter(), False))

    def scroll_down(self, widget, event, data=None):
        adj = widget.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    def scroll_down2(self):
        adj = self.log_scroller.get_vadjustment()
        adj.set_value(adj.get_lower() - adj.get_page_size())

    def get_report_initial(self):
        # Clear textview
        self.report_text.get_buffer().delete(
            self.report_text.get_buffer().get_start_iter(),
            self.report_text.get_buffer().get_end_iter())

        message = "Run Analyze Graph, Check Isolation, and Check Recovery to generate the report"
        end_iter = self.report_text.get_buffer().get_end_iter()
        self.report_text.get_buffer().insert_markup(end_iter, message, -1)

    def get_report(self):
        # Clear textview
        self.report_text.get_buffer().delete(
            self.report_text.get_buffer().get_start_iter(),
            self.report_text.get_buffer().get_end_iter())

        message = "<b><big>Weakness Report</big></b>\n\n"
        if self.check_isolability_done and self.check_recoverability_done:
            if self.num_non_isolable == 0 and self.num_non_recoverable == 0:
                message += f"The graph {self.filename.split('/')[-1]} shows no weaknesses.\n"
            else:
                message += f"The graph {self.filename.split('/')[-1]} shows the following weaknesses:\n"
                for component in self.non_isolable:
                    message += f"\tComponent {component} is not isolable\n"
                message += "\n"
                for mode in self.non_recoverable:
                    message += f"\tMode {get_node_name(self.G, mode)} is not " \
                               f"{self.number_of_faults_entry.get_text()}-fault tolerant\n"
            if self.num_non_isolable or self.num_non_recoverable:
                message += "\nThe following components and modes show no weaknesses:\n"
                for component in self.isolable:
                    message += f"\tComponent {component} is isolable\n"
                for mode in self.recoverable:
                    message += f"\tMode {get_node_name(self.G, mode)} is " \
                               f"{self.number_of_faults_entry.get_text()}-fault tolerant\n"
                message += "\n"
        else:
            message += "Run Check Isolation and Check Recovery to include an assessment on the isolability of the components and recoverability of the modes.\n"

        if self.analysis_done:
            message += f"Assuming the component fault probabilities defined in ‘{self.filename_fault_probs.split('/')[-1]}’, the modes have these fault probabilities:\n"

            fault_probs = {mode: get_fault_probability(self.G, mode, self.get_probabilities()) for mode in find_root_nodes(self.G)}
            fault_probs_sorted = dict(sorted(fault_probs.items(), key=lambda item: item[1], reverse=True))
            for mode in fault_probs_sorted:
                message += f"\tThe fault probability for mode {get_node_name(self.G, mode)} is {100 * fault_probs_sorted[mode]:.5f} %\n"

        end_iter = self.report_text.get_buffer().get_end_iter()
        self.report_text.get_buffer().insert_markup(end_iter, message, -1)

    def initialize_liststore(self, all_equipment):
        # Creating the ListStore model
        self.equipment_liststore = Gtk.ListStore(str, str)
        self.update_enter_state(all_equipment)

    def update_enter_state(self, all_equipment):
        self.equipment_liststore.clear()
        for equipment in all_equipment:
            self.equipment_liststore.append([equipment, "suspicious"])

    def initialize_treelist(self, states_liststore, change_callback):
        # creating the treeview and adding the columns
        treeview = Gtk.TreeView(model=self.equipment_liststore)
        for i, column_title in enumerate(["Equipment"]):
            renderer = Gtk.CellRendererText()
            column = Gtk.TreeViewColumn(column_title, renderer, text=i)
            treeview.append_column(column)

        cellrenderercombo = Gtk.CellRendererCombo(model=states_liststore)
        cellrenderercombo.set_property("editable", True)
        cellrenderercombo.set_property("has-entry", False)
        cellrenderercombo.set_property('sensitive', True)
        cellrenderercombo.set_property('mode', Gtk.CellRendererMode.EDITABLE)
        cellrenderercombo.set_property('height', 25)
        cellrenderercombo.set_property("model", states_liststore)
        cellrenderercombo.set_property("text_column", 0)
        cellrenderercombo.connect("changed", change_callback)
        column = Gtk.TreeViewColumn("State", cellrenderercombo)
        treeview.append_column(column)
        column.add_attribute(cellrenderercombo, "text", 1)

        # setting up the layout, putting the treeview in a scrollwindow, and the buttons in a row
        scrollable_treelist = Gtk.ScrolledWindow()
        scrollable_treelist.set_vexpand(True)
        scrollable_treelist.add(treeview)
        return scrollable_treelist

    def on_combo_changed(self, cellrenderercombo, treepath, treeiter):
        self.equipment_liststore[treepath][1] = self.states_liststore[treeiter][0]

    def export_action(self, widget):
        equipment_state = [self.equipment_liststore[state][1] for state in range(len(self.all_equipment))]
        with open(self.filename_initial_state, 'w') as file_ref:
            file_ref.write(str(equipment_state))


def main():
    window = MainWindow()
    window.set_default_size(1500, 900)
    window.connect('delete-event', Gtk.main_quit)
    window.show_all()

    logging.basicConfig(
        format="[%(levelname)s] %(funcName)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = MyHandler(window.log_output)
    #logger.addHandler(handler)

    Gtk.main()


if __name__ == '__main__':
    main()
