#!/usr/bin/env python3.8
#
# Copyright 2008 Jose Fonseca
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import os
import shutil

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Vte', '2.91')
from gi.repository import Gtk, GObject, Vte
from gi.repository import GLib

import time
import logging
import textwrap

# import sys
# print(sys.path)
#sys.path.insert(0, './xdot.py/')
#sys.path.append('..')
# print(sys.path)
import xdot
#from implementation.src import mcts
import networkx as nx
import pydot

from graph_analysis.graph_analysis import create_graph_list, get_layers, get_node_name, find_root_nodes, find_leaf_nodes, check_isolability, check_recoverability, get_root_node_names, examine_successor, get_mode_indices, get_mode_indices_appended
from graph_analysis.generate_available_modes import generate_available_modes
from graph_analysis.generate_mode_switcher import generate_mode_switcher
from graph_analysis.generate_config_json import generate_config_json, generate_config_json_isolation
from graph_analysis.run_prism import run_prism
from graph_analysis.run_dtcontrol import run_dtcontrol
from graph_analysis.generate_actions_list import create_actions_list, generate_actions_list
from graph_analysis.generate_reconfigure import generate_reconfigure

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
        self.filename = ""
        self.filename_fault_probs = ""
        self.filename_mode_costs = ""
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
        paned.set_position(600)
        # paned.set_wide_handle(True)
        self.add(paned)

        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        grid.set_column_homogeneous(True)
        # grid.set_row_homogeneous(True)

        self.button_import = Gtk.Button(label="Import Graph")
        self.button_import.connect("clicked", self.on_open)
        grid.attach(self.button_import, 0, 1, 1, 1)
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
        grid.attach(self.graph_stats, 0, 2, 2, 1)

        self.children_to_keep_label = Gtk.Label(label="Actions to keep: ")
        grid.attach(self.children_to_keep_label, 0, 3, 1, 1)

        self.children_to_keep_entry = Gtk.Entry()
        self.children_to_keep_entry.set_text("2")
        grid.attach_next_to(self.children_to_keep_entry, self.children_to_keep_label, Gtk.PositionType.RIGHT, 1, 1)

        self.simulations_per_node_label = Gtk.Label(label="Simulations per node: ")
        grid.attach(self.simulations_per_node_label, 0, 4, 1, 1)

        self.simulations_per_node_entry = Gtk.Entry()
        self.simulations_per_node_entry.set_text("10")
        grid.attach_next_to(self.simulations_per_node_entry, self.simulations_per_node_label, Gtk.PositionType.RIGHT, 1, 1)

        self.button_check_isolation = Gtk.Button(label="Check Isolation")
        self.button_check_isolation.connect("clicked", self.check_isolation)
        self.button_check_isolation.set_sensitive(False)
        grid.attach(self.button_check_isolation, 0, 6, 1, 1)
        self.button_build_isolation = Gtk.Button(label="Build Isolation")
        self.button_build_isolation.set_sensitive(False)
        self.button_build_isolation.connect("clicked", self.build_prune_and_compress)
        grid.attach_next_to(self.button_build_isolation, self.button_check_isolation, Gtk.PositionType.RIGHT, 1, 1)
        self.isolation_info = Gtk.Label()
        self.isolation_info.set_xalign(0)  # left-aligned
        self.isolation_info.set_markup("<b><big>Isolation info</big></b>\n"
                                  + " - ? components can be isolated\n"
                                  + " - ? components cannot be isolated\n")
        grid.attach(self.isolation_info, 0, 7, 2, 1)

        self.button_check_recovery = Gtk.Button(label="Check Recovery")
        self.button_check_recovery.connect("clicked", self.check_recovery)
        self.button_check_recovery.set_sensitive(False)
        grid.attach(self.button_check_recovery, 0, 8, 1, 1)
        self.button_build_recovery = Gtk.Button(label="Build Recovery")
        self.button_build_recovery.connect("clicked", self.build_recovery)
        self.button_build_recovery.set_sensitive(False)
        grid.attach_next_to(self.button_build_recovery, self.button_check_recovery, Gtk.PositionType.RIGHT, 1, 1)
        self.recovery_info = Gtk.Label()
        self.recovery_info.set_xalign(0)  # left-aligned
        self.recovery_info.set_markup("<b><big>Recovery info</big></b>\n"
                                 + " - ? modes are fault-tolerant\n"
                                 + " - ? modes are not fault-tolerant\n")
        grid.attach(self.recovery_info, 0, 9, 2, 1)

        self.notebook = Gtk.Notebook()
        # First page, xdot view of the graph
        self.page1 = xdot.DotWidget()
        # self.page1.set_dotcode(dotcode)
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

        # Fourth page
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

        grid.attach(self.notebook, 2, 1, 5, 9)
        paned.add1(grid)

        # Terminal and Python Log
        self.terminal_notebook = Gtk.Notebook()

        # Terminal view
        self.terminal = Vte.Terminal()
        self.pty = Vte.Pty.new_sync(Vte.PtyFlags.DEFAULT)
        self.terminal.set_pty(self.pty)
        self.pty.spawn_async(
            self.directory,
            ["/bin/sh"],
            None,
            GLib.SpawnFlags.DO_NOT_REAP_CHILD,
            None,
            None,
            -1,
            None,
            self.ready
        )
        # self.pty.spawn_async(
        #     self.directory,
        #     ["/bin/python3"],
        #     None,
        #     GLib.SpawnFlags.DO_NOT_REAP_CHILD,
        #     None,
        #     None,
        #     -1,
        #     None,
        #     self.ready
        # )
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
        # self.log_output.connect('size-allocate', self.scroll_down)

        self.log_scroller = Gtk.ScrolledWindow()
        # self.log_scroller.connect('size-allocate', self.scroll_down)
        self.log_scroller.set_hexpand(True)
        self.log_scroller.set_vexpand(True)
        self.log_scroller.add(self.log_output)

        self.terminal_notebook.append_page(child=self.log_scroller, tab_label=Gtk.Label(label='Log'))

        # grid.attach(self.terminal_notebook, 0, 10, 7, 5)
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

            self.open_file(self.filename)
            self.page1.zoom_to_fit()
            self.get_graph_stats_filename(self.filename)
            self.import_graph(self.filename)
            self.terminal_notebook.set_current_page(1)
            self.get_graph_stats_initial(self.filename, self.G)
            self.read_probabilities()
            self.read_costs()
        else:
            chooser.destroy()

    def open_file(self, filename):
        try:
            fp = open(filename, 'rb')
            self.page1.set_dotcode(fp.read(), filename)
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

        layers = get_layers(self.G)
        self.all_equipment = sorted([get_node_name(self.G, node) for node in find_leaf_nodes(self.G, layers)])
        all_equipment_set = set(self.all_equipment)
        logging.info(f"All equipment: {[(index, component) for index, component in enumerate(self.all_equipment)]}")
        # logging.info(f"Configuration: {get_configuration(G, all_equipment)}")
        # logging.info(f"Configuration dict: {get_configuration_dict(G, all_equipment)}")
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
        layers = get_layers(G)
        self.all_equipment = sorted([get_node_name(G, node) for node in find_leaf_nodes(G, layers)])
        all_equipment_set = set(self.all_equipment)
        logging.info(f"All equipment: {[(index, component) for index, component in enumerate(self.all_equipment)]}")
        # logging.info(f"Configuration: {get_configuration(G, all_equipment)}")
        # logging.info(f"Configuration dict: {get_configuration_dict(G, all_equipment)}")

        logging.info("Analyze the configuration graph")
        (self.unique_graph_list, unique_node_lists, self.leaf_name_lists) = create_graph_list(G, verbose=False)

        # set button states
        self.button_check_isolation.set_sensitive(True)
        self.button_build_isolation.set_sensitive(True)
        self.button_check_recovery.set_sensitive(True)
        self.button_build_recovery.set_sensitive(True)
        self.analysis_done = True
        self.get_report()
        self.scroll_down2()

    def check_isolation(self, button):
        self.button_check_isolation.set_sensitive(False)
        self.terminal_notebook.set_current_page(1)
        logging.info("Checking isolation")
        self.non_isolable = check_isolability(self.all_equipment, self.leaf_name_lists)
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
            self.directory + "/../../temp/",
            self.directory + "/../../temp/prism_strategy_config.json")
        strategy_name = 'temp/prism_strategy.prism'
        self.feed_input(f'dtcontrol --input {strategy_name} --use-preset avg --benchmark-file benchmark.json --rerun\n')

    def prune_graph(self, button):
        self.feed_input(
            f'python3 src/mcts.py --modecosts {self.filename_mode_costs} --equipfailprobs {self.filename_fault_probs} --successorstokeep {self.children_to_keep_entry.get_text()} --simulationsize {self.simulations_per_node_entry.get_text()} {self.filename}\n')

    def ready(self, pty, task):
        pass
        # print('pty ', pty)

    def feed_input(self, text):
        text = bytearray(text, "utf-8")
        self.terminal.feed_child(text)

    def display_input(self, text):
        self.log_output.set_editable(True)
        self.log_output.get_buffer().insert_at_cursor(text)
        self.log_output.set_editable(False)

    def check_recovery(self, button):
        self.button_check_recovery.set_sensitive(False)
        self.terminal_notebook.set_current_page(1)
        logging.info("Checking recovery")
        self.non_recoverable = check_recoverability(self.G, self.all_equipment, self.leaf_name_lists)
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
        self.terminal_notebook.set_current_page(0)
        self.feed_input(f"python3 src/build_recovery.py {self.directory} {self.filename}\n")
        # start_time = time.time()
        # verbose = True
        # directory_name = self.directory + "/recovery_" + self.filename.split('/')[-1].split('.')[0] + "/"
        # if os.path.exists(directory_name):
        #     shutil.rmtree(directory_name)
        # os.makedirs(directory_name)
        # available_modes_filename = "available_modes.c"
        # print("Generate " + available_modes_filename)
        # generate_available_modes(self.G, self.unique_graph_list, directory_name + available_modes_filename, verbose)
        #
        # mode_switcher_filename = "mode_switcher.prism"
        # print("Generate " + mode_switcher_filename)
        # with open(directory_name + "mode_switcher.props", "w") as text_file:
        #     print('Pmax=? [ F "mode_selected" ]\n', file=text_file)
        # generate_mode_switcher(get_mode_indices(self.G), get_mode_indices_appended(self.G), directory_name + mode_switcher_filename)
        #
        # print("Model-checking with PRISM")
        # self.terminal_notebook.set_current_page(0)
        # prism_path = self.directory + "/prism/bin/prism"
        # mode_switcher_strategy_filename = "strategy_" + mode_switcher_filename
        # mode_switcher_properties_filename = "mode_switcher" + ".props"
        # command = run_prism(prism_path, directory_name + mode_switcher_filename, directory_name + mode_switcher_properties_filename,
        #                     directory_name + mode_switcher_strategy_filename, verbose)
        # self.feed_input(command)
        #
        # print("Generate config JSON")
        # self.terminal_notebook.set_current_page(1)
        # mode_switcher_config_filename = "strategy_" + mode_switcher_filename.split(".")[0] + "_config.json"
        # generate_config_json(get_mode_indices(self.G), get_mode_indices_appended(self.G), directory_name + mode_switcher_config_filename)
        #
        # print("Run dtControl and move decision tree")
        # self.terminal_notebook.set_current_page(0)
        # command = run_dtcontrol(directory_name + mode_switcher_strategy_filename, verbose)
        # self.feed_input(command)
        #
        # self.terminal_notebook.set_current_page(1)
        # reconfigure_filename = "reconfigure.c"
        # print("Generate " + reconfigure_filename)
        # actions_list = create_actions_list(directory_name + mode_switcher_strategy_filename)
        # print(actions_list)
        # generate_reconfigure(self.G, actions_list, get_mode_indices(self.G), get_mode_indices_appended(self.G), directory_name + reconfigure_filename)
        # print("This configuration took " + str(time.time() - start_time) + "s")

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
                    message += f"\tMode {get_node_name(self.G, mode)} is not single-fault tolerant\n"
        else:
            message += "Run Check Isolation and Check Recovery to include an assessment on the isolability of the components and recoverability of the modes.\n"

        if self.analysis_done:
            message += f"\nAssuming the component fault probabilities defined in ‘{self.filename_fault_probs.split('/')[-1]}’, the modes have these fault probabilities:\n"

            fault_probs = {mode: examine_successor(self.G, mode, self.get_probabilities()) for mode in find_root_nodes(self.G)}
            fault_probs_sorted = dict(sorted(fault_probs.items(), key=lambda item: item[1], reverse=True))
            for mode in fault_probs_sorted:
                message += f"\tThe fault probability for mode {get_node_name(self.G, mode)} is {100 * fault_probs_sorted[mode]:.5f} %\n"

        end_iter = self.report_text.get_buffer().get_end_iter()
        self.report_text.get_buffer().insert_markup(end_iter, message, -1)


def main():
    # window = MyDotWindow()
    window = MainWindow()
    window.set_default_size(1500, 900)
    window.connect('delete-event', Gtk.main_quit)
    window.show_all()

    logging.basicConfig(
        format="[%(levelname)s] %(funcName)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # self.logger = logging.getLogger("Example")
    handler = MyHandler(window.log_output)
    # self.handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    Gtk.main()

if __name__ == '__main__':
    main()
