# built-in libraries
import logging
import re

# Gtk3
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

# third-party libraries
from tabulate import tabulate
import xdot

# project-specific libraries
from graph_analysis.graph_analysis import get_node_name, get_node_id


class InfoDialog(Gtk.Dialog):
    def __init__(self, info, nx_graph, all_equipment, component_lists,
                 configuration_list):
        super().__init__(title="Extended State Description")
        self.add_buttons(Gtk.STOCK_OK, Gtk.ResponseType.OK)
        self.set_default_size(700, 500)
        content = Gtk.TextView()
        content.set_editable(False)
        content.set_monospace(True)
        content_scroller = Gtk.ScrolledWindow()
        content_scroller.set_hexpand(True)
        content_scroller.set_vexpand(True)
        content_scroller.add(content)
        description = ""
        value_string = re.search(r"\[([0-9, ]+)\]",
                                 info.split('\\n')[0])
        values = value_string.group(1).replace(" ", "").split(',')
        mode_and_configuration = info.split('\\n')[1]
        if mode_and_configuration == 'Done':
            headers = ['component', 'value']
            isolated_component = all_equipment[values.index('1')]
        else:
            mode = mode_and_configuration.split('_')[0]
            mode_id = get_node_id(nx_graph, mode)
            configuration_index = int(mode_and_configuration.split('_')[1])
            headers = ['component', 'value', 'usage in next mode']
            used_components = component_lists[mode_id][configuration_index]

        suspects = []
        for component, value in zip(all_equipment, values):
            if value == '1':
                value_string = 'suspicious'
            else:
                value_string = 'available'
            if not mode_and_configuration == 'Done':
                if component in used_components:
                    usage = 'used'
                else:
                    usage = 'not used'
                suspects.append([component, value_string, usage])
            else:
                suspects.append([component, value_string])
        description += tabulate(suspects, headers=headers)
        description += "\n\n"

        if mode_and_configuration == 'Done':
            description += f"Component {isolated_component} isolated."
        else:
            description += f"Mode {mode} in configuration "
            description += f"{configuration_index} uses the components "
            description += ', '.join(used_components)
            description += ".\n"
            configuration = configuration_list[mode_id][configuration_index]
            for assembly in configuration:
                node_id = list(nx_graph.predecessors(assembly))[0]
                assembly_name = get_node_name(nx_graph, node_id)
                active_children = [list(nx_graph.successors(assembly))[index]
                                   for index in configuration[assembly]]
                active_child_names = sorted([get_node_name(nx_graph, child)
                                             for child in active_children])
                description += f"Assembly {assembly_name} uses children "
                description += f"{', '.join(active_child_names)}.\n"

        end_iter = content.get_buffer().get_end_iter()
        content.get_buffer().insert_markup(end_iter, description, -1)
        box = self.get_content_area()
        box.add(content_scroller)
        self.show_all()


class MyDotWidget(xdot.ui.DotWidget):
    def __init__(self):
        xdot.ui.DotWidget.__init__(self)
        self.connect('clicked', self.on_url_clicked)
        self.nx_graph = None
        self.all_equipment = []
        self.component_lists = {}
        self.configuration_list = {}

    def set_graph_and_all_equipment(self, graph, all_equipment):
        self.nx_graph = graph
        self.all_equipment = all_equipment

    def set_leaf_name_and_configuration_list(self,
                                             component_lists,
                                             configuration_list):
        self.component_lists = component_lists
        self.configuration_list = configuration_list

    def on_url_clicked(self, widget, url, event):
        dialog = InfoDialog(info=url,
                            nx_graph=self.nx_graph,
                            all_equipment=self.all_equipment,
                            component_lists=self.component_lists,
                            configuration_list=self.configuration_list)
        dialog.connect('response', lambda dialog, response: dialog.destroy())
        return True


def add_url_property(graph_filename):
    dotcode = ""
    with open(graph_filename, 'r') as input:
        for line in input.readlines():
            if '[' in line and not '->' in line:
                # This is a node, so we will add its ID as a URL property
                node_name = line.replace(';', '')\
                                .replace('\t', '')\
                                .replace('\n', '')
                dotcode += f"\t{node_name} [URL={node_name}];\n"
            else:
                dotcode += line
    return dotcode
