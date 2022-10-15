import regex
import json


def create_actions_list(input_strategy):
    action_names = []
    with open(input_strategy) as source:
        lines = source.readlines()
        for line in lines:
            current_action = regex.findall("(?<=\d+:)[a-zA-Z_]+", line)[0]
            if current_action not in action_names:
                action_names.append(current_action)
    return action_names


def generate_actions_list(mode_switcher_strategy_filename, action_list_filename, verbose):
    actions_list = create_actions_list(mode_switcher_strategy_filename)
    print(actions_list) if verbose else None
    with open(action_list_filename, 'w') as destination:
        destination.write(json.dumps(actions_list, indent=4))
