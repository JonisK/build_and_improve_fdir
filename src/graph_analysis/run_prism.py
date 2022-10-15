import os


def run_prism(prism_path, mode_switcher_filename, mode_switcher_properties_filename, mode_switcher_strategy_filename, verbose):
    command = prism_path + " " \
              + mode_switcher_filename + " "\
              + mode_switcher_properties_filename + " " \
              + "-exportstrat '" + mode_switcher_strategy_filename + ":type=actions' " \
              + "-exportstates '" + mode_switcher_strategy_filename.split(".")[0] + "_states.prism'"
    print(command) if verbose else None
    if not verbose:
        command += " > /dev/null"
    return command

