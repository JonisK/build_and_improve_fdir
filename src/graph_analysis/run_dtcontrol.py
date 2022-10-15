import os


def run_dtcontrol(mode_switcher_strategy_filename, verbose):
    command = "dtcontrol --input " + mode_switcher_strategy_filename + \
               " --use-preset avg --rerun --benchmark-file benchmark.json" + \
               " --timeout 6h"
    print(command)
    if not verbose:
        command += " > /dev/null 2>&1"
    return command