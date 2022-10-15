def get_initialization(mode_indices, mode_indices_appended):
    initialization = """mdp

module select_mode
  // modes:
"""
    print("modes:")
    for mode in mode_indices_appended:
        initialization += "  // " + str(mode_indices_appended[mode]) + ": " + mode + "\n"
        print(str(mode_indices_appended[mode]) + ": " + mode)

    initialization += "  // " + str(len(mode_indices_appended)) + ": none_selected\n\n"

    initialization += "  // external input\n"
    initialization += "  desired_mode: [0.." + str(len(mode_indices_appended) - 1) + "];\n\n"
    initialization += "  // internal mode\n"
    initialization += "  mode: [0.." + str(len(mode_indices_appended)) + "];\n\n"

    initialization += "  // pointing modes\n"
    for mode in mode_indices:
        initialization += "  " + mode + ": [0..1];\n"

    # print(initialization)
    return initialization

def get_environment():
    environment = """
  // rotational velocity:
  // 0: low velocity, during precision pointing and maneuvering
  // 1: high velocity, unsuitable for precise pointing modes
  // 2: critical velocity, use detumbling
  rotational_velocity: [0..2];

  // battery level:
  // 0: sufficient
  // 1: low
  // 2: critical
  battery_level: [0..2];

"""
    return environment

def get_actions(mode_indices, mode_indices_appended):
    actions = ""
    # mode off
    actions += "  [select_mode_off] (battery_level=0 & rotational_velocity=0 & desired_mode=" + str(
        mode_indices_appended["off"]) + ")\n"

    if "DETUMB" in mode_indices.keys():
        actions += "    | (rotational_velocity=1 & DETUMB=0)\n"
    elif "DETUMB_MAG" in mode_indices.keys():
        actions += "    | (rotational_velocity=1 & DETUMB_MAG=0)\n"
    elif "DETUMB_RCS" in mode_indices.keys():
        actions += "    | (rotational_velocity=1 & DETUMB_RCS=0)\n"
    else:
        actions += "    | rotational_velocity=1\n"

    if "SUN" in mode_indices.keys():
        actions += "    | (battery_level=1 & SUN=0)\n"
    elif "SUN_MAG" in mode_indices.keys():
        actions += "    | (battery_level=1 & SUN_MAG=0)\n"
    elif "SUN_RCS" in mode_indices.keys():
        actions += "    | (battery_level=1 & SUN_RCS=0)\n"
    else:
        actions += "    | battery_level=1\n"

    actions += "    | rotational_velocity>1 | battery_level>1 \n"
    actions += "    | (rotational_velocity=1 & battery_level=1) \n"
    for mode in mode_indices:
        actions += "    | (battery_level=0 & rotational_velocity=0 & desired_mode=" + str(
            mode_indices[mode]) + " & " + mode + "=0)\n"

    # actions += "    | (" + " & ".join([f"{mode}=0" for mode in mode_indices]) + ")\n"

    actions += "    -> (mode'=" + str(mode_indices_appended["off"]) + ");\n\n"

    # nominal modes
    for mode in mode_indices:
        actions += "  [select_mode_" + mode + "] (battery_level=0 & rotational_velocity=0 & desired_mode=" + str(
            mode_indices[mode]) + " & " + mode + "=1)\n"
        if mode == "DETUMB":
            actions += "    | (battery_level=0 & rotational_velocity=1 & DETUMB=1)\n"
        elif mode == "DETUMB_MAG":
            actions += "    | (battery_level=0 & rotational_velocity=1 & DETUMB_MAG=1)\n"
        elif mode == "DETUMB_RCS":
            actions += "    | (battery_level=0 & rotational_velocity=1 & DETUMB_RCS=1)\n"
        if mode == "SUN":
            actions += "    | (battery_level=1 & rotational_velocity=0 & SUN=1) \n"
        elif mode == "SUN_MAG":
            actions += "    | (battery_level=1 & rotational_velocity=0 & SUN_MAG=1) \n"
        elif mode == "SUN_RCS":
            actions += "    | (battery_level=1 & rotational_velocity=0 & SUN_RCS=1) \n"
        actions += "    -> (mode'=" + str(mode_indices[mode]) + ");\n"

    # print(actions)
    return actions

def get_endmodule(mode_indices_appended):
    endmodule = """endmodule

label "mode_selected" = mode!=""" + str(len(mode_indices_appended)) + """;
label "desired_mode_selected" = mode=desired_mode;
"""
    # print(endmodule)
    return endmodule

def get_initial_states(mode_indices, mode_indices_appended):
    initial_states = "init\n  mode=" + str(len(mode_indices_appended))
    initial_states += " & desired_mode>=0 & rotational_velocity>=0 & battery_level>=0\n  "
    for mode in mode_indices:
        initial_states += "& " + mode + ">=0 "
    initial_states += "\nendinit"
    # print(initial_states)
    return initial_states

def generate_mode_switcher(mode_indices, mode_indices_appended, filename):
    template = get_initialization(mode_indices, mode_indices_appended) \
               + get_environment() \
               + get_actions(mode_indices, mode_indices_appended) \
               + get_endmodule(mode_indices_appended) \
               + get_initial_states(mode_indices, mode_indices_appended)
    # print(template)
    with open(filename, "w") as text_file:
        # with open("mode_switcher_template.prism", "w") as text_file:
        print(template, file=text_file)

