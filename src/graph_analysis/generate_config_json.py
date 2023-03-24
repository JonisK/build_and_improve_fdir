import json
import os
import shutil


def generate_config_json(mode_indices, mode_indices_appended, filename):
    config = {"x_column_types": {"categorical": []},
              "y_column_types": {},
              "x_column_names": ["mode", "desired_mode"],
              "x_category_names": {}
              }
    config["x_column_types"]["categorical"] = list(range(len(mode_indices) + 4))
    config["x_column_names"] += list(mode_indices)
    config["x_column_names"] += ["rotational_velocity", "battery_level"]
    config["x_category_names"]["mode"] = list(mode_indices_appended) + ["none_selected"]
    config["x_category_names"]["desired_mode"] = list(mode_indices_appended)
    for variable in mode_indices:
        config["x_category_names"][variable] = ["not_available", "available"]
    config["x_category_names"]["rotational_velocity"] = ["low", "high", "critical"]
    config["x_category_names"]["battery_level"] = ["sufficient", "low", "critical"]

    with open(filename, "w") as text_file:
        print(json.dumps(config, indent=4), file=text_file)


def generate_config_json_isolation(all_equipment, directory_name, filename, hidden_variable=False):
    num_components = (len(all_equipment) + 1) if hidden_variable else len(all_equipment)
    config = {"x_column_types": {"categorical": list(range(num_components))},
              "y_column_types": {},
              "x_column_names": all_equipment + (["faulty_component"] if hidden_variable else []),
              "x_category_names": {}
              }
    for component in all_equipment:
        config["x_category_names"][component] = ["available", "suspicious"]
    if hidden_variable:
        config["x_category_names"]["faulty_component"] = all_equipment

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    #     shutil.rmtree(directory_name)
    # os.makedirs(directory_name)

    with open(filename, "w") as text_file:
        print(json.dumps(config, indent=4), file=text_file)
