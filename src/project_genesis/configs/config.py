import os
import yaml


def get_config(name: str) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), name)
    config_files = os.listdir(config_path)

    config = {}

    for file in config_files:
        with open(os.path.join(config_path, file)) as f:
            config.update(yaml.safe_load(f))

    return config