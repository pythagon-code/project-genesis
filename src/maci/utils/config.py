from pathlib import Path
from ruamel.yaml import YAML


def get_config(directory: str) -> dict:
    dir_path = Path(directory)
    yaml = YAML()
    config = {}
    for child in dir_path.glob("*.yaml"):
        with open(child) as f:
            config.update(yaml.load(f))

    return config