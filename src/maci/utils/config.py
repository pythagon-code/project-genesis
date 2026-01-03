from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat


def get_config(directory: str) -> dict:
    dir_path = Path(directory)
    config = {}
    yaml = YAML()
    for child in dir_path.glob("*.yaml"):
        with open(child) as f:
            config.update(yaml.load(f))

    return config