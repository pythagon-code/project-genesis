from pathlib import Path
import yaml


def get_config(directory: str) -> dict:
    dir_path = Path(directory)
    config = {}
    for child in dir_path.glob("*.yaml"):
        with open(child) as f:
            config.update(yaml.safe_load(f))

    return config