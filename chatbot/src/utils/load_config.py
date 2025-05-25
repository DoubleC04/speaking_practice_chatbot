import yaml

def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)