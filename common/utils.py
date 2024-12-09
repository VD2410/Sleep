import yaml
from common.definitions import ROOT_DIR

def load_config_file(config_path=None):
    """
    This method reads config.yaml file

    Inputs:
        None

    Returns:
        config_data: Dict with Config.yaml data
    """
    if config_path:
        with open(f'{ROOT_DIR}/{config_path}') as file:
            config_data = yaml.safe_load(file)
    else:
        with open(f'{ROOT_DIR}/config/config.yaml') as file:
            config_data = yaml.safe_load(file)
    return config_data