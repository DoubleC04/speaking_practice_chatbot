import logging
import yaml
import os

def setup_logging():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    log_level = getattr(logging, config["logging"]["level"], logging.INFO)
    log_file = config["logging"].get("file", "logs/app.log")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


