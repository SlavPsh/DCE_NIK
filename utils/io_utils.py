import os
import shutil
import toml
import logging
from datetime import datetime


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    return config


def setup_logging(config, output_dir, job='training'):
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, f'{job}.log')
    # Clear existing handlers (important for repeated calls in sweep runs)
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def unique_output_dir(config, run_name='run'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output']['base_path'], f"{timestamp}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def copy_config_to_output(config_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_path, output_dir)
