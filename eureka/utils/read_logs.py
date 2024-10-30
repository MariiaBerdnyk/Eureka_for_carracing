import os

from eureka.eureka import RL_BASELINE_ROOT_DIR

LOGS_DIRECTORY = f"{RL_BASELINE_ROOT_DIR}/wandb"
PATH_FROM_DIR_TO_LOG_FILE = "files/output.log"

def

def get_last_logs():
    dirs = os.listdir(LOGS_DIRECTORY)
    dir_with_logs = dirs[-1]
    full_logs_path = os.path.join(LOGS_DIRECTORY, dir_with_logs, PATH_FROM_DIR_TO_LOG_FILE)
    with open(full_logs_path, "w") as log_file:
