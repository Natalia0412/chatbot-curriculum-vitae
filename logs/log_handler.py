import logging
import os


MAX_LOGS = 5

def create_log_directory(log_directory = 'logs'):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)


def get_log_directory(log_file_name = 'log-app.py', log_directory = 'logs'):
    return os.path.join(log_directory, log_file_name)


def read_logs(log_file_name = 'log-app.py'):
    if os.path.exists(log_file_name):
        with open(log_file_name, 'r') as f:
            logs = f.readlines()
        return [log.strip() for log in logs]
    return []

def write_logs(logs, log_file_name = 'log-app.py'):
    with open(log_file_name, 'w') as f:
        for log in logs:
            f.write(log + '\n')

def logs_recording (new_log,log_file_name = 'log-app.py'):
    logs = read_logs(log_file_name)
    logs.append(new_log)
    logs = logs[-MAX_LOGS:]
    write_logs(logs)


