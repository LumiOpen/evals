import argparse
import collections
from datetime import datetime, timedelta
import json
import os
import subprocess
import time

STATUS_RUNNING = [
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESV_DEL_HOLD",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESIZING",
    "SIGNALING",
    "SPECIAL_EXIT",  # unclear if this should be treated as a pending state
    "STAGE_OUT",
    "STOPPED",       # unclear if this should be treated as a pending state
    "SUSPENDED",     # unclear if this should be treated as a pending state
]

# in all of these states, the job is not running and will not run again
# so it would be safe to restart if necessary.
STATUS_NOT_RUNNING = [
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
]

def read_command_log():
    entries = {}
    if os.path.exists("command_history.jsonl"):
        with open("command_history.jsonl", "r") as f:
            for line in f.readlines():
                entry = json.loads(line.rstrip())
                entries[entry["job_id"]] = entry
        return entries

def get_jobs():
    command = ['squeue', '--me', '-o', '%i %T %j']
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            print(f"Error: {error}")
            return []

        running_jobs = []
        output_lines = output.decode('utf-8').split('\n')[1:]  # Skip the header line
        for line in output_lines:
            if line:  # Skip empty lines
                job_id, status, job_name = line.split()
                if status in STATUS_RUNNING:
                    running_jobs.append(job_id)
        return running_jobs
    except FileNotFoundError:
        # squeue not available (not on SLURM system)
        return []

def identify_scheduled_tasks():
    log_entries = read_command_log()
    identified_jobs = dict()
    for job_id in get_jobs():
        if job_id in log_entries:
            entry = log_entries[job_id]
            # Include backend in the key to distinguish between HF and vLLM jobs
            backend = entry.get("backend", "hf")  # Default to "hf" for old entries
            identified_jobs[(entry["model"], entry["eval"], backend)] = entry
    return identified_jobs
        



    


