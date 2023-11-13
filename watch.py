import argparse
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
    with open("command_history.jsonl", "r") as f:
        for line in f.readlines():
            entry = json.loads(line.rstrip())
            entries[entry["job_id"]] = entry
    return entries

def get_running_jobs():
    
    #username = os.getlogin()
    command = ['squeue', '--me', '-o', '%i %T %j']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(f"Error: {error}")
        return None

    running_jobs = []
    output_lines = output.decode('utf-8').split('\n')[1:]  # Skip the header line
    for line in output_lines:
        if line:  # Skip empty lines
            job_id, status, job_name = line.split()
            if status in STATUS_RUNNING:
                running_jobs.append(job_id)
    return running_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='run once and exit')
    args = parser.parse_args()


    jobs = []
    while True:
        entries = read_command_log()
        new_jobs = get_running_jobs()
        for job in new_jobs:
            if job not in jobs:
                # newly running job
                entry = entries.get(job, None)
                if entry is None:
                    print(f"Unrecognized job {job}, ignoring...")
                    continue
                comment = entry.get("comment", None)
                if comment is not None:
                    print(f'{job} {entry["model"]} {entry["eval"]} is running. ({comment})')
                else:
                    print(f'{job} {entry["model"]} {entry["eval"]} is running.')
        for job in jobs:
            if job not in new_jobs:
                # newly exited job
                entry = entries.get(job, None)
                if entry is None:
                    continue
                comment = entry.get("comment", None)
                if comment is not None:
                    print(f'{job} {entry["model"]} {entry["eval"]} has completed. ({comment}')
                else:
                    print(f'{job} {entry["model"]} {entry["eval"]} has completed.')
                if os.path.exists(entry["output_file"]):
                    print("Output follows =========")
                    with open(entry["output_file"], "r") as f:
                        results = f.read()
                    print(results)
                else:
                    print(f'No output found! Error log: {entry["err_log"]}')

        jobs = new_jobs

        if args.once:
            break

        time.sleep(60)

if __name__ == "__main__":
    main()
