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
    with open("command_history.jsonl", "r") as f:
        for line in f.readlines():
            entry = json.loads(line.rstrip())
            entries[entry["job_id"]] = entry
    return entries

def get_running_jobs():
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
    parser.add_argument('--hist', action='store_true', help='list recently completed jobs')
    parser.add_argument('--days', default=3, type=int, help='days of history')


    args = parser.parse_args()

    if args.hist:
        entries = read_command_log()

        # workaround for bad entry in my log
        if None in entries:
            del entries[None]

        # find all jobs in the last N days
        report = collections.defaultdict(list)
        now = datetime.now()
        delta = timedelta(days=args.days)
        for k in reversed(sorted(entries.keys())):
            entry = entries[k]
            timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            if now - timestamp <= delta:
                report[entry["model"]].append(entry)

        running_jobs = set(get_running_jobs())
        for model in sorted(report.keys()):
            entries = sorted(report[model], key=lambda x: x["eval"])

            print(f"Model: {model}")
            print(f"Results dir: {os.path.dirname(entries[0]['output_file'])}")

            completed = []
            incomplete = []
            running = []
            for entry in entries:
                if entry["job_id"] in running_jobs:
                    running.append(entry)
                elif os.path.exists(entry["output_file"]):
                    completed.append(entry)
                else:
                    incomplete.append(entry)
            if completed:
                print(f"Completed:")
                for entry in completed:
                    print(f"    {entry['eval']}")
            if running:
                print("Running/Queued:")
                for entry in running:
                    print(f"    {entry['eval']} {entry['job_id']}")

            # first let's filter these down to evals that have not
            # subsequently succeeded.
            complete_failures = []
            if incomplete:
                succeeded = set([i["eval"] for i in completed])
                complete_failures = [i for i in incomplete if i["eval"] not in succeeded]

            if complete_failures:
                # these jobs are not in queue and no results exist
                print("Failed:")
                for entry in complete_failures:
                    if os.path.exists(entry['err_log']):
                        print(f"    {entry['eval']} {entry['err_log']}")
                    else:
                        print(f"    {entry['eval']} no err log, cancelled?")
            print("")
            
                
            
            
        return


    jobs = []
    while True:
        entries = read_command_log()
        new_jobs = get_running_jobs()
        for job in new_jobs:
            if job not in jobs:
                # newly running job
                entry = entries.get(job, None)
                if entry is None:
                    # unrecognized job
                    continue
                comment = entry.get("comment", None)
                if comment is not None:
                    print(f'{job} {entry["model"]} {entry["eval"]} is running. ({comment})')
                else:
                    print(f'{job} {entry["model"]} {entry["eval"]} is running.')
                if os.path.exists(entry["err_log"]):
                    with open(entry["err_log"], "r") as f:
                        lines = f.readlines()
                        print(lines[-1])
                
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
