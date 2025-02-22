import argparse
import collections
from datetime import datetime, timedelta
import json
import os
import subprocess
import time
from evals.slurm import read_command_log, get_jobs


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

        running_jobs = set(get_jobs())
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
        new_jobs = get_jobs()
        for job in new_jobs:
            if job not in jobs:
                # newly running job
                entry = entries.get(job, None)
                if entry is None:
                    # unrecognized job
                    continue
                if os.path.exists(entry["err_log"]):
                    print(f'{job} {entry["model"]} {entry["eval"]} is running.')
                    with open(entry["err_log"], "r") as f:
                        lines = f.readlines()
                        if lines:
                            print(lines[-1])
                        else:
                            print("<no log entries>")
                else:
                    print(f'{job} {entry["model"]} {entry["eval"]} is queued.')

                
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
