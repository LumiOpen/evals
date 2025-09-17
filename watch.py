import argparse
import collections
from datetime import datetime, timedelta
import json
import os
import subprocess
import time
from evals.slurm import read_command_log, get_jobs


MIN_EVAL_COL_WIDTH = len("humaneval-unstripped_pass@10") + 4


def _format_elapsed(delta: timedelta) -> str:
    """Return a short human readable description like '2 days 3 hours'."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        total_seconds = 0

    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60

    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours and len(parts) < 2:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if not parts and minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        parts.append("less than a minute")

    return " ".join(parts)


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
                entry["_queued_dt"] = timestamp
                report[entry["model"]].append(entry)

        running_jobs = set(get_jobs())
        for model in sorted(report.keys()):
            entries = sorted(report[model], key=lambda x: x["eval"])

            print(f"Model: {model}")
            print(f"Results dir: {os.path.dirname(entries[0]['output_file'])}")

            # we deduplicated completed evals by eval_name because the logic
            # below can't differentiate between which of a number of possible
            # jobs actually was the one that completed, and there's no real
            # reason to care so long as at least one of them has.
            completed = {}
            incomplete = []
            running = []
            for entry in entries:
                if entry["job_id"] in running_jobs:
                    running.append(entry)
                elif os.path.exists(entry["output_file"]):
                    completed[entry["eval"]] = entry
                else:
                    incomplete.append(entry)
            if completed:
                print(f"Completed:")
                for eval_name in sorted(completed.keys()):
                    print(f"    {eval_name}")
            if running:
                print("Running/Queued:")
                eval_col_width = max(
                    max((len(entry["eval"]) for entry in running), default=0) + 2,
                    MIN_EVAL_COL_WIDTH,
                )
                job_col_width = max((len(str(entry["job_id"])) for entry in running), default=0)
                if job_col_width:
                    job_col_width += 1
                for entry in running:
                    queued_dt = entry.get("_queued_dt")
                    if queued_dt is None:
                        queued_dt = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                    age = _format_elapsed(now - queued_dt)
                    eval_name = entry["eval"].ljust(eval_col_width)
                    job_id_str = str(entry["job_id"])
                    job_id = job_id_str.ljust(job_col_width) if job_col_width else f"{job_id_str} "
                    print(f"    {eval_name}{job_id} (queued {age} ago)")

            # first let's filter these down to evals that have not
            # subsequently succeeded.
            complete_failures = []
            if incomplete:
                succeeded = set(completed.keys())
                complete_failures = [i for i in incomplete if i["eval"] not in succeeded]

            if complete_failures:
                # these jobs are not in queue and no results exist
                print("Failed:")
                eval_col_width = max(
                    max((len(entry["eval"]) for entry in complete_failures), default=0) + 2,
                    MIN_EVAL_COL_WIDTH,
                )
                for entry in complete_failures:
                    if os.path.exists(entry['err_log']):
                        eval_name = entry["eval"].ljust(eval_col_width)
                        print(f"    {eval_name}{entry['err_log']}")
                    else:
                        eval_name = entry["eval"].ljust(eval_col_width)
                        print(f"    {eval_name}no err log, cancelled?")
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
