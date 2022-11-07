import argparse
import csv
import logging
import os
import subprocess
import time
from multiprocessing import Process

import yaml

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    level=logging.INFO,
    datefmt="[%X]",
)
logger = logging.getLogger(__name__)


def parse_yaml(arg: argparse.Namespace):
    content = None
    with open(arg.input, encoding="utf-8") as f:
        content = yaml.safe_load(f.read())
    return content


def init_arguments_parser():
    parser = argparse.ArgumentParser(description="memory test")
    parser.add_argument(
        "-i", "--input", default="conf.yml", help="path of case configuration yaml file"
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0")
    parser.add_argument(
        "-o",
        "--output",
        default="/tmp/",
        help="output dir, default is /tmp/",
    )
    return parser


def collect_memory(eval_pid: int, name: str, output: str):
    import csv

    import matplotlib.pyplot as plt
    import psutil
    from psutil import Process

    f = open(output + name + ".csv", mode="w", encoding="utf-8", newline="")

    csv_writer = csv.DictWriter(
        f,
        fieldnames=[
            "rss",
            "vms",
            "data",
        ],
    )
    csv_writer.writeheader()
    find = True
    rss = list()

    while find:
        find = False
        for p in psutil.process_iter():
            if p.pid == eval_pid and p.status() != "zombie":
                a = {
                    "rss": p.memory_info().rss,
                    "vms": p.memory_info().vms,
                    "data": p.memory_info().data,
                }
                rss.append(1.0 * p.memory_info().rss / (1024 * 1024 * 1024))
                csv_writer.writerow(a)
                find = True
                time.sleep(1)

    plt.plot(rss)
    plt.title(name)
    plt.xlabel("second")
    plt.ylabel("rss GB")
    plt.savefig(output + name + ".png")
    f.close()


def run_and_collect_memory(case: dict, arg: argparse.Namespace):
    p_list = []
    eval_p = subprocess.Popen(case["cmd"], shell=True, cwd=case["work_dir"])
    time.sleep(case["warmup"])
    p_memory = Process(
        target=collect_memory, args=(eval_p.pid, case["title"], arg.output)
    )
    p_memory.start()
    logger.info("start collect_memory process")
    deadline = time.time() + case["timeout"]
    while time.time() < deadline and eval_p.poll() is None:
        time.sleep(1)
    if eval_p.poll() is None:
        eval_p.kill()
    p_memory.join()
    eval_p.wait()
    return


def main():
    parser = init_arguments_parser()
    args = parser.parse_args()
    yaml_content = parse_yaml(args)
    for case in yaml_content["testcases"]:
        run_and_collect_memory(case["case"], args)


if __name__ == "__main__":
    main()
