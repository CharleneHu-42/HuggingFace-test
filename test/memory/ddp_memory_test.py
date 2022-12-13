import argparse
import csv
import logging
import os
import re
import subprocess
import time
from multiprocessing import Process

import oneccl_bindings_for_pytorch
import torch.distributed as dist
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


def collect_memory(eval_pid: int, name: str, output: str, deadline):
    import csv
    import psutil

    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    backend = "ccl"
    dist.init_process_group(backend)

    if dist.get_rank() == 0:
        f = open(output + name + ".csv", mode="w", encoding="utf-8", newline="")

        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "memory-rss",
                "memory-vms",
                "memory-data",
                "cpu-percent",
            ],
        )
        csv_writer.writeheader()

    find = True

    while find and time.time() < deadline:
        find = False
        for p in psutil.process_iter():
            if p.pid == eval_pid:
                a = {
                    "rank": dist.get_rank(),
                    "memory-rss": p.memory_info().rss,
                    "memory-vms": p.memory_info().vms,
                    "memory-data": p.memory_info().data,
                    "cpu-percent": p.cpu_percent(),
                }
                output = [None for _ in range(dist.get_world_size())]
                dist.gather_object(
                    a,
                    output if dist.get_rank() == 0 else None,
                    dst=0,
                )
                if dist.get_rank() == 0:
                    for item in output:
                        csv_writer.writerow(item)
                find = True
                time.sleep(1)
    if dist.get_rank() == 0:
        f.close()
    dist.barrier()


def run_and_collect_memory(case: dict, arg: argparse.Namespace):
    current_env = os.environ.copy()
    current_env["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
    eval_p = subprocess.Popen(
        re.split("\s+", case["cmd"]), cwd=case["work_dir"], env=current_env
    )
    time.sleep(case["warmup"])
    deadline = time.time() + case["timeout"]
    logger.info("start collect_memory process")
    collect_memory(eval_p.pid, case["title"], arg.output, deadline)
    if eval_p.poll() is None:
        eval_p.kill()
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
