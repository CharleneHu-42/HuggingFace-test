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


def collect_memory(name: str, cmd: str, output: str, deadline):
    import csv

    import psutil

    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
    backend = "ccl"
    dist.init_process_group(backend)

    if dist.get_rank() == 0:
        f = open(output + name + ".csv", mode="w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "pid",
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
            if len(p.cmdline()) > 1 and cmd in (p.cmdline()[1]):
                a = {
                    "rank": dist.get_rank(),
                    "pid": p.pid,
                    "memory-rss": p.memory_info().rss * 1.0 / 1024 / 1024 / 1024,
                    "memory-vms": p.memory_info().vms * 1.0 / 1024 / 1024 / 1024,
                    "memory-data": p.memory_info().data * 1.0 / 1024 / 1024 / 1024,
                    "cpu-percent": p.cpu_percent(),
                }
                logger.info(f"{a}")
                if dist.get_world_size() > 1:
                    output = [None for _ in range(dist.get_world_size())]
                    dist.gather_object(
                        a,
                        output if dist.get_rank() == 0 else None,
                        dst=0,
                    )
                    if dist.get_rank() == 0:
                        for item in output:
                            csv_writer.writerow(item)
                else:
                    csv_writer.writerow(a)
                find = True
        time.sleep(1)
    if dist.get_rank() == 0:
        f.close()

    if dist.get_world_size() > 1:
        dist.barrier()


def run_and_collect_memory(case: dict, arg: argparse.Namespace):
    deadline = time.time() + case["timeout"]
    logger.info("start collect_memory process")
    collect_memory(case["title"], case["cmd"], arg.output, deadline)
    return


def main():
    parser = init_arguments_parser()
    args = parser.parse_args()
    yaml_content = parse_yaml(args)
    for case in yaml_content["testcases"]:
        run_and_collect_memory(case["case"], args)


if __name__ == "__main__":
    main()
