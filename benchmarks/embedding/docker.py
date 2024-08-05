import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from models import Model
from utils import HTTP_PROXY, HTTPS_PROXY, NO_PROXY, PORT, Namespace

DATA_VOLUME = Path.home() / ".cache" / "huggingface" / "hub"
SUPPORTED_PLATFORMS = ["gaudi2", "a100", "cpu"]
JSON_OUTPUT = True


def add_docker_args(parser: argparse.ArgumentParser):
    """
    Adds the required docker arguments to the argument parser.
    """
    parser.add_argument(
        "--docker_container",
        type=str,
        default="tei-gaudi",
        help="Name or hash of the docker container to launch.",
    )
    parser.add_argument(
        "--data_volume",
        type=Path,
        default=DATA_VOLUME,
        help="Data cache for Huggingface Models.",
    )
    parser.add_argument(
        "--truncate",
        default=False,
        action="store_true",
        help="Truncate sentences to model token limit.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="gaudi2",
        choices=SUPPORTED_PLATFORMS,
        help="Plotform that the benchmark is running on.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Switches log to debug mode.",
    )
    parser.add_argument(
        "--docker_port", type=int, default=PORT, help="Port for sending requests"
    )
    parser.add_argument(
        "--docker_env_vars",
        "-e",
        type=str,
        nargs="*",
        help="Docker environment variables in the format `--docker_env_vars <var1>=<val1> <var2>=<val2> ...`",
    )
    parser.add_argument(
        "--trust_remote_code",
        default=False,
        action="store_true",
        help="Enables remote code from model. Required for some models",
    )
    parser.add_argument(
        "--max_client_batch_size",
        type=int,
        default=32,
        help="Batch size allowed by client",
    )


@dataclass
class DockerArgs(Namespace):
    docker_container: str
    data_volume: Path
    truncate: bool
    platform: str
    debug: bool
    docker_port: int
    docker_env_vars: Optional[List[str]]
    trust_remote_code: bool
    max_client_batch_size: int

    def validate(self):
        self.data_volume = self.data_volume.resolve()
        self.platform = self.platform.lower()

        if self.platform not in SUPPORTED_PLATFORMS:
            raise NotImplementedError(f"Platform {self.platform} not implemented")

        if self.docker_env_vars is not None:
            for env_var in self.docker_env_vars:
                components = env_var.split("=")
                if len(components) != 2:
                    raise ValueError(
                        f"Environment variable {env_var} has incompatible components: expected exactly 1 `=`. Got {len(components) - 1}."
                    )
                if len(components[0]) == 0:
                    raise ValueError(
                        f"Environment variable {env_var} does not have a valid variable assigned."
                    )

        super().validate()

    @property
    def is_habana(self) -> bool:
        return self.platform == "gaudi2"

    @property
    def is_nvidia(self) -> bool:
        return self.platform in ["a100"]


class DockerProcess:
    def __init__(
        self,
        model: Model,
        stdout,
        docker_args: DockerArgs,
        error_callback: Optional[Callable] = None,
        name_prefix: Optional[str] = None,
    ):
        self.name = f"{model.file_str}_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if name_prefix is not None:
            self.name = f"{name_prefix}_{self.name}"
        cmd = [
            "docker",
            "run",
            "--name",
            self.name,
            "--rm",
            "-p",
            f"{docker_args.docker_port}:80",
            "-v",
            f"{docker_args.data_volume}:/data",
            "--ipc=host",
            "-e",
            f"HTTP_PROXY={HTTP_PROXY}",
            "-e",
            f"HTTPS_PROXY={HTTPS_PROXY}",
            "-e",
            f"NO_PROXY={NO_PROXY}",
            "-e",
            "MAX_WARMUP_SEQUENCE_LENGTH=512",
        ]
        if docker_args.trust_remote_code or model.remote_code_required:
            cmd.extend(["-e", "TRUST_REMOTE_CODE=1"])
        if docker_args.docker_env_vars is not None:
            for env_var in docker_args.docker_env_vars:
                cmd.extend(["-e", env_var])
        if docker_args.debug:
            cmd.extend(["-e", "LOG_LEVEL=debug"])
        if docker_args.is_habana:
            cmd.extend(
                [
                    "--runtime=habana",
                    "-e",
                    "HABANA_VISIBLE_DEVICES=all",
                    "-e",
                    "OMPI_MCA_btl_vader_single_copy_mechanism=none",
                    "--cap-add=sys_nice",
                ]
            )
        elif docker_args.is_nvidia:
            cmd.extend(["--gpus", "all"])
        # Docker container args
        cmd.extend(
            [
                docker_args.docker_container,
                "--model-id",
                model.name,
                "--pooling",
                "cls",
                f"--max-client-batch-size={docker_args.max_client_batch_size}",
            ]
        )
        if JSON_OUTPUT:
            cmd.append("--json-output")
        if model.rev is not None:
            cmd.extend(["--revision", model.rev])
        if docker_args.truncate:
            cmd.append("--auto-truncate")
        self.cmd = cmd
        self.stdout = stdout
        self.error_callback = error_callback

    def run(self):
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.DEVNULL,
            stdout=self.stdout,
        )
        logger.info(f"Started docker process with {self.cmd}")

    def close(self):
        self.stdout.close()
        kill_process = subprocess.Popen(["docker", "kill", self.name])
        kill_process.wait()
        self.process.wait()

    def wait_until_ready(
        self,
        log_file: Path,
        timeout: Optional[float] = None,
        check_interval: float = 0.5,
        print_interval: float = 10.0,
    ):
        start_time = time.time()
        elapsed_time = 0
        last_line_read = 0
        last_print_time = time.time()

        while self.process.poll() is None and (
            timeout is None or elapsed_time < timeout
        ):
            # Check logs
            with open(log_file, "r") as f:
                lines = f.readlines()
            for line in lines[last_line_read:]:
                message = json.loads(line).get("message", "").strip()
                if message == "Ready":
                    logger.info(f"Docker environment ready in {elapsed_time}.")
                    return
            last_line_read = len(lines)

            # Warnings
            time_since_print = time.time() - last_print_time
            if elapsed_time > 30 and time_since_print > print_interval:
                logger.info(f"Docker has taken {elapsed_time:.5f} seconds to run...")
                last_print_time = time.time()

            time.sleep(check_interval)
            elapsed_time = time.time() - start_time
        logger.error("Process failed to load.")
        if timeout is not None and elapsed_time >= timeout:
            raise TimeoutError(
                f"Docker environment timed out. {elapsed_time} > {timeout}"
            )
        elif self.process.returncode is not None:
            raise RuntimeError(
                f"Docker environment exited early: Status {self.process.returncode}."
            )
        else:
            raise RuntimeError(
                f"An unknown issue arose. Please check logs at {log_file}"
            )

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, _exctype, excinst, exctb):
        self.close()

        if excinst is None:
            # Regular exit
            return
        if self.error_callback is not None:
            self.error_callback()
        raise excinst


def main():
    parser = argparse.ArgumentParser(
        description="Launch a docker environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-distilroberta-v1",
        help="Model name to launch the docker environment with.",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Revision of the model to use"
    )
    add_docker_args(parser)
    args = parser.parse_args(namespace=Namespace())
    model = Model(args.model_name, args.revision)
    docker_args = args.separate_into((DockerArgs,), warn_unused=False)
    docker_args.validate()

    logger.info(f"Docker script launched for model {model} with {docker_args}")
    logger.info("Press <CTRL>-C to exit.")

    with DockerProcess(model, sys.stdout, docker_args):
        while True:
            time.sleep(1)


if __name__ == "__main__":
    JSON_OUTPUT = False
    main()
