import argparse
import csv
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    TEIRequestFuncInput,
    TEIRequestFuncOutput,
    sync_get_request,
)


@dataclass
class BenchmarkMetrics:
    completed: int
    mean_latency_ms: float
    median_latency_ms: float
    p99_latency_ms: float
    request_throughput: float


def sample_allnli_requests(
    seed: int,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    min_length: int,
    max_length: int,
) -> List[str]:
    dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")
    anchor_dataset: List[str] = dataset["anchor"]  # type: ignore
    # Shuffle the dataset
    random.seed(seed)
    random.shuffle(anchor_dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[str] = []
    for i in range(len(anchor_dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = anchor_dataset[i]
        prompt_token_ids = tokenizer(prompt).input_ids
        token_len = len(prompt_token_ids)
        if token_len < min_length:
            # Prune too short sequences.
            continue
        if token_len > max_length:
            # Prune too long sequences.
            continue
        filtered_dataset.append(prompt)

    return filtered_dataset


def calculate_metrics(
    outputs: List[TEIRequestFuncOutput],
    dur_s: float,
) -> BenchmarkMetrics:
    completed = 0
    latencies = []
    for i in range(len(outputs)):
        output = outputs[i]
        if output.success:
            completed += output.batch_size
            latencies.append(output.latency)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        mean_latency_ms=float(np.mean(latencies or 0) * 1000),
        median_latency_ms=float(np.median(latencies or 0) * 1000),
        p99_latency_ms=float(np.percentile(latencies or 0, 99) * 1000),
        request_throughput=completed / dur_s,
    )

    return metrics


def benchmark_with_bs(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer_id: str,
    input_requests: List[str],
    request_func,
    batch_size: int,
    request_rate: float,
    disable_tqdm: bool,
    save_result: bool,
) -> BenchmarkMetrics:
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    benchmark_start_time = time.perf_counter()
    outputs = []
    for prompt in sync_get_request(input_requests, request_rate, batch_size):
        request_func_input = TEIRequestFuncInput(
            model=model_id, prompt=prompt, api_url=api_url
        )
        output = request_func(request_func_input=request_func_input, pbar=pbar)
        outputs.append(output)
    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics = calculate_metrics(outputs=outputs, dur_s=benchmark_duration)

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("batch size:", batch_size))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Mean latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median latency (ms):", metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 latency (ms):", metrics.p99_latency_ms))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "request_throughput": metrics.request_throughput,
        "errors": [output.error for output in outputs],
    }
    if save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )

        # Merge with benchmark result
        result_json = {**result_json, **result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{batch_size}-{current_dt}.json"  # noqa
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)
    return metrics


def benchmark_single_client(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer_id: str,
    input_requests: List[str],
    request_rate: float,
    disable_tqdm: bool,
    save_results: bool,
):
    request_func = ASYNC_REQUEST_FUNCS.get(backend)
    if request_func is None:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt = input_requests[0]
    test_input = TEIRequestFuncInput(
        model=model_id, prompt=test_prompt, api_url=api_url
    )
    test_output = request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")
    print(f"Traffic request rate: {request_rate}")
    bs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    outputs: List[Tuple[int, BenchmarkMetrics]] = []
    for bs in bs_list:
        m = benchmark_with_bs(
            backend,
            api_url,
            model_id,
            tokenizer_id,
            input_requests,
            request_func,
            bs,
            request_rate,
            disable_tqdm,
            save_results,
        )
        if m.completed == len(input_requests):
            outputs.append((bs, m))
        else:
            logger.warning(
                f"Benchmark failed for batch size {bs}: Completed only {m.completed} / {len(input_requests)}"
            )

    # Save CSV
    if args.save_result:
        filename = f"{backend}-{args.request_rate}qps-{model_id.split('/')[-1]}.csv"
        if args.result_dir:
            filename = os.path.join(args.result_dir, filename)
        with open(filename, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [
                    "Batch Size",
                    "Mean Latency (ms)",
                    "P50 Latency (ms)",
                    "P99 Latency (ms)",
                    "Throughput (req/s)",
                ]
            )
            csv_writer.writerows(
                (
                    bs,
                    m.mean_latency_ms,
                    m.median_latency_ms,
                    m.p99_latency_ms,
                    m.request_throughput,
                )
                for bs, m in outputs
            )


def main(args: argparse.Namespace):
    print(args)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id)

    input_requests = sample_allnli_requests(
        seed=args.seed,
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    benchmark_single_client(
        backend=backend,
        api_url=api_url,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        input_requests=input_requests,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        save_results=args.save_result,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tei",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=4,
        help="min length of input prompt's token id.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="max length of input prompt's token id.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )

    args = parser.parse_args()
    main(args)
