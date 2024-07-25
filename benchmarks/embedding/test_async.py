import argparse
import asyncio
import csv
import itertools
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

from backend_request_func import ASYNC_REQUEST_FUNCS, TEIRequestFuncOutput
from base_parser import add_base_args


@dataclass
class BenchmarkMetrics:
    completed: int
    mean_latency_ms: float
    median_latency_ms: float
    p99_latency_ms: float
    throughput: float


def sample_allnli_requests(
    seed: int,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    min_length: int,
    max_length: int,
) -> List[str]:
    dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")
    anchor_dataset: List[str] = dataset["anchor"]  # type: ignore
    # Shuffle the dataset.
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
        if not (min_length <= token_len <= max_length):
            # Prune too short or too long sequences.
            continue
        filtered_dataset.append(prompt)

    return filtered_dataset


def calculate_metrics(
    outputs: List[TEIRequestFuncOutput],
    dur_s: float,
) -> BenchmarkMetrics:
    completed = 0
    latencies: List[float] = []
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
        throughput=completed / dur_s,
    )

    return metrics


async def benchmark_multi_clients(
    backend: str,
    api_url: str,
    model_id: str,
    client_num: int,
    batch_size: int,
    input_requests: List[str],
    request_rate: float,
    disable_tqdm: bool,
):
    request_func = ASYNC_REQUEST_FUNCS.get(backend)
    if request_func is None:
        raise ValueError(f"Unknown backend: {backend}")
    print(f"Traffic request rate: {request_rate}")
    benchmark_start_time = time.perf_counter()
    tasks = []
    split_requests = [input_requests[i::client_num] for i in range(client_num)]
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    tasks = [
        request_func(api_url, requests, batch_size, request_rate, pbar)
        for requests in split_requests
    ]
    outputs: List[List[TEIRequestFuncOutput]] = await asyncio.gather(*tasks)
    flattened_outputs = list(itertools.chain(*outputs))

    if pbar is not None:
        pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics = calculate_metrics(outputs=flattened_outputs, dur_s=benchmark_duration)

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("batch size:", batch_size))
    print("{:<40} {:<10}".format("Client num:", client_num))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Mean latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median latency (ms):", metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 latency (ms):", metrics.p99_latency_ms))
    print("{:<40} {:<10.2f}".format("Throughput (sentences/s):", metrics.throughput))
    print("=" * 50)
    return metrics


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

    client_nums = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    outputs: List[Tuple[int, BenchmarkMetrics]] = []

    for clients in client_nums:
        m = asyncio.run(
            benchmark_multi_clients(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                client_num=clients,
                batch_size=args.batch_size,
                input_requests=input_requests,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
            )
        )

        if m.completed >= len(input_requests):
            outputs.append((clients, m))
        else:
            logger.warning(
                f"Benchmark failed for client num {clients}: Completed only {m.completed} / {len(input_requests)}"
            )

    # Save CSV
    if args.save_result:
        filename = f"{backend}-async-{args.request_rate}qps-{args.batch_size}bs-{model_id.split('/')[-1]}.csv"
        if args.result_dir:
            filename = os.path.join(args.result_dir, filename)
        with open(filename, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "Number of Clients",
                "Mean Latency (ms)",
                "P50 Latency (ms)",
                "P99 Latency (ms)",
                "Throughput (sentences/s)",
            ])
            csv_writer.writerows(
                (
                    cs,
                    "{:.2f}".format(m.mean_latency_ms),
                    "{:.2f}".format(m.median_latency_ms),
                    "{:.2f}".format(m.p99_latency_ms),
                    "{:.2f}".format(m.throughput),
                )
                for cs, m in outputs
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
        "--batch_size",
        type=int,
        default=1,
        help="batch size of an input request prompt.",
    )
    add_base_args(parser)

    args = parser.parse_args()
    main(args)
