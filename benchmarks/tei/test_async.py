import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
import itertools
import aiohttp

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, TEIRequestFuncInput,
                                  TEIRequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset

from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class BenchmarkMetrics:
    completed: int
    mean_latency_ms: int
    median_latency_ms: int
    p99_latency_ms: int
    request_throughput: float


def sample_allnli_requests(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    min_length: int,
    max_length: int
) -> List[str]:
    dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")
    anchor_dataset = dataset['anchor']
    # Shuffle the dataset.
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
) -> Tuple[BenchmarkMetrics, List[int]]:
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
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        mean_latency_ms=np.mean(latencies or 0) * 1000,
        median_latency_ms=np.median(latencies or 0) * 1000,
        p99_latency_ms=np.percentile(latencies or 0, 99) * 1000,
        request_throughput=completed / dur_s
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
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    print(f"Traffic request rate: {request_rate}")
    benchmark_start_time = time.perf_counter()
    tasks = []
    split_requests = [input_requests[i::client_num] for i in range(client_num)]
    pbar = None if disable_tqdm else tqdm(total=len(input_requests)//client_num)
    tasks = [request_func(api_url,
                          requests,
                          batch_size,
                          request_rate,
                          pbar) for requests in split_requests]
    outputs: List[List[TEIRequestFuncOutput]] = await asyncio.gather(*tasks)
    flattened_outputs = list(itertools.chain(*outputs))

    if not disable_tqdm:
        pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics = calculate_metrics(
        outputs=flattened_outputs,
        dur_s=benchmark_duration
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("batch size:", batch_size))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10.2f}".format("Mean latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median latency (ms):",
                                    metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 latency (ms):", metrics.p99_latency_ms))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("=" * 50)

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
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
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        min_length=args.min_length,
        max_length= args.max_length
    )

    client_num_list = [2,4,8,16]
    #sweep for bs=1
    for client_num in client_num_list:
        asyncio.run(
            benchmark_multi_clients(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                client_num=client_num,
                batch_size=1,
                input_requests=input_requests,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm)
        )
        time.sleep(5)
    #sweep for bs=4
    for client_num in client_num_list:
        asyncio.run(
            benchmark_multi_clients(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                client_num=client_num,
                batch_size=4,
                input_requests=input_requests,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm)
        )
        time.sleep(5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="tei-async",
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
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
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