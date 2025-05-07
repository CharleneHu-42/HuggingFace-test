import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
from loguru import logger
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from backend_request_func import (
    SYNC_REQUEST_FUNCS,
    TEIRequestFuncInput,
    sync_get_request,
)
from base_parser import add_base_args
from test_async import BenchmarkMetrics, calculate_metrics, sample_allnli_requests


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
    print("{:<40} {:<10.2f}".format("Throughput (sentences/s):", metrics.throughput))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "throughput": metrics.throughput,
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
        os.makedirs(os.path.join(args.result_dir, "results"), exist_ok=True)
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{batch_size}-{current_dt}.json"  # noqa
        file_name = os.path.join("results", file_name)
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
    max_bs: int,
):
    request_func = SYNC_REQUEST_FUNCS.get(backend)
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
        if bs > max_bs:
            break
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
                    "Throughput (sentences/s)",
                ]
            )
            csv_writer.writerows(
                (
                    bs,
                    "{:.2f}".format(m.mean_latency_ms),
                    "{:.2f}".format(m.median_latency_ms),
                    "{:.2f}".format(m.p99_latency_ms),
                    "{:.2f}".format(m.throughput),
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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

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
        max_bs=args.max_bs,
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
        choices=list(SYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--max-bs",
        type=int,
        default=512,
        help="Maximum batch size to test.",
    )
    add_base_args(parser)

    args = parser.parse_args()
    main(args)
