import mmlu
import simple_bench
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        choices=["mmlu", "simple_bench"],
        default="simple_bench",
        help="benchmark task name",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["trt-llm", "optimum-intel", "hf", "ipex", "vllm", "tgi"],
        default="hf",
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="huggingface model name"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
        default="fp16",
    )
    parser.add_argument(
        "--input_tokens",
        default=32,
        type=int,
        help="choose from [32, 64, 128, 256, 512, 1024]",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "xpu"],
        default="xpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm_up_steps", type=int, default=10)
    parser.add_argument(
        "--engine_dir",
        type=str,
        default=None,
        help="tensorrt engine, only valid for trt-llm backend",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="directory to save the benchmark result",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument(
        "--tgi_endpoint",
        type=str,
        default="http://127.0.0.1:8888",
        help="tgi inference endpoint, only valid for tgi backend",
    )
    group = parser.add_argument_group(title="mmlu-only args")
    group.add_argument(
        "--ntrain",
        type=int,
        default=5,
        help="number of examples to be included in the prompt",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu",
        help=(
            "Path to the data directory. If not available, "
            "download https://people.eecs.berkeley.edu/~hendrycks/data.tar."
            "Only applicable for MMLU task"
        ),
    )
    group = parser.add_argument_group(title="simple_bench-only args")
    parser.add_argument("--run_steps", type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    task_name = args.task_name
    task_map = dict(
        mmlu=mmlu.main,
        simple_bench=simple_bench.main,
    )
    if task_name in task_map.keys():
        task_fn = task_map.get(task_name)
        report_dict = task_fn(args)
    else:
        raise ValueError(
            f"{task_name} is not a valid task name. Choose from {list(task_map.keys())}"
        )

    kv_pairs = [f"{k}={v}" for k, v in report_dict.items()]
    line = "\n".join(kv_pairs)
    logging.info(line)


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"args = {args}")
    main(args)
