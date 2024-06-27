import argparse
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from benchmark_utils import BenchmarkPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DTYPE_STR_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

MODEL_LIST = [
    "gpt-j",
    "llama",
    "gpt-neox",
    "opt",
    "falcon",
    "bloom",
    "baichuan",
    "t5",
    "gpt2",
]


def load_prompt(model_name, input_tokens):
    project_root = Path(__file__).parents[2]
    with open(project_root / "tests/workloads/datasets/prompt.json", "r") as f:
        prompts = json.load(f)

    matched_model = [name for name in MODEL_LIST if name in model_name.lower()]
    if len(matched_model) == 0:
        matched_model = "gpt-j"

    prompt = prompts[matched_model[0]][str(input_tokens)]
    return prompt


def main(args):

    set_seed(args.seed)
    pipeline = BenchmarkPipeline(args)

    prompt = load_prompt(args.model_name, args.input_tokens)
    logging.info(f"prompt = {prompt}")
    batch_prompt = [prompt] * args.batch_size

    for _ in range(args.warm_up_steps + args.run_steps):
        output = pipeline(batch_prompt)

    logging.info(f"generated text={output}")

    report_dict = pipeline.report()

    return report_dict
