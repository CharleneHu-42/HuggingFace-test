import json
import os
from transformers import set_seed
import logging
from benchmark_utils import BenchmarkPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def load_prompt(model_name, input_tokens, data_dir):
    with open(os.path.join(data_dir, "prompt.json"), "r") as f:
        prompts = json.load(f)

    matched_model = [name for name in MODEL_LIST if name in model_name.lower()]
    if len(matched_model) == 0:
        matched_model = "gpt-j"

    prompt = prompts[matched_model[0]][str(input_tokens)]
    return prompt


def main(args):

    set_seed(args.seed)
    pipeline = BenchmarkPipeline(args)

    prompt = load_prompt(args.model_name, args.input_tokens, args.data_dir)
    logging.info(f"prompt = {prompt}")
    batch_prompt = [prompt] * args.batch_size

    for _ in range(args.warm_up_steps + args.run_steps):
        output = pipeline(batch_prompt)

    logging.info(f"generated text={output}")

    report_dict = pipeline.report()

    return report_dict
