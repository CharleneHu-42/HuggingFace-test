import argparse
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="huggingface model name",
    )
    parser.add_argument(
        "--input_tokens",
        default=32,
        type=int,
        help="choose from [32, 64, 128, 256, 512, 1024]",
    )
    parser.add_argument("--output_tokens", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--tgi_endpoint",
        type=str,
        default="http://127.0.0.1:8080",
        help="the tgi inference endpoint",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "tgi", "opt-intel", "vllm"],
        default="hf",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
        default="fp16",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "xpu"],
        default="xpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm_up_steps", type=int, default=10)
    parser.add_argument("--run_steps", type=int, default=10)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)

    args = parser.parse_args()
    return args


def main(args):
    logging.info(f"args = {args}")

    set_seed(args.seed)
    dtype = DTYPE_STR_MAPPING[args.data_type]
    # load prompt
    project_root = Path(__file__).parents[3]
    with open(project_root / "tests/workloads/datasets/prompt.json", "r") as f:
        prompt = json.load(f)

    matched_model = [name for name in MODEL_LIST if name in args.model.lower()]
    if len(matched_model) == 0:
        matched_model = "gpt-j"
    prompt = prompt[matched_model[0]][str(args.input_tokens)]
    logging.info(f"prompt = {prompt}")
    input_seq = [prompt] * args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_inputs = tokenizer(input_seq, return_tensors="pt")
    input_token_len = len(model_inputs["input_ids"][0])
    assert input_token_len == int(args.input_tokens)

    if args.backend == "tgi":
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=args.tgi_endpoint)

        def send_request(prompt):
            output = client.text_generation(
                prompt=prompt,
                details=True,
                do_sample=False,
                temperature=None,
                max_new_tokens=args.output_tokens,
            )
            return output

        latencies = []
        for _ in range(args.warm_up_steps + args.run_steps):
            pre = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
                futures = [
                    executor.submit(send_request, prompt) for prompt in input_seq
                ]
                outputs = [future.result() for future in futures]
            latencies.append((time.perf_counter() - pre) * 1000)
        latency = sum(latencies[args.warm_up_steps :]) / args.run_steps

        generated_text = outputs[0].generated_text
        generated_token_num = outputs[0].details.generated_tokens

    elif args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=args.output_tokens,
        )

        latencies = []
        for _ in range(args.warm_up_steps + args.run_steps):
            pre = time.perf_counter()
            outputs = llm.generate(prompts=input_seq, sampling_params=sampling_params)
            latencies.append((time.perf_counter() - pre) * 1000)
        latency = sum(latencies[args.warm_up_steps :]) / args.run_steps

        generated_text = outputs[0].outputs[0].text
        generated_token_num = len(outputs[0].outputs[0].token_ids)

    else:
        if args.backend == "opt-intel":
            from optimum.intel import IPEXModelForCausalLM

            model = IPEXModelForCausalLM.from_pretrained(
                args.model, torch_dtype=dtype, export=True
            )
        elif args.backend == "hf":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            model.to(args.device)
        else:
            raise ValueError(
                f"only hf, tgi, opt-intel, vllm are accepted for backend, but {args.backend} was given."
            )

        generation_kwargs = dict(
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            max_new_tokens=args.output_tokens,
        )

        generator = pipeline(
            task="text-generation", model=model, device=args.device, tokenizer=tokenizer
        )

        latencies = []
        for _ in range(args.warm_up_steps + args.run_steps):
            pre = time.perf_counter()
            outputs = generator(
                input_seq,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
            latencies.append((time.perf_counter() - pre) * 1000)

        latency = sum(latencies[args.warm_up_steps :]) / args.run_steps

        generated_text = outputs[0][0]["generated_text"]
        generated_token_num = (
            tokenizer(prompt + generated_text, return_tensors="pt")["input_ids"].size()[
                1
            ]
            - input_token_len
        )

    assert generated_token_num == args.output_tokens
    logging.info(
        f"token latency = {round(latency/(generated_token_num * args.batch_size), 2)} ms"
    )
    logging.info(f"output={generated_text}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
