import time
import torch
import json
from transformers import pipeline, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)

import sys
import os

sys.path.append(os.path.dirname(__file__) + "/..")

from common import get_args, get_torch_dtype, wrap_forward_for_benchmark

WARMUP = 10
RUN = 10


def generate(generator, input_sentence, device, dtype, enable):
    latency = []
    forward_latency = []
    with torch.autocast(device, dtype, enable), torch.no_grad(), torch.inference_mode():
        for i in range(WARMUP + RUN):
            generator.forward_time = 0
            pre = time.time()
            output = generator(input_sentence, **generation_kwargs)
            latency.append((time.time() - pre) * 1000)
            forward_latency.append(generator.forward_time * 1000)

    return sum(latency[WARMUP:]) / RUN, output, sum(forward_latency[WARMUP:]) / RUN


def benchmark(generator, input_sentence, device, dtype, enable):
    input_len = len(tokenizer(input_sentence)["input_ids"])
    logging.info(f"input tokens length is {input_len}")

    generation_kwargs["max_new_tokens"] = 1

    first_latency, out, first_forward_latency = generate(
        generator, input_sentence, device, dtype, enable
    )
    out_num = len(tokenizer(out[0]["generated_text"])["input_ids"]) - input_len
    logging.info(
        f"1st token latency = {first_latency} ms, pipeline_forward_time = {first_forward_latency} ms({first_forward_latency/first_latency})"
    )
    logging.info(f"output token nums = {out_num}")

    generation_kwargs["max_new_tokens"] = 32
    latency, out, forward_latency = generate(
        generator, input_sentence, device, dtype, enable
    )
    out_num = len(tokenizer(out[0]["generated_text"])["input_ids"]) - input_len
    logging.info(f"2nd+ token latency = {(latency - first_latency) / (out_num - 1)} ms")
    logging.info(f"output token nums = {out_num}")
    logging.info(f"output = {out}")
    logging.info(
        f"pipeline average time = {latency} ms, pipeline_forward_time = {forward_latency} ms({forward_latency/latency})"
    )


if __name__ == "__main__":
    args = get_args()
    model_id = args.model_id

    logging.info(f"args = {args}")

    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    with open("./datasets/prompt.json", "r") as f:
        prompt = json.load(f)

    generation_kwargs = dict(do_sample=False, num_beams=4, use_cache=True)
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.compute_dtype)
    enable = dtype != torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generator = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        tokenizer=tokenizer,
        **generation_kwargs,
    )
    wrap_forward_for_benchmark(generator)

    if not args.ipex_optimize and not args.torch_compile:
        benchmark(generator, prompt["gpt-j"]["512"], device, dtype, enable)
    elif args.ipex_optimize:
        from optimum.intel import inference_mode as ipex_inference_mode

        logging.info("Use ipex optimization")
        with ipex_inference_mode(
            generator, dtype=dtype, verbose=False, jit=args.jit
        ) as ipex_pipe:
            benchmark(ipex_pipe, prompt["gpt-j"]["512"], device, dtype, enable)
    elif args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        generator.model.generate = torch.compile(
            generator.model.generate, backend=args.backend
        )
        benchmark(generator, prompt["gpt-j"]["512"], device, dtype, enable)
