import os
import sys
import time
import torch
import json
import logging

rank = int(os.environ.get("RANK", 0))
if rank == 0:
    logging.basicConfig(level=logging.INFO)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import ContextManagers

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    get_args,
    get_torch_dtype,
    synchronize_device,
    get_batched_prompts,
)

inference_context = [torch.no_grad()]
MODEL_LIST = ["gpt-j", "llama", "gpt-neox", "opt", "falcon", "bloom", "t5", "gpt2"]


def generate(model, inputs, warm_up_steps, run_steps):
    latency = []
    inputs["generation_config"] = generation_config
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            synchronize_device(model.device.type)
            pre = time.time()
            output = model.generate(**inputs)
            synchronize_device(model.device.type)
            latency.append((time.time() - pre) * 1000)

    return sum(latency[warm_up_steps:]) / run_steps, output


def benchmark(
    model, warm_up_steps, run_steps, inputs, output_tokens, batch_size
):
    input_len = inputs["input_ids"].shape[-1]
    logging.info(f"input tokens length is {input_len}")

    _, _ = generate(model, inputs, 1, 1)
    generation_config.max_new_tokens = 1
    generation_config.min_new_tokens = 1

    first_latency, out = generate(
        model, inputs, warm_up_steps, run_steps
    )

    logging.info(f"1st token latency = {first_latency} ms")
    logging.info(f"output token nums = {batch_size}")

    generation_config.max_new_tokens = output_tokens
    generation_config.min_new_tokens = output_tokens
    latency, out = generate(model, inputs, warm_up_steps, run_steps)
    out_num = output_tokens
    logging.info(
        f"2nd+ token latency = {(latency - first_latency) / (out_num - 1)} ms"
    )
    logging.info(f"output token nums = {out_num*batch_size}")
    logging.info(f"output = {tokenizer.batch_decode(out, skip_special_tokens=True)}")
    logging.info(
        f"total average time [ms] {latency}"
    )


if __name__ == "__main__":
    args = get_args()
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    tp_plan = args.tp_plan
    device = args.device
    logging.info(f"args = {args}")

    with open("./datasets/prompt.json", "r") as f:
        prompt = json.load(f)

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    model_kwargs = dict(torch_dtype=torch_dtype)
    if tp_plan:
        tp_backend = None
        device_id = None
        if device == "cpu":
            tp_backend = "gloo"
        elif device == "xpu":
            tp_backend = "xccl"
            device_id = torch.device(f"xpu:{rank}")
        elif device == "cuda":
            tp_backend = "nccl"
            device_id = torch.device(f"cuda:{rank}")
        torch.distributed.init_process_group(tp_backend, device_id=device_id)
        model_kwargs["tp_plan"] = tp_plan

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    generation_config = model.generation_config
    generation_config.do_sample = args.do_sample
    generation_config.use_cache = True
    generation_config.temperature = 1.0
    generation_config.num_beams = args.num_beams
    generation_config.max_new_tokens = args.output_tokens
    generation_config.min_new_tokens = args.output_tokens
    generation_config.top_p = 1.0
    # TP do not support static cache for now.
    # generation_config.cache_implementation="static"

    if "falcon" in model_id:
        # For the correct shape of static cache
        if not getattr(model.config, "new_decoder_architecture", False):
            model.config.num_key_value_heads = 1

    model_type = [name for name in MODEL_LIST if name in model_id.lower()]
    if len(model_type) == 0:
        model_type = ["gpt-j"]

    prompt = prompt[model_type[0]][str(args.input_tokens)]
    input_seq = get_batched_prompts(prompt, args.batch_size)
    inputs = tokenizer(input_seq, padding=True, return_tensors="pt").to(model.device)

    if args.optimum_intel:
        from optimum.intel import IPEXModelForCausalLM
        model = IPEXModelForCausalLM(model, export=True, torch_dtype=torch_dtype)
    elif args.ipex_optimize_transformers:
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize_transformers(model, dtype=torch_dtype, device=device)
    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        # pipeline warmup
        _, _ = generate(model, inputs, 1, 1)
        model.forward = torch.compile(model.forward, backend=args.backend)
        # compile warmup
        _, _ = generate(model, inputs, 1, 1)

    benchmark(
            model,
            warm_up_steps,
            run_steps,
            inputs,
            output_tokens=args.output_tokens,
            batch_size=args.batch_size,
        )
