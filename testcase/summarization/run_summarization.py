import argparse
import time
import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig
import logging
logging.basicConfig(level=logging.INFO)


def str2bool(str):
    return True if str.lower() == 'true' else False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None, type=str, required=True)
    parser.add_argument("--compute_dtype", default="float32", type=str)
    parser.add_argument("--ipex_optimize", default='False', type=str2bool)
    parser.add_argument("--jit", default='False', type=str2bool)
    parser.add_argument("--torch_compile", default='False', type=str2bool)
    parser.add_argument("--model_dtype", default="float32", type=str)
    parser.add_argument("--backend", default="inductor", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    return args

def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16 
    else:
        return torch.float32

def generate(generator, input_sentence, device, dtype, enable):
    latency = []
    with torch.autocast(device, dtype, enable), torch.no_grad(), torch.inference_mode():
        for i in range(20):
            pre = time.time()
            output = generator(input_sentence, **generation_kwargs)
            latency.append((time.time() - pre) * 1000)

    return sum(latency[10:]) / 10, output

def benchmark(generator, input_sentence, device, dtype, enable):
    input_len = len(tokenizer(input_sentence)["input_ids"])
    logging.info(f"input tokens length is {input_len}")

    generation_kwargs["max_new_tokens"] = 1
    first_latency, out = generate(generator, input_sentence, device, dtype, enable)
    out_num = len(tokenizer(out[0]["summary_text"])["input_ids"]) - 2
    logging.info(f"1st token latency = {first_latency} ms")
    logging.info(f"output token nums = {out_num}")

    generation_kwargs["max_new_tokens"] = 32
    second_latency, out = generate(generator, input_sentence, device, dtype, enable)
    out_num = len(tokenizer(out[0]["summary_text"])["input_ids"]) - 2
    logging.info(f"2nd+ token latency = {(second_latency - first_latency) / (out_num - 1)} ms")
    logging.info(f"output token nums = {out_num}")
    logging.info(f"output = {out}")


if __name__ == "__main__":
    args = get_args()
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logging.info(f"args = {args}")
    with open('./datasets/prompt.json','r') as f:
        prompt = json.load(f)

    device = args.device 
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
        
    generation_kwargs = dict(do_sample=False, num_beams=4, use_cache=True)
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.compute_dtype)
    enable = (dtype != torch.float32)

    generator = pipeline(
        "summarization",
        model=model_id,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
        **generation_kwargs,
    )

    if args.jit:
        raise ValueError("Summarization does not support jit trace")

    if args.torch_compile:
        if device == "cpu":
            raise ValueError("Torch compile for summarization in CPU is not work, please change the script if you want to reproduce the bug")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        logging.info(f"Use torch compile with {args.backend} backend")
        generator.model.generate = torch.compile(
            generator.model.generate, backend=args.backend
        )
        # Can only choose one of them to run
        benchmark(generator, prompt["gpt-j"]["32"], device, dtype, enable)
        # benchmark(generator, prompt["gpt-j"]["512"], device, dtype, enable)
    elif args.ipex_optimize:
        import intel_extension_for_pytorch as ipex
        logging.info("Use ipex optimize")
        generator.model = ipex.optimize(
            generator.model,
            dtype=dtype,
            inplace=True,
        )
        benchmark(generator, prompt["gpt-j"]["32"], device, dtype, enable)
        benchmark(generator, prompt["gpt-j"]["512"], device, dtype, enable)
    else:
        benchmark(generator, prompt["gpt-j"]["32"], device, dtype, enable)
        benchmark(generator, prompt["gpt-j"]["512"], device, dtype, enable)
