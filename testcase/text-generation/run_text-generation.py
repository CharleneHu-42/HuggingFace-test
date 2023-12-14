import argparse
import time
import torch
import json
from transformers import pipeline, AutoTokenizer
import logging
logging.basicConfig(level=logging.INFO)


def str2bool(str):
    return True if str.lower() == 'true' else False

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", default='False', type=str2bool)
parser.add_argument("--ipex_optimize", default='False', type=str2bool)
parser.add_argument("--jit", default='False', type=str2bool)
parser.add_argument("--torch_compile", default='False', type=str2bool)
parser.add_argument("--torch_dtype", default="float32", type=str)
parser.add_argument("--backend", default="ipex", type=str)
args = parser.parse_args()
model_id = args.model_id

logging.info(f"args = {args}")

device = "cuda" if torch.cuda.is_available() else "cpu"

with open('./datasets/prompt.json','r') as f:
    prompt = json.load(f)

generation_kwargs = dict(do_sample=False, num_beams=4, use_cache=True)
torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline(
    "text-generation", model=model_id, torch_dtype=torch_dtype, device=device, tokenizer=tokenizer, **generation_kwargs
)



def generate(generator, input_sentence):
    latency = []
    # Warmup
    for i in range(5):
        out = generator(input_sentence, **generation_kwargs)

    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=args.bf16
    ):
        for i in range(10):
            pre = time.time()
            out = generator(input_sentence, **generation_kwargs)
            latency.append((time.time() - pre) * 1000)

    return sum(latency) / len(latency), out


def benchmark(generator, input_sentence):
    input_len = len(tokenizer(input_sentence)["input_ids"])
    logging.info(f"input tokens num is {input_len}")

    # warm up
    generation_kwargs["max_new_tokens"] = 32
    latency, out = generate(generator, input_sentence)

    generation_kwargs["max_new_tokens"] = 1
    first_latency, out = generate(generator, input_sentence)
    out_num = len(tokenizer(out[0]["generated_text"])["input_ids"]) - input_len
    logging.info(f"1st token latency = {first_latency}ms")
    logging.info(f"output token nums = {out_num}")

    generation_kwargs["max_new_tokens"] = 32
    second_latency, out = generate(generator, input_sentence)
    out_num = len(tokenizer(out[0]["generated_text"])["input_ids"]) - input_len
    logging.info(f"2nd+ token latency = {(second_latency - first_latency) / (out_num - 1)}ms")
    logging.info(f"output token nums = {out_num}")
    logging.info(f"output = {out}")


if not args.ipex_optimize and not args.torch_compile:
    benchmark(generator, prompt["gpt-j"]["32"])
    benchmark(generator, prompt["gpt-j"]["512"])
elif args.ipex_optimize:
    logging.info("Use ipex optimization")
    from optimum.intel import inference_mode as ipex_inference_mode

    with ipex_inference_mode(
        generator, dtype=torch.bfloat16 if args.bf16 else torch.float32, verbose=False, jit=args.jit
    ) as ipex_pipe:
        benchmark(ipex_pipe, prompt["gpt-j"]["32"])
        benchmark(ipex_pipe, prompt["gpt-j"]["512"])
elif args.torch_compile:
    logging.info(f"Use torch compile with {args.backend} backend")
    #import intel_extension_for_pytorch

    with torch.inference_mode(), torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16 if args.bf16 else torch.float32):
        generator.model.generate = torch.compile(
            generator.model.generate, backend=args.backend
        )
        # Can only choose one of them to run
        #benchmark(generator, prompt["gpt-j"]["32"])
        benchmark(generator, prompt["gpt-j"]["512"])
