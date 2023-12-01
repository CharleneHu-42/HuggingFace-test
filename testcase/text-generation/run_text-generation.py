import argparse
import time
import torch
import json
from transformers import pipeline, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--ipex_optimize", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--torch_compile", action="store_true")
args = parser.parse_args()
model_id = args.model_id

f = open("/home/jiqingfe/datasets/prompt.json")
prompt = json.load(f)

generation_kwargs = dict(do_sample=False, num_beams=4, use_cache=True)
torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
generator = pipeline("text-generation", model=model_id, torch_dtype=torch_dtype, **generation_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def generate(generator, input_sentence):
    latency = []
    # Warmup
    for i in range(5):
        out = generator(input_sentence, **generation_kwargs)

    for i in range(10):
        pre = time.time()
        out = generator(input_sentence, **generation_kwargs)
        latency.append((time.time()-pre)*1000)

    return sum(latency)/len(latency), out
    
def benchmark(generator, input_sentence):
    input_len = len(tokenizer(input_sentence)['input_ids'])
    print(f"input tokens num is {input_len}")

    generation_kwargs["max_new_tokens"] = 1
    first_latency, out = generate(generator, input_sentence)
    out_num = len(tokenizer(out[0]['generated_text'])['input_ids']) - input_len
    print(f"1st token latency = {first_latency}ms")
    print(f"output token nums = {out_num}")

    generation_kwargs["max_new_tokens"] = 32
    second_latency, out = generate(generator, input_sentence)
    out_num = len(tokenizer(out[0]['generated_text'])['input_ids']) - input_len
    print(f"2nd+ token latency = {(second_latency - first_latency) / (out_num - 1)}ms")
    print(f"output token nums = {out_num}")
    print(f"output = {out}")


if not args.ipex_optimize and not args.torch_compile:
    benchmark(generator, prompt["gpt-j"]["32"])
    benchmark(generator, prompt["gpt-j"]["512"])
elif args.ipex_optimize:
    print("Use ipex optimization")
    from optimum.intel import inference_mode as ipex_inference_mode

    with ipex_inference_mode(generator, dtype=torch_dtype, verbose=False, jit=args.jit) as ipex_pipe:
        benchmark(ipex_pipe, prompt["gpt-j"]["32"])
        benchmark(ipex_pipe, prompt["gpt-j"]["512"])
elif args.torch_compile:
    print("Use torch compile with ipex backend")
    import intel_extension_for_pytorch
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=True):
        generator.model.generate = torch.compile(generator.model.generate, backend="ipex", dynamic=True)
        benchmark(generator, prompt["gpt-j"]["32"])
        benchmark(generator, prompt["gpt-j"]["512"])
