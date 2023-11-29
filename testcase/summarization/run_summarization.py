import argparse
import time
import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", action="store_true")
args = parser.parse_args()
model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)

f = open("./prompt.json")
prompt = json.load(f)

generation_kwargs = dict(do_sample=False, num_beams=4, use_cache=True)
torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
generator = pipeline("summarization", model=model_id, tokenizer=tokenizer, **generation_kwargs, torch_dtype=torch_dtype)


# max_length=out_num+1, min_length=out_num, 


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
    out_num = len(tokenizer(out[0]['summary_text'])['input_ids']) - 2
    print(f"1st token latency = {first_latency}ms")
    print(f"output token nums = {out_num}")

    generation_kwargs["max_new_tokens"] = 32
    second_latency, out = generate(generator, input_sentence)
    out_num = len(tokenizer(out[0]['summary_text'])['input_ids']) - 2
    print(f"2nd+ token latency = {(second_latency - first_latency) / (out_num - 1)}ms")
    print(f"output token nums = {out_num}")
    print(f"output = {out}")


benchmark(generator, prompt["gpt-j"]["32"])
benchmark(generator, prompt["gpt-j"]["512"])
