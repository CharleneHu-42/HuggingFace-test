import argparse
import time
import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--out_num", default=32, type=int, required=False)
parser.add_argument("--input_length", default=32, type=str, required=False)
args = parser.parse_args()
model_id = args.model_id

f = open("./prompt.json")
prompt = json.load(f)
input_sentence = prompt["gpt-j"][args.input_length]

generation_kwargs = dict(max_new_tokens=args.out_num, do_sample=False, num_beams=4, use_cache=True)
generator = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, **generation_kwargs)
input_len = len(tokenizer(input_sentence)['input_ids'])
print(f"input tokens num is {input_len}")

def generate(generator):
    for i in range(10):
        pre = time.time()
        out = generator(input_sentence, **generation_kwargs)
        print(f"Generate time costs {time.time()-pre} seconds")
        print(f"output = {out}")
        print(f"output token nums = {len(tokenizer(out[0]['generated_text'])['input_ids']) - input_len}")

generate(generator)