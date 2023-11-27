import argparse
import time
import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--out_num", default=32, type=int, required=False)
parser.add_argument("--input_length", default=32, type=str, required=False)
args = parser.parse_args()
model_id = args.model_id
out_num = args.out_num

tokenizer = AutoTokenizer.from_pretrained(model_id)

f = open("./prompt.json")
prompt = json.load(f)
input_sentence = prompt["gpt-j"][args.input_length]

generation_kwargs = dict(max_length=out_num+1, min_length=out_num, do_sample=False, num_beams=4, use_cache=True)
generator = pipeline("summarization", model=model_id, tokenizer=tokenizer, **generation_kwargs, torch_dtype=torch.bfloat16)
input_len = len(tokenizer(input_sentence)['input_ids'])
# 2 is the prefix, not real tokens.
print(f"input tokens num is {input_len - 2}")

def generate(generator):
    for i in range(10):
        pre = time.time()
        out = generator(input_sentence, **generation_kwargs)
        print(f"Generate time costs {time.time()-pre} seconds")
        print(f"output = {out}")
        print(f"output token nums = {len(tokenizer(out[0]['summary_text'])['input_ids']) - 2}")

generate(generator)