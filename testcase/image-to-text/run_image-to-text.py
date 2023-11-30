
from transformers import pipeline
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", action="store_true")
args = parser.parse_args()
model_id = args.model_id

torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", torch_dtype=torch_dtype)

for i in range(10):
    pre = time.time()
    out = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")
    print(f"Generate time costs {time.time()-pre} seconds")
print(f"output = {out}")

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]
