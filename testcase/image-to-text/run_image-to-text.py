
from transformers import pipeline
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--torch_compile", action="store_true")
parser.add_argument("--ipex_optimize", action="store_true")
args = parser.parse_args()
model_id = args.model_id

torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", torch_dtype=torch_dtype)

if args.torch_compile:
    print("Use torch compile with ipex backend")
    import intel_extension_for_pytorch
    image_to_text.model.generate = torch.compile(image_to_text.model.generate, backend="ipex")
    image_to_text.model = torch.compile(image_to_text.model, backend="ipex")
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16):
        for i in range(10):
            pre = time.time()
            out = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")
            print(f"Generate time costs {time.time()-pre} seconds")
        print(f"output = {out}")
elif args.ipex_optimize:
    print("Use ipex optimize")
    import intel_extension_for_pytorch as ipex
    image_to_text.model = ipex.optimize(image_to_text.model, dtype=torch_dtype, inplace=True)
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16):
        for i in range(10):
            pre = time.time()
            out = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")
            print(f"Generate time costs {time.time()-pre} seconds")
        print(f"output = {out}")
else:
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16):
        for i in range(10):
            pre = time.time()
            out = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")
            print(f"Generate time costs {time.time()-pre} seconds")
        print(f"output = {out}")

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]
