import requests
from PIL import Image
from transformers import pipeline
import torch
import time
import argparse
import logging
import sys 
sys.setrecursionlimit(10000000)

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

def generate(generator, device, dtype, raw_image, question, enable):
    time_costs = []
    with torch.autocast(device, dtype, enable), torch.no_grad(), torch.inference_mode():
        for i in range(20):
            pre = time.time()
            output = generator(raw_image, question, topk=1)
            time_costs.append((time.time()-pre)*1000)

    logging.info(f"total time [ms]: {time_costs}")
    logging.info(f"average time [ms] {sum(time_costs[10:]) / 10}")
    logging.info(f"output = {output}")

def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16 
    else:
        return torch.float32


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    model_id = args.model_id

    device = args.device 
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex 

    image_path = "./datasets/vqa_cats.jpg"
    raw_image = Image.open(image_path).convert("RGB")
    question = "how many dogs are in the picture?"

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.compute_dtype)
    enable = (dtype != torch.float32)

    pipe = pipeline("visual-question-answering", model=model_id, torch_dtype=torch_dtype, device=device)

    if args.jit:
        raise ValueError("Visual-question-answering does not support jit trace")
        
    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        pipe.model = torch.compile(pipe.model, backend=args.backend)
        pipe.model.generate = torch.compile(pipe.model.generate, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex 
        pipe.model = ipex.optimize(pipe.model, dtype=dtype, inplace=True)

    generate(pipe, device, dtype, raw_image, question, enable)
