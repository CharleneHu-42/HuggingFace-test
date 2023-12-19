import requests
from PIL import Image
from transformers import pipeline
import torch
import time
import argparse
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
logging.info(f"args = {args}")
model_id = args.model_id

device = "cuda" if torch.cuda.is_available() else "cpu"

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
question = "how many dogs are in the picture?"

torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32


pipe = pipeline("visual-question-answering", model=model_id, torch_dtype=torch_dtype, device=device)
    
if args.torch_compile:
    logging.info(f"Use torch compile with {args.backend} backend")
    import intel_extension_for_pytorch

    pipe.model = torch.compile(pipe.model, backend=args.backend)
    pipe.model.generate = torch.compile(pipe.model.generate, backend=args.backend)
elif args.ipex_optimize:
    logging.info("Use ipex optimize")
    import intel_extension_for_pytorch as ipex
    pipe.model = ipex.optimize(
        pipe.model, dtype=torch.bfloat16 if args.bf16 else torch.float32, inplace=True
    )
else:
    pass

with torch.autocast(device_type=device, dtype=torch.bfloat16 if args.bf16 else torch.float32), \
    torch.inference_mode(), torch.no_grad():
    for i in range(20):
        pre = time.time()
        outputs = pipe(raw_image, question, topk=1)
        logging.info(f"Generate time costs {time.time()-pre} seconds")
logging.info(outputs)