from transformers import pipeline
import torch
import time
import argparse
import logging
import requests
import PIL.Image
logging.basicConfig(level=logging.INFO)
def str2bool(str):
    return True if str.lower() == 'true' else False

def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16 
    else:
        return torch.float32
    
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
logging.info(f"args = {args}")
model_id = args.model_id

device = args.device 
if device == "xpu":
    import intel_extension_for_pytorch as ipex
torch_dtype = get_torch_dtype(args.model_dtype) 
dtype = get_torch_dtype(args.compute_dtype)

image_to_text = pipeline(
    "image-to-text",
    model=model_id,
    device=device,
    torch_dtype=torch_dtype,
)

if args.torch_compile:
    logging.info(f"Use torch compile with {args.backend} backend")
    if args.backend == "ipex":
        import intel_extension_for_pytorch as ipex
    image_to_text.model.generate = torch.compile(
        image_to_text.model.generate, backend=args.backend
    )
    image_to_text.model = torch.compile(image_to_text.model, backend=args.backend)
elif args.ipex_optimize:
    logging.info("Use ipex optimize")
    import intel_extension_for_pytorch as ipex
    
    image_to_text.model = ipex.optimize(
        image_to_text.model,
        dtype=dtype,
        inplace=True,
    )
else:
    pass


timeout = 3000
image_url = "https://ankur3107.github.io/assets/images/image-captioning-example.png"
image = PIL.Image.open(requests.get(image_url, stream=True, timeout=timeout).raw)
with torch.autocast(device_type=device, dtype=torch.bfloat16 if args.bf16 else torch.float32), \
    torch.inference_mode(), torch.no_grad():
    for i in range(10):
        pre = time.time()
        out = image_to_text(image)
        logging.info(f"Generate time costs {time.time()-pre} seconds")

logging.info(f"output = {out}")