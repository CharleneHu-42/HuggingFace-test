import requests
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    ViltProcessor,
    ViltForQuestionAnswering,
)
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

args = parser.parse_args()
logging.info(f"args = {args}")
model_id = args.model_id

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
question = "how many dogs are in the picture?"

torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32

if "vilt" in model_id:
    processor = ViltProcessor.from_pretrained(model_id)
    model = ViltForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch_dtype)
    inputs = processor(raw_image, question, return_tensors="pt")

    if args.torch_compile:
        logging.info("Use torch compile with ipex backend")
        import intel_extension_for_pytorch

        model = torch.compile(model, backend="ipex")
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
            for i in range(20):
                pre = time.time()
                outputs = model(**inputs)
                logging.info(f"Generate time costs {time.time()-pre} seconds")
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        logging.info("Predicted answer:", model.config.id2label[idx])
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        model = ipex.optimize(
            model, dtype=torch.bfloat16 if args.bf16 else torch.float32, inplace=True
        )
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
            for i in range(20):
                pre = time.time()
                outputs = model(**inputs)
                logging.info(f"Generate time costs {time.time()-pre} seconds")
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        logging.info("Predicted answer:", model.config.id2label[idx])
    else:
        with torch.cpu.amp.autocast(
            enabled=args.bf16
        ), torch.no_grad(), torch.inference_mode():
            for i in range(10):
                pre = time.time()
                outputs = model(**inputs)
                logging.info(f"Generate time costs {time.time()-pre} seconds")
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        logging.info("Predicted answer:", model.config.id2label[idx])
else:
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch_dtype)

    inputs = processor(raw_image, question, return_tensors="pt")

    if args.torch_compile:
        logging.info("Use torch compile with ipex backend")
        import intel_extension_for_pytorch

        model.generate = torch.compile(model.generate, backend="ipex")
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
            for i in range(20):
                pre = time.time()
                out = model.generate(**inputs)
                logging.info(f"Generate time costs {time.time()-pre} seconds")
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        model = ipex.optimize(
            model, dtype=torch.bfloat16 if args.bf16 else torch.float32, inplace=True
        )
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
            for i in range(20):
                pre = time.time()
                out = model.generate(**inputs)
                logging.info(f"Generate time costs {time.time()-pre} seconds")
    else:
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
            for i in range(10):
                pre = time.time()
                out = model.generate(**inputs)
                logging.info(f"Generate time costs {time.time()-pre} seconds")

    logging.info(processor.decode(out[0], skip_special_tokens=True))
