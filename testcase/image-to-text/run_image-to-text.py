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

torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
image_to_text = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    device=device,
    torch_dtype=torch_dtype,
)

if args.torch_compile:
    logging.info(f"Use torch compile with {args.backend} backend")
    import intel_extension_for_pytorch

    image_to_text.model.generate = torch.compile(
        image_to_text.model.generate, backend=args.backend
    )
    image_to_text.model = torch.compile(image_to_text.model, backend=args.backend)
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=args.bf16
    ):
        for i in range(10):
            pre = time.time()
            out = image_to_text(
                "https://ankur3107.github.io/assets/images/image-captioning-example.png"
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")
        logging.info(f"output = {out}")
elif args.ipex_optimize:
    logging.info("Use ipex optimize")
    import intel_extension_for_pytorch as ipex

    image_to_text.model = ipex.optimize(
        image_to_text.model,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        inplace=True,
    )
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=args.bf16
    ):
        for i in range(10):
            pre = time.time()
            out = image_to_text(
                "https://ankur3107.github.io/assets/images/image-captioning-example.png"
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")
        logging.info(f"output = {out}")
else:
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=args.bf16
    ):
        for i in range(10):
            pre = time.time()
            out = image_to_text(
                "https://ankur3107.github.io/assets/images/image-captioning-example.png"
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")
        logging.info(f"output = {out}")

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]
