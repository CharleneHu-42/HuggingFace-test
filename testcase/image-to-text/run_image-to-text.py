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

def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16 
    else:
        return torch.float32

def generate(generator, image, device, dtype, enable):
    time_costs = []
    with torch.autocast(device, dtype, enable), torch.inference_mode(), torch.no_grad():
        for i in range(20):
            pre = time.time()
            output = generator(image)
            time_costs.append((time.time()-pre)*1000)

    logging.info(f"total time [ms]: {time_costs}")
    logging.info(f"average time [ms] {sum(time_costs[10:]) / 10}")
    logging.info(f"output = {output}")


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    model_id = args.model_id

    device = args.device 
    if device == "xpu":
        import intel_extension_for_pytorch as ipex
    torch_dtype = get_torch_dtype(args.model_dtype) 
    dtype = get_torch_dtype(args.compute_dtype)
    enable = (dtype != torch.float32)

    image_to_text = pipeline(
        "image-to-text",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )

    if args.jit:
        raise ValueError("Image-to-text does not support jit trace")

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

    image_url = "https://ankur3107.github.io/assets/images/image-captioning-example.png"
    image = PIL.Image.open(requests.get(image_url, stream=True, timeout=3000).raw)

    generate(image_to_text, image, device, dtype, enable)