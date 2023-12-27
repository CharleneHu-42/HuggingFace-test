import argparse
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig
from datasets import load_from_disk
import logging

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

def generate(generator, pipe_input, device, dtype, enable):
    time_costs = []
    with torch.autocast(device, dtype, enable), torch.no_grad(), torch.inference_mode():
        for i in range(20):
            pre = time.time()
            output = generator(pipe_input)
            time_costs.append((time.time()-pre)*1000)

    logging.info(f"total time [ms]: {time_costs}")
    logging.info(f"average time [ms] {sum(time_costs[10:]) / 10}")
    logging.info(f"output = {output}")


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    model_id = args.model_id
    device = args.device 
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex

    data = load_from_disk("./datasets/speech_demo")
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.compute_dtype)
    enable = (dtype != torch.float32)

    if args.jit:
        raise ValueError("Automatic-speech-recognition does not support jit trace")

    if "pyannote" not in model_id:
        generator = pipeline(
            "automatic-speech-recognition", model=model_id, device=device, torch_dtype=torch_dtype
        )
        logging.info(data["train"][0])

        if args.torch_compile:
            if args.backend == "ipex":
                import intel_extension_for_pytorch as ipex
            logging.info(f"using torch compile with {args.backend} backend")
            generator.model.generate = torch.compile(
                generator.model.generate, backend=args.backend
            )
            generator.model = torch.compile(generator.model, backend=args.backend)
        elif args.ipex_optimize:
            import intel_extension_for_pytorch as ipex
            logging.info("Use ipex optimize")
            generator.model = ipex.optimize(generator.model, dtype=dtype, inplace=True)

        generate(generator, data["train"][0]["audio"]["array"], device, dtype, enable)
    else:
        from pyannote.audio import Pipeline

        generator = Pipeline.from_pretrained(model_id)
        generator.to(torch.device(device))
        # numpy does not support bf16
        if args.torch_compile:
            logging.info(f"using torch compile with {args.backend} backend")
            if args.backend == "ipex":
                import intel_extension_for_pytorch as ipex
            generator = torch.compile(generator, backend=args.backend)
        elif args.ipex_optimize:
            logging.info("Pyannote do not support ipex optimize")

        generate(generator, "./datasets/speech.wav", device, dtype, enable)
