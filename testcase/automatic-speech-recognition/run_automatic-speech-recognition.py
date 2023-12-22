import argparse
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig
from datasets import load_from_disk
import logging
logging.basicConfig(level=logging.INFO)
def str2bool(str):
    return True if str.lower() == 'true' else False

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


def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16 
    else:
        return torch.float32

device = args.device 
if device == 'xpu':
    import intel_extension_for_pytorch as ipex

data = load_from_disk("./datasets/speech_demo")
torch_dtype = get_torch_dtype(args.model_dtype)
dtype = get_torch_dtype(args.compute_dtype)

if "pyannote" not in model_id:
    generator = pipeline(
        "automatic-speech-recognition", model=model_id, device=device, torch_dtype=torch_dtype
    )
        
    logging.info(data["train"][0])

    def generate(generator):
        with torch.autocast(device_type=device, dtype=dtype), torch.no_grad(), torch.inference_mode():
            for i in range(10):
                pre = time.time()
                out = generator(data["train"][0]["audio"]["array"])
                logging.info(f"Generate time costs {time.time()-pre} seconds")
                logging.info(f"output = {out}")

    if args.torch_compile:
        logging.info(f"using torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        generator.model.generate = torch.compile(
            generator.model.generate, backend=args.backend
        )
        generator.model = torch.compile(generator.model, backend=args.backend)
        generate(generator)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex
        
        generator.model = ipex.optimize(
            generator.model,
            dtype=dtype,
            inplace=True,
        )
        generate(generator)
    else:
        generate(generator)

else:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")
    pipeline.to(torch.device(device))
    
    # numpy does not support bf16
    if args.torch_compile:
        logging.info(f"using torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        with torch.autocast(device_type=device, dtype=dtype), torch.no_grad(), torch.inference_mode():
            pipeline = torch.compile(pipeline, backend=args.backend)
            for i in range(10):
                pre = time.time()
                diarization = pipeline("./datasets/speech.wav")
                logging.info(f"Generate time costs {time.time()-pre} seconds")
    elif args.ipex_optimize:
        logging.info("Do not support ipex optimize")
    else:
        with torch.autocast(device_type=device, dtype=dtype), torch.no_grad(), torch.inference_mode():
            for i in range(10):
                pre = time.time()
                diarization = pipeline("./datasets/speech.wav")
                logging.info(f"Generate time costs {time.time()-pre} seconds")

