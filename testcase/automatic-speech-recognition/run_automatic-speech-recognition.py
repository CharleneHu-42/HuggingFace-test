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
parser.add_argument("--bf16", default='False', type=str2bool)
parser.add_argument("--ipex_optimize", default='False', type=str2bool)
parser.add_argument("--jit", default='False', type=str2bool)
parser.add_argument("--torch_compile", default='False', type=str2bool)
parser.add_argument("--torch_dtype", default="float32", type=str)
parser.add_argument("--backend", default="ipex", type=str)
args = parser.parse_args()
logging.info(f"args = {args}")
model_id = args.model_id

data = load_from_disk("./datasets/speech_demo")
torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32

if "pyannote" not in model_id:
    generator = pipeline(
        "automatic-speech-recognition", model=model_id, torch_dtype=torch_dtype
    )
    logging.info(data["train"][0])

    def generate(generator):
        with torch.cpu.amp.autocast(
            enabled=args.bf16
        ), torch.inference_mode(), torch.no_grad():
            for i in range(10):
                pre = time.time()
                out = generator(data["train"][0]["audio"]["array"])
                logging.info(f"Generate time costs {time.time()-pre} seconds")
                logging.info(f"output = {out}")

    if args.torch_compile:
        logging.info(f"using torch compile with {args.backend} backend")
        import intel_extension_for_pytorch

        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
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
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
            inplace=True,
        )
        generate(generator)
    else:
        generate(generator)

else:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")

    # numpy does not support bf16
    if args.torch_compile:
        logging.info(f"using torch compile with {args.backend} backend")
        import intel_extension_for_pytorch

        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
            pipeline = torch.compile(pipeline, backend=args.backend)
            for i in range(10):
                pre = time.time()
                diarization = pipeline("./datasets/speech.wav")
                logging.info(f"Generate time costs {time.time()-pre} seconds")
    elif args.ipex_optimize:
        logging.info("Do not support ipex optimize")
    else:
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=args.bf16
        ):
            for i in range(10):
                pre = time.time()
                diarization = pipeline("./datasets/speech.wav")
                logging.info(f"Generate time costs {time.time()-pre} seconds")
