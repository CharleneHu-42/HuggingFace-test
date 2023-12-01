import argparse
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig
from datasets import load_from_disk


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--torch_compile", action="store_true")
parser.add_argument("--ipex_optimize", action="store_true")
args = parser.parse_args()
model_id = args.model_id

data = load_from_disk("/home/jiqingfe/datasets/speech_demo")
torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

if "pyannote" not in model_id:
    generator = pipeline("automatic-speech-recognition", model=model_id, torch_dtype=torch_dtype)
    print(data["train"][0])

    def generate(generator):
        with torch.cpu.amp.autocast(enabled=args.bf16), torch.inference_mode(), torch.no_grad():
            for i in range(10):
                pre = time.time()
                out = generator(data["train"][0]["audio"]["array"])
                print(f"Generate time costs {time.time()-pre} seconds")
                print(f"output = {out}")

    if args.torch_compile:
        print("Use torch compile with ipex backend")
        import intel_extension_for_pytorch
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16):
            generator.model.generate = torch.compile(generator.model.generate, backend="ipex")
            generator.model = torch.compile(generator.model, backend="ipex")
            generate(generator)
    elif args.ipex_optimize:
        print("Use ipex optimize")
        import intel_extension_for_pytorch as ipex
        generator.model = ipex.optimize(generator.model, dtype=torch_dtype, inplace=True)
        generate(generator)
    else:
        generate(generator)

else: 
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")

    # numpy does not support bf16
    if args.torch_compile:
        print("Use torch compile with ipex backend")
        import intel_extension_for_pytorch
        with torch.inference_mode(), torch.no_grad():
            pipeline = torch.compile(pipeline, backend="ipex")
            for i in range(10):
                pre = time.time()
                diarization = pipeline("/home/jiqingfe/datasets/speech.wav")
                print(f"Generate time costs {time.time()-pre} seconds")
    elif args.ipex_optimize:
        print("Do not support ipex optimize")
    else:
        with torch.inference_mode(), torch.no_grad():
            for i in range(10):
                pre = time.time()
                diarization = pipeline("/home/jiqingfe/datasets/speech.wav")
                print(f"Generate time costs {time.time()-pre} seconds")