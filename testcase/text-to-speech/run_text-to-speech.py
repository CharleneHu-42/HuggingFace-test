from transformers import pipeline
from datasets import load_dataset, load_from_disk
import soundfile as sf
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--torch_compile", action="store_true")
parser.add_argument("--ipex_optimize", action="store_true")
args = parser.parse_args()
model_id = args.model_id

torch.manual_seed(1024)
torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
synthesiser = pipeline("text-to-speech", model_id, torch_dtype=torch_dtype)

embeddings_dataset = load_from_disk("/home/jiqingfe/datasets/speech_vector")
speaker_embedding = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.
forward_params = {"speaker_embeddings": speaker_embedding} if "t5" in model_id else None

if args.torch_compile:
    print("Use torch compile with ipex backend")
    import intel_extension_for_pytorch
    synthesiser.model.generate = torch.compile(synthesiser.model.generate, backend="ipex")
    synthesiser.model = torch.compile(synthesiser.model, backend="ipex")
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16):
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser("Hello, my dog is cooler than you!", forward_params=forward_params)
            print(f"Generate time costs {time.time()-pre} seconds")
elif args.ipex_optimize:
    print("Use ipex optimize")
    import intel_extension_for_pytorch as ipex
    synthesiser.model = ipex.optimize(synthesiser.model, dtype=torch_dtype, inplace=True)
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16):
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser("Hello, my dog is cooler than you!", forward_params=forward_params)
            print(f"Generate time costs {time.time()-pre} seconds")
else:
    with torch.cpu.amp.autocast(enabled=args.bf16), torch.no_grad(), torch.inference_mode():
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser("Hello, my dog is cooler than you!", forward_params=forward_params)
            print(f"Generate time costs {time.time()-pre} seconds")

