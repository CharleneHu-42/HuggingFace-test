from transformers import pipeline
from datasets import load_dataset, load_from_disk
import soundfile as sf
import torch
import time
import argparse
import logging
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
if device == 'xpu':
    import intel_extension_for_pytorch as ipex 

torch.manual_seed(1024)
torch_dtype = get_torch_dtype(args.model_dtype)
dtype = get_torch_dtype(args.compute_dtype)
synthesiser = pipeline("text-to-speech", model_id, device=device, torch_dtype=torch_dtype)

embeddings_dataset = load_from_disk("./datasets/speech_vector")
speaker_embedding = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0).to(device)

# You can replace this embedding with your own as well.
forward_params = {"speaker_embeddings": speaker_embedding} if "t5" in model_id else None

if args.torch_compile:
    logging.info(f"Use torch compile with {args.backend} backend")
    if args.backend == "ipex":
        import intel_extension_for_pytorch as ipex
    synthesiser.model.generate = torch.compile(
        synthesiser.model.generate, backend=args.backend
    )
    synthesiser.model = torch.compile(synthesiser.model, backend=args.backend)

    with torch.autocast(device_type=device, dtype=dtype), torch.no_grad(), torch.inference_mode():
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser(
                "Hello, my dog is cooler than you!", forward_params=forward_params
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")
elif args.ipex_optimize:
    logging.info("Use ipex optimize")
    import intel_extension_for_pytorch as ipex 
    synthesiser.model = ipex.optimize(
        synthesiser.model,
        dtype=dtype,
        inplace=True,
    )
    with torch.autocast(device_type=device, dtype=dtype), torch.no_grad(), torch.inference_mode():
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser(
                "Hello, my dog is cooler than you!", forward_params=forward_params
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")
else:
    with torch.autocast(device_type=device, dtype=dtype), torch.no_grad(), torch.inference_mode():
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser(
                "Hello, my dog is cooler than you!", forward_params=forward_params
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")