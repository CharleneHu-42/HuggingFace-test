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

torch.manual_seed(1024)
torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
synthesiser = pipeline("text-to-speech", model_id, torch_dtype=torch_dtype)

embeddings_dataset = load_from_disk("./datasets/speech_vector")
speaker_embedding = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.
forward_params = {"speaker_embeddings": speaker_embedding} if "t5" in model_id else None

if args.torch_compile:
    logging.info("Use torch compile with ipex backend")
    import intel_extension_for_pytorch

    synthesiser.model.generate = torch.compile(
        synthesiser.model.generate, backend="ipex"
    )
    synthesiser.model = torch.compile(synthesiser.model, backend="ipex")
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=args.bf16
    ):
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
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        inplace=True,
    )
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=args.bf16
    ):
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser(
                "Hello, my dog is cooler than you!", forward_params=forward_params
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")
else:
    with torch.cpu.amp.autocast(
        enabled=args.bf16
    ), torch.no_grad(), torch.inference_mode():
        for i in range(10):
            torch.manual_seed(1024)
            pre = time.time()
            speech = synthesiser(
                "Hello, my dog is cooler than you!", forward_params=forward_params
            )
            logging.info(f"Generate time costs {time.time()-pre} seconds")
