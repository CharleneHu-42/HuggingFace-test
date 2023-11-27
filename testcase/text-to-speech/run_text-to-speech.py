from transformers import pipeline
from datasets import load_dataset, load_from_disk
import soundfile as sf
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
args = parser.parse_args()
model_id = args.model_id

torch.manual_seed(1024)
synthesiser = pipeline("text-to-speech", model_id)

embeddings_dataset = load_from_disk("./speech_vector")
speaker_embedding = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

forward_params = {"speaker_embeddings": speaker_embedding} if "t5" in model_id else None

with torch.cpu.amp.autocast(enabled=True), torch.no_grad():
    for i in range(10):
        torch.manual_seed(1024)
        pre = time.time()
        speech = synthesiser("Hello, my dog is cooler than you!", forward_params=forward_params)
        print(f"Generate time costs {time.time()-pre} seconds")

