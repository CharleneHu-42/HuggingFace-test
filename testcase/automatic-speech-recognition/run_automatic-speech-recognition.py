import argparse
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig
from datasets import load_from_disk


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
args = parser.parse_args()
model_id = args.model_id

data = load_from_disk("./speech_demo")

if "pyannote" not in model_id:
    generator = pipeline("automatic-speech-recognition", model=model_id)
    print(data["train"][0])

    def generate(generator):
        with torch.cpu.amp.autocast(enabled=True), torch.no_grad():
            for i in range(10):
                pre = time.time()
                out = generator(data["train"][0]["audio"]["array"])
                print(f"Generate time costs {time.time()-pre} seconds")
                print(f"output = {out}")
    generate(generator)

else: 
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")

    # numpy does not support bf16
    with torch.cpu.amp.autocast(enabled=False), torch.no_grad():
        for i in range(10):
            pre = time.time()
            diarization = pipeline("./OSR_us_000_0010_8k.wav")
            print(f"Generate time costs {time.time()-pre} seconds")