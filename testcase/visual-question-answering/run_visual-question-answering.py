import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, ViltProcessor, ViltForQuestionAnswering
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str, required=True)
parser.add_argument("--bf16", action="store_true")
args = parser.parse_args()
model_id = args.model_id

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "how many dogs are in the picture?"

if "vilt" in model_id:
    processor = ViltProcessor.from_pretrained(model_id)
    model = ViltForQuestionAnswering.from_pretrained(model_id)
    inputs = processor(raw_image, question, return_tensors="pt")
    with torch.cpu.amp.autocast(enabled=args.bf16), torch.no_grad():
        for i in range(10):
            pre = time.time()
            outputs = model(**inputs)
            print(f"Generate time costs {time.time()-pre} seconds")
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
else:
    processor = BlipProcessor.from_pretrained(model_id)
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = BlipForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch_dtype)

    inputs = processor(raw_image, question, return_tensors="pt")

    for i in range(10):
        pre = time.time()
        out = model.generate(**inputs)
        print(f"Generate time costs {time.time()-pre} seconds")

    print(processor.decode(out[0], skip_special_tokens=True))

