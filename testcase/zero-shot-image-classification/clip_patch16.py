from PIL import Image
import requests
import torch 
import time 

from transformers import CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch16"

seed=24
torch.manual_seed(seed)
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)


time_list = []
for _ in range(10):
    torch.manual_seed(seed)
    start = time.time()
    outputs = model(**inputs)
    end = time.time()
    time_list.append(end-start)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)

print(f"total time: {time_list}")

last_five = time_list[5:]

print(f"average time: {sum(last_five)/len(last_five)}")