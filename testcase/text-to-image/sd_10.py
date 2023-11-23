from diffusers import DiffusionPipeline
import torch
import time


seed=20
torch.manual_seed(seed)
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
pipe.to("cpu")

prompt = ["An astronaut riding a green horse"]

duration_list = []
for i in range(10):
    start = time.time()
    torch.manual_seed(seed)
    image = pipe(prompt=prompt).images[0]
    duration = time.time() - start
    duration_list.append(duration)    
    image.save(f"img_{i}.jpg", 'JPEG')
    
print(f"total time: {duration_list}")

last_five = duration_list[5:]
print(f"average time: {sum(last_five)/len(last_five)}")