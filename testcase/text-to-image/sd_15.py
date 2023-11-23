from diffusers import StableDiffusionPipeline
import torch
import time

seed=24
torch.manual_seed(seed)

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cpu")

prompt = "a photo of an astronaut riding a horse on mars"

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