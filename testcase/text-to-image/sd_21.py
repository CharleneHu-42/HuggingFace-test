
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import time

seed=20
torch.manual_seed(seed)
model_id = "stabilityai/stable-diffusion-2-1"
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cpu")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a photo of an astronaut riding a horse on mars"

duration_list = []
for i in range(10):
    start = time.time()
    torch.manual_seed(seed)
    image = pipe(prompt=prompt).images[0]
    duration = time.time() - start
    duration_list.append(duration)    
    image.save(f"sd_21_img_{i}.jpg", 'JPEG')
    
print(f"total time: {duration_list}")

last_five = duration_list[5:]
print(f"average time: {sum(last_five)/len(last_five)}")