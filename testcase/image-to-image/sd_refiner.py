import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import time 

seed=20
torch.manual_seed(seed)
model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, use_safetensors=True
)
pipe = pipe.to("cpu")

url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
init_image = load_image(url).convert("RGB")
prompt = "a photo of an astronaut riding a horse on mars"


duration_list = []
for i in range(10):
    start = time.time()
    torch.manual_seed(seed)
    image = pipe(prompt, image=init_image).images
    duration = time.time() - start
    duration_list.append(duration)   
    image[0].save(f"img_{i}.jpg", 'JPEG')
    
print(f"total time: {duration_list}")

last_five = duration_list[5:]
print(f"average time: {sum(last_five)/len(last_five)}")