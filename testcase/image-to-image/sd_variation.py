from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
import time 
import torch 

seed=20
torch.manual_seed(seed)
model_id = "lambdalabs/sd-image-variations-diffusers"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  model_id,
  revision="v2.0",
  )

sd_pipe = sd_pipe.to("cpu")

im = Image.open("new.jpg")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to("cpu").unsqueeze(0)


duration_list = []
for i in range(10):
    start = time.time()
    torch.manual_seed(seed)
    out = sd_pipe(inp, guidance_scale=3)
    duration = time.time() - start
    duration_list.append(duration)   
    out["images"][0].save(f"img_{i}.jpg", 'JPEG')
    
    
print(f"total time: {duration_list}")

last_five = duration_list[5:]
print(f"average time: {sum(last_five)/len(last_five)}")
