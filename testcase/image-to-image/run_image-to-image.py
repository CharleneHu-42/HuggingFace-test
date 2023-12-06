import PIL
import requests
import torch
import intel_extension_for_pytorch as ipex
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionImageVariationPipeline,
)
import time
import argparse
from torchvision import transforms
import os
import logging
logging.basicConfig(level=logging.INFO)

SEED = 20
IMG_URL = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
SAMPLE_IMAGE = "example.jpg"
PROMPT = "turn him into cyborg"

MODEL_INPUT_SIZE = {
    "timbrooks/instruct-pix2pix": {
        "sample": (3, 8, 64, 64),
        "timestep": 1.0,
        "encoder_hidden_states": (3, 77, 768),
    },
    "stabilityai/stable-diffusion-xl-refiner-1.0": {
        "sample": (2, 4, 64, 64),
        "timestep": 1.0,
        "encoder_hidden_states": (2, 77, 1280),
        "text_embeds": (2, 1280),
        "time_ids": (2, 5),
    },
    "lambdalabs/sd-image-variations-diffusers": {
        "sample": (2, 4, 64, 64),
        "timestep": 90,
        "encoder_hidden_states": (2, 1, 768),
    },
}


def str2bool(str):
    return True if str.lower() == 'true' else False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None, type=str, required=True)
    parser.add_argument("--bf16", default='False', type=str2bool)
    parser.add_argument("--ipex_optimize", default='False', type=str2bool)
    parser.add_argument("--jit", default='False', type=str2bool)
    parser.add_argument("--torch_compile", default='False', type=str2bool)
    parser.add_argument("--torch_dtype", default="float32", type=str)

    args = parser.parse_args()
    return args


def load_model(model_id, seed, model_dtype):
    torch.manual_seed(seed)
    if model_id == "timbrooks/instruct-pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype, safety_checker=None
        )
        pipe = pipe.to("cpu")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif model_id == "stabilityai/stable-diffusion-xl-refiner-1.0":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype, use_safetensors=True
        )
        pipe = pipe.to("cpu")
    elif model_id == "lambdalabs/sd-image-variations-diffusers":
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype, revision="v2.0"
        )
        pipe = pipe.to("cpu")
    else:
        raise ValueError(
            f"the given model id is incorrect. it is {model_id}"
        )

    return pipe


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def benchmark(pipe, prompt, image, seed, nb_pass, model_id):
    elapsed_time = []
    for i in range(nb_pass):
        start = time.time()
        torch.manual_seed(seed)
        if model_id == "lambdalabs/sd-image-variations-diffusers":
            new_image = pipe(image, guidance_scale=3).images[0]
        elif model_id == "timbrooks/instruct-pix2pix":
            new_image = pipe(
                prompt, image=image, num_inference_steps=10, image_guidance_scale=1
            ).images[0]
        else:
            new_image = pipe(prompt=prompt, image=image).images[0]
        duration = time.time() - start
        elapsed_time.append(duration)
        new_image.save(f"img_{i}.jpg", "JPEG")

    return elapsed_time


def prepare_jit_inputs(model_id, jit):
    # import inspect
    # signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.__call__)

    sample_size = MODEL_INPUT_SIZE[model_id]["sample"]
    timestep_size = MODEL_INPUT_SIZE[model_id]["timestep"]
    encoder_hidden_states_size = MODEL_INPUT_SIZE[model_id]["encoder_hidden_states"]

    sample_example = torch.randn(sample_size)
    timestep_example = torch.tensor(timestep_size)
    encoder_hidden_states_example = torch.randn(encoder_hidden_states_size)

    if model_id == "stabilityai/stable-diffusion-xl-refiner-1.0":
        text_embeds_size = MODEL_INPUT_SIZE[model_id]["text_embeds"]
        time_ids_size = MODEL_INPUT_SIZE[model_id]["time_ids"]
        text_embeds_example = torch.randn(text_embeds_size)
        time_ids_example = torch.randn(time_ids_size)

        if jit:
            example_inputs = {
                "sample": sample_example,
                "timestep": timestep_example,
                "encoder_hidden_states": encoder_hidden_states_example,
                "added_cond_kwargs": {
                    "text_embeds": text_embeds_example,
                    "time_ids": time_ids_example,
                },
            }
        else:
            example_inputs = (
                sample_example,
                timestep_example,
                encoder_hidden_states_example,
                None,
                None,
                None,
                None,
                {"text_embeds": text_embeds_example, "time_ids": time_ids_example},
            )
    else:
        if jit:
            example_inputs = {
                "sample": sample_example,
                "timestep": timestep_example,
                "encoder_hidden_states": encoder_hidden_states_example,
            }
        else:
            example_inputs = (
                sample_example,
                timestep_example,
                encoder_hidden_states_example,
            )

    return example_inputs


def apply_jit_trace(pipeline, model_id, attr_list, dtype):
    logging.info("using jit trace for acceleration...")
    for name in attr_list:
        model = getattr(pipeline, name)
        model.eval()
        example_inputs = prepare_jit_inputs(model_id, True)

        with torch.cpu.amp.autocast(
            enabled=True if dtype == torch.bfloat16 else False, dtype=dtype
        ), torch.no_grad():
            traced_model = torch.jit.trace(
                model, example_kwarg_inputs=example_inputs, strict=False
            )

        traced_model = torch.jit.freeze(traced_model.eval())

        traced_model(**example_inputs)
        traced_model(**example_inputs)

        setattr(pipeline, name, traced_model)

    return pipeline


def optimize_with_ipex(pipe, model_id, dtype):
    logging.info("using ipex optimize for acceleration...")
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    input_example = prepare_jit_inputs(model_id, False)

    # optimize with IPEX
    pipe.unet = ipex.optimize(
        pipe.unet.eval(), dtype=dtype, inplace=True, sample_input=input_example
    )
    pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=dtype, inplace=True)

    return pipe


def apply_torch_compile(pipe):
    logging.info("using torch compile for acceleration...")
    pipe.unet = torch.compile(pipe.unet, backend="ipex")

    return pipe


def load_image(model_id):
    if model_id == "lambdalabs/sd-image-variations-diffusers":
        image_path = os.path.join(os.path.dirname(__file__), SAMPLE_IMAGE)
        image = PIL.Image.open(image_path)
        tform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=False,
                ),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        image = tform(image).to("cpu").unsqueeze(0)
    else:
        image = download_image(IMG_URL)

    return image


if __name__ == "__main__":
    args = get_args()
    model_id = args.model_id
    use_bf16 = args.bf16
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile
    logging.info(f"args = {args}")
    image = load_image(model_id)
    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
    pipe = load_model(model_id, SEED, torch_dtype)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if use_ipex_optimize:
        pipe = optimize_with_ipex(pipe, model_id, dtype=dtype)
    if use_jit:
        pipe = apply_jit_trace(pipe, model_id, ["unet"], dtype=dtype)
    if use_torch_compile:
        pipe = apply_torch_compile(pipe)

    if use_bf16:
        logging.info("using BF16 for acceleration...")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            elapsed_time = benchmark(pipe, PROMPT, image, SEED, 5, model_id)
    else:
        elapsed_time = benchmark(pipe, PROMPT, image, SEED, 5, model_id)

    logging.info(f"total time: {elapsed_time}")
    logging.info(f"average time: {sum(elapsed_time[3:])/len(elapsed_time[3:])}")
