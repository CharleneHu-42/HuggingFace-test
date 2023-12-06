"""
Use IPEX+JIT to accelerate inference performance
Steps:
- invoke ipex.optimize() to apply optimizations 
- invoke torch.jit.trace() and torch.jit.freeze() for model conversion 
- inference 
"""

from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
import torch
import intel_extension_for_pytorch as ipex
import time
import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO)
SEED = 20
PROMPT = "An astronaut riding a green horse"

MODEL_INPUT_SIZE = {
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "sample": (2, 4, 128, 128),
        "timestep": 1.0,
        "encoder_hidden_states": (2, 77, 2048),
        "text_embeds": (2, 1280),
        "time_ids": (2, 6),
    },
    "runwayml/stable-diffusion-v1-5": {
        "sample": (2, 4, 64, 64),
        "timestep": 90,
        "encoder_hidden_states": (2, 77, 768),
    },
    "stabilityai/stable-diffusion-2-1": {
        "sample": (2, 4, 96, 96),
        "timestep": 90,
        "encoder_hidden_states": (2, 77, 1024),
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
    if model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype, use_safetensors=True
        )
    elif model_id == "runwayml/stable-diffusion-v1-5":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype
        )
    elif model_id == "stabilityai/stable-diffusion-2-1":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(
            "the given model id is incorrect. it should be one of [10, 15, 21]."
        )

    pipe = pipe.to("cpu")

    return pipe


def benchmark(pipe, prompt, seed, nb_pass):
    elapsed_time = []
    for i in range(nb_pass):
        start = time.time()
        torch.manual_seed(seed)
        image = pipe(prompt=prompt).images[0]
        duration = time.time() - start
        elapsed_time.append(duration)
        image.save(f"img_{i}.jpg", "JPEG")

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

    if model_id == "stabilityai/stable-diffusion-xl-base-1.0":
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
    pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)

    input_example = prepare_jit_inputs(model_id, False)

    # optimize with IPEX
    pipe.unet = ipex.optimize(
        pipe.unet.eval(), dtype=dtype, inplace=True, sample_input=input_example
    )
    pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=dtype, inplace=True)
    pipe.text_encoder = ipex.optimize(
        pipe.text_encoder.eval(), dtype=dtype, inplace=True
    )

    return pipe


def apply_torch_compile(pipe):
    logging.info("using torch compile for acceleration...")
    pipe.unet = torch.compile(pipe.unet, backend="ipex")

    return pipe


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args={args}")
    model_id = args.model_id
    use_bf16 = args.bf16
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile

    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32

    pipe = load_model(model_id, SEED, model_dtype=torch_dtype)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    if use_ipex_optimize:
        pipe = optimize_with_ipex(pipe, model_id, dtype=dtype)
    if use_jit:
        pipe = apply_jit_trace(pipe, model_id, ["unet"], dtype=dtype)
    if use_torch_compile:
        pipe = apply_torch_compile(pipe)

    if use_bf16:
        logging.info("using BF16 for acceleration...")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            elapsed_time = benchmark(pipe, PROMPT, SEED, 5)
    else:
        elapsed_time = benchmark(pipe, PROMPT, SEED, 5)

    logging.info(f"total time: {elapsed_time}")
    logging.info(f"average time: {sum(elapsed_time[3:])/len(elapsed_time[3:])}")
