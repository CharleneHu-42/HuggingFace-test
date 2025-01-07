from PIL import Image
import av
import torch
import time
import logging
import sys
import numpy as np
from transformers.utils import ContextManagers
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
)


sys.setrecursionlimit(10000000)
SEED = 42

import os

sys.path.append(os.path.dirname(__file__) + "/..")
from common import get_args, get_torch_dtype, wrap_forward_for_benchmark, synchronize_device

logging.basicConfig(level=logging.INFO)

inference_context = [torch.inference_mode()]


def generate(model, inputs, warm_up_steps, run_steps):
    time_costs = []
    # forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            # model.forward_time = 0
            set_seed(SEED)
            synchronize_device(model.device.type)
            pre = time.time()
            if model_id == "KangarooGroup/kangaroo":
                outputs, history = model.chat(**inputs)
            else:
                outputs = model.generate(**inputs, generation_config=generation_config)
            synchronize_device(model.device.type)
            time_costs.append((time.time() - pre) * 1000)
            # forward_times.append(model.forward_time * 1000)

    average_time = sum(time_costs[warm_up_steps:]) / run_steps
    # average_fwd_time = sum(forward_times[warm_up_steps:]) / run_steps
    logging.info(f"total time [ms]: {time_costs}")
    logging.info(
        f"model average time [ms] {average_time}"
    )
    if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        generate_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
    elif model_id == "KangarooGroup/kangaroo":
        generate_text = outputs
    logging.info(f"output = {generate_text}")


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_video_inputs(model_id):
    video_path = "./datasets/sample_demo_1.mp4"
    container = av.open(video_path)
    # sample uniformly 8 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    videos = read_video_pyav(container, indices)
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": "Why is this video funny?"},
                {"type": "video"},
                ],
        },
    ]
    prompt = conversation[0]["content"][0]["text"]
    if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, videos=videos, padding=True, return_tensors="pt").to(model.device)
    elif model_id == "KangarooGroup/kangaroo":
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        inputs = dict(video_path=video_path,
                    query=prompt,
                    tokenizer=tokenizer,
                    max_new_tokens=20,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    )

    return inputs


def get_model_and_proceessor(model_id, torch_dtype, device):
    model, processor, tokenizer = None, None, None
    if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
        ).to(device)

        processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    elif model_id == "KangarooGroup/kangaroo":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)

    return model, processor, tokenizer


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id

    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    # define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    set_seed(SEED)
    model, processor, tokenizer = get_model_and_proceessor(model_id, torch_dtype, device)
    # wrap_forward_for_benchmark(model)
    inputs = get_video_inputs(model_id)

    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.use_cache = True
    generation_config.temperature = 1.0
    generation_config.max_new_tokens = 20
    generation_config.min_new_tokens = 20
    generation_config.top_p = 1.0

    if args.jit:
        raise ValueError("Image-feature-extraction does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        # import torch._dynamo.config
        # torch._dynamo.config.capture_scalar_outputs = True
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        model.forward = torch.compile(model.forward, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        model = ipex.optimize(model, dtype=torch_dtype, inplace=True)

    generate(model, inputs, warm_up_steps, run_steps)
