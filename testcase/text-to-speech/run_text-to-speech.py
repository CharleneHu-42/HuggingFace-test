from transformers import pipeline
from datasets import load_from_disk
import torch
import time
import logging
from transformers.utils import ContextManagers

logging.basicConfig(level=logging.INFO)

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/..")

from common import get_args, get_torch_dtype, wrap_forward_for_benchmark
import common

inference_context = [torch.inference_mode()]

def generate(generator, device, dtype, forward_params, enable):
    time_costs = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(RUN + WARMUP):
            generator.forward_time = 0
            pre = time.time()
            output = generator(
                "Hello, my dog is cooler than you!", forward_params=forward_params
            )
            time_costs.append((time.time() - pre) * 1000)
            forward_times.append(generator.forward_time * 1000)

    average_time = sum(time_costs[WARMUP:]) / RUN
    average_fwd_time = sum(forward_times[WARMUP:]) / RUN
    logging.info(f"total time [ms]: {time_costs}")
    logging.info(
        f"pipeline average time [ms] {average_time}, average fwd time [ms] {average_fwd_time}({average_fwd_time/average_time})"
    )
    logging.info(f"output = {output}")


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    model_id = args.model_id

    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    synthesiser = pipeline(
        "text-to-speech", model_id, device=device, torch_dtype=torch_dtype
    )
    wrap_forward_for_benchmark(synthesiser)

    embeddings_dataset = load_from_disk("./datasets/speech_vector")
    speaker_embedding = (
        torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0).to(device)
    )

    # You can replace this embedding with your own as well.
    forward_params = (
        {"speaker_embeddings": speaker_embedding} if "t5" in model_id else {}
    )
    forward_params["do_sample"] = False

    if args.jit:
        raise ValueError("Text-to-speech does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        synthesiser.model.generate = torch.compile(
            synthesiser.model.generate, backend=args.backend
        )
        synthesiser.model = torch.compile(synthesiser.model, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        synthesiser.model = ipex.optimize(synthesiser.model, dtype=dtype, inplace=True)

    generate(synthesiser, device, dtype, forward_params, enable)
