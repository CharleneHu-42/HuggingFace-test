import argparse
import torch
import time
import random
from transformers import AwqConfig, BitsAndBytesConfig, set_seed

set_seed(42)
torch.use_deterministic_algorithms(True)

def str2bool(str):
    return True if str.lower() == "true" else False

def synchronize_device(device):
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None, type=str, required=True)
    parser.add_argument("--autocast_dtype", default="float32", type=str)
    parser.add_argument("--ipex_optimize", default="False", type=str2bool)
    parser.add_argument("--jit", default="False", type=str2bool)
    parser.add_argument("--torch_compile", default="False", type=str2bool)
    parser.add_argument("--model_dtype", default="float32", type=str)
    parser.add_argument("--backend", default="inductor", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_beams", default=4, type=int)
    parser.add_argument(
        "--input_tokens",
        default=32,
        type=int,
        help="choose from [32, 64, 128, 256, 512, 1024]",
    )
    parser.add_argument("--output_tokens", default=32, type=int)
    parser.add_argument("--do_sample", default="False", type=str2bool)
    parser.add_argument("--ipex_optimize_transformers", default="False", type=str2bool)
    parser.add_argument("--warm_up_steps", default=10, type=int)
    parser.add_argument("--run_steps", default=10, type=int)
    parser.add_argument("--optimum_intel", default="False", type=str2bool)
    parser.add_argument("--quant_algo", default=None, type=str,
        help="choose from [bitsandbytes, autoawq]")
    parser.add_argument("--quant_dtype", default=None, type=str,
        help="choose from [int, nf4, fp4, int4]")
    parser.add_argument("--tp_plan", default=None, type=str,
        help="tensor parallelism strategy")
    args = parser.parse_args()

    return args


def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    else:
        return torch.float32


def get_bitsandbytes_config(quant_type):
    if quant_type == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_type in ("nf4", "fp4"):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_quant_type=quant_type,
                                                 bnb_4bit_use_double_quant=True)
    else:
        quantization_config = None

    return quantization_config


def get_awq_config(quant_type):
    if quant_type == "int4":
        quantization_config = AwqConfig(version="ipex")
    else:
        quantization_config = None

    return quantization_config


def wrapped_forward(self, model_inputs, **forward_params):
    start_time = time.time()
    model_outputs = self.__class__._orig_forward(self, model_inputs, **forward_params)
    end_time = time.time()
    self.forward_time += end_time - start_time
    return model_outputs


def wrap_forward_for_benchmark(pipeline):
    pipeline.forward_time = 0
    pipeline.__class__._orig_forward = pipeline.__class__._forward
    pipeline.__class__._forward = wrapped_forward


def get_batched_prompts(prompt, batch_size):
    prompt_list = [prompt]
    token_list = prompt.split(" ")
    assert len(token_list) > 18
    for _ in range(batch_size - 1):
        prompt_len = random.randint(16, len(token_list) - 2)
        new_prompt = " ".join(token_list[:prompt_len])
        prompt_list.append(new_prompt)
    return prompt_list
