import argparse
import torch


def str2bool(str):
    return True if str.lower() == 'true' else False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None, type=str, required=True)
    parser.add_argument("--compute_dtype", default="float32", type=str)
    parser.add_argument("--ipex_optimize", default='False', type=str2bool)
    parser.add_argument("--jit", default='False', type=str2bool)
    parser.add_argument("--torch_compile", default='False', type=str2bool)
    parser.add_argument("--model_dtype", default="float32", type=str)
    parser.add_argument("--backend", default="inductor", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    return args


def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16 
    else:
        return torch.float32
