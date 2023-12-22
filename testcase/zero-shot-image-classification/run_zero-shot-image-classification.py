from PIL import Image
import requests
import torch
import time
import logging
import argparse
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

SEED = 24
TEXT = ["a photo of a cat", "a photo of a dog"]
IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


MODEL_INPUT_SIZE = {
    "input_ids": (1, 7),
    "pixel_values": (1, 3, 224, 224),
    "attention_mask": (1, 7),
}

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

def load_model(model_id, seed, model_dtype, device):
    torch.manual_seed(seed)
    classifier = pipeline("zero-shot-image-classification", model=model_id, torch_dtype=model_dtype, device=device, return_dict=False)
    return classifier 


def benchmark(pipeline, image_url, labels, seed, nb_pass):
    elapsed_time = []
    for _ in range(nb_pass):
        start = time.time()
        torch.manual_seed(seed)
        outputs = pipeline(image_url, candidate_labels=labels)
        duration = time.time() - start
        elapsed_time.append(duration*1000)
        logging.info(outputs)
    return elapsed_time


def prepare_jit_inputs(device):
    
    input_ids_example = torch.randint(200, size=MODEL_INPUT_SIZE["input_ids"])
    pixel_values_example = torch.randn(MODEL_INPUT_SIZE["pixel_values"])
    attention_mask_example = torch.randint(1, size=MODEL_INPUT_SIZE["attention_mask"])

    input_ids_example = input_ids_example.to(device)
    pixel_values_example = pixel_values_example.to(device)
    attention_mask_example = attention_mask_example.to(device)
        
    return input_ids_example, pixel_values_example, attention_mask_example


def apply_jit_trace(classifier, dtype, device):
    logging.info("using jit trace for acceleration...")
    (
        input_ids_example,
        pixel_values_example,
        attention_mask_example,
    ) = prepare_jit_inputs(device)
    
    example_inputs = {
        "input_ids": input_ids_example,
        "pixel_values": pixel_values_example,
        "attention_mask": attention_mask_example,
    }

    with torch.autocast(device_type=device, dtype=dtype), torch.no_grad():
        classifier.model = torch.jit.trace(
            classifier.model, example_kwarg_inputs=example_inputs, strict=False
        )

    classifier.model = torch.jit.freeze(classifier.model.eval())

    classifier.model(**example_inputs)
    classifier.model(**example_inputs)

    return classifier


def optimize_with_ipex(classifier, dtype, device):
    logging.info("using ipex optimize for acceleration...")
    import intel_extension_for_pytorch as ipex
    
    (
        input_ids_example,
        pixel_values_example,
        attention_mask_example,
    ) = prepare_jit_inputs(device)

    classifier.model = ipex.optimize(
        classifier.model,
        dtype=dtype,
        inplace=True,
        sample_input=(input_ids_example, pixel_values_example, attention_mask_example),
    )

    return classifier


def apply_torch_compile(classifier, backend):
    logging.info(f"using torch compile with {backend} backend for acceleration...")
    if backend == "ipex":
        import intel_extension_for_pytorch as ipex
    classifier.model = torch.compile(classifier.model, backend=backend)
    return classifier

def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16 
    else:
        return torch.float32
    
if __name__ == "__main__":
    args = get_args()
    logging.info(f"args={args}")
    model_id = args.model_id
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile
    backend = args.backend

    device = args.device 
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
    
    dtype = get_torch_dtype(args.compute_dtype)
    torch_dtype = get_torch_dtype(args.model_dtype)
    classifier = load_model(model_id, SEED, torch_dtype, device)
    
    if use_ipex_optimize:
        classifier = optimize_with_ipex(classifier, dtype=dtype, device=device)
    if use_jit:
        classifier = apply_jit_trace(classifier, dtype=dtype, device=device)
    if use_torch_compile:
        classifier = apply_torch_compile(classifier, backend)

    with torch.autocast(device_type=device, dtype=dtype), torch.no_grad():
        elapsed_time = benchmark(classifier, IMG_URL, TEXT, SEED, 20)

    logging.info(f"total time [ms]: {elapsed_time}")
    logging.info(f"average time [ms]: {sum(elapsed_time[10:])/len(elapsed_time[10:])}")