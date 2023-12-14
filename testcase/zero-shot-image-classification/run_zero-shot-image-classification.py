from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import intel_extension_for_pytorch as ipex
import time
import logging
import argparse
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
    parser.add_argument("--bf16", default='False', type=str2bool)
    parser.add_argument("--ipex_optimize", default='False', type=str2bool)
    parser.add_argument("--jit", default='False', type=str2bool)
    parser.add_argument("--torch_compile", default='False', type=str2bool)
    parser.add_argument("--torch_dtype", default="float32", type=str)
    parser.add_argument("--backend", default="ipex", type=str)
    args = parser.parse_args()
    return args


def load_model(model_id, seed, model_dtype, device):
    torch.manual_seed(seed)
    model = CLIPModel.from_pretrained(
        model_id, torch_dtype=model_dtype, return_dict=False
    )
    processor = CLIPProcessor.from_pretrained(model_id, torch_dtype=model_dtype)

    model.to(device)
    return model, processor


def benchmark(model, inputs, seed, nb_pass):
    elapsed_time = []
    for _ in range(nb_pass):
        start = time.time()
        torch.manual_seed(seed)
        outputs = model(**inputs)
        duration = time.time() - start
        elapsed_time.append(duration)
        logits_per_image = outputs[1]
        logits_per_image = torch.reshape(logits_per_image, (1, 2))
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        logging.info(probs)

    return elapsed_time


def prepare_jit_inputs():
    input_ids_example = torch.randint(200, size=MODEL_INPUT_SIZE["input_ids"])
    pixel_values_example = torch.randn(MODEL_INPUT_SIZE["pixel_values"])
    attention_mask_example = torch.randint(1, size=MODEL_INPUT_SIZE["attention_mask"])

    return input_ids_example, pixel_values_example, attention_mask_example


def apply_jit_trace(model, dtype):
    logging.info("using jit trace for acceleration...")
    (
        input_ids_example,
        pixel_values_example,
        attention_mask_example,
    ) = prepare_jit_inputs()
    example_inputs = {
        "input_ids": input_ids_example,
        "pixel_values": pixel_values_example,
        "attention_mask": attention_mask_example,
    }

    with torch.cpu.amp.autocast(
        enabled=True if dtype == torch.bfloat16 else torch.float32, dtype=dtype
    ), torch.no_grad():
        model = torch.jit.trace(
            model, example_kwarg_inputs=example_inputs, strict=False
        )

    model = torch.jit.freeze(model.eval())

    model(**example_inputs)
    model(**example_inputs)

    return model


def optimize_with_ipex(model, dtype):
    logging.info("using ipex optimize for acceleration...")

    (
        input_ids_example,
        pixel_values_example,
        attention_mask_example,
    ) = prepare_jit_inputs()

    model = ipex.optimize(
        model,
        dtype=dtype,
        inplace=True,
        sample_input=(input_ids_example, pixel_values_example, attention_mask_example),
    )

    return model


def apply_torch_compile(model, backend):
    logging.info(f"using torch compile with {backend} backend for acceleration...")
    model = torch.compile(model, backend=backend)

    return model


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args={args}")
    model_id = args.model_id
    use_bf16 = args.bf16
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile
    backend = args.backend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
    model, processor = load_model(model_id, SEED, torch_dtype, device)
    
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if use_ipex_optimize:
        model = optimize_with_ipex(model, dtype=dtype)
    if use_jit:
        model = apply_jit_trace(model, dtype=dtype)
    if use_torch_compile:
        model = apply_torch_compile(model, backend)

    image = Image.open(requests.get(IMG_URL, stream=True).raw)
    inputs = processor(text=TEXT, images=image, return_tensors="pt", padding=True)
    
    if device == 'cuda':
        inputs = inputs.to(device)
        
    if use_bf16:
        logging.info("using BF16 for acceleration...")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            elapsed_time = benchmark(model, inputs, SEED, 10)
    else:
        elapsed_time = benchmark(model, inputs, SEED, 10)

    logging.info(f"total time: {elapsed_time}")
    logging.info(f"average time: {sum(elapsed_time[3:])/len(elapsed_time[3:])}")
