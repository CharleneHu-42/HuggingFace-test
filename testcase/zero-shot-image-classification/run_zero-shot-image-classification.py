from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import intel_extension_for_pytorch as ipex
import time
import logging 
import argparse


SEED = 24
TEXT = ["a photo of a cat", "a photo of a dog"]
IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
MODEL_ID_NAME = {
    "14": "openai/clip-vit-large-patch14",
    "16": "openai/clip-vit-base-patch16",
    "32": "openai/clip-vit-base-patch32"
}
MODEL_DTYPE = {
    "14": "fp32",
    "16": "fp32",
    "32": "fp32"
}


MODEL_INPUT_SIZE = {"input_ids": (1, 7), "pixel_values": (1,3,224,224), "attention_mask": (1, 7)}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default='14',
        help="model id for the respective model, choose one from (14, 16, 32)"
    )
    
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="whether to use pytorch bfloat16 data type for acceleration on intel platforms"
    )
    
    parser.add_argument(
        "--ipex_optimize",
        action="store_true",
        help="whether to use ipex for acceleration on intel platforms"
    )
    
    parser.add_argument(
        "--jit",
        action="store_true",
        help="whether to use jit for acceleration on intel platforms"
    )
    
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="whether to use torch.compile() for acceleration on intel platforms"
    )
    
    args = parser.parse_args()   
    return args


def load_model(model_id, seed, model_dtype):
    model_name = MODEL_ID_NAME[model_id]        
    torch.manual_seed(seed)
    model = CLIPModel.from_pretrained(model_name, torch_dtype=model_dtype, return_dict=False)
    processor = CLIPProcessor.from_pretrained(model_name, torch_dtype=model_dtype)
    
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
        logits_per_image = torch.reshape(logits_per_image, (1,2))
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        print(probs)
    
    return elapsed_time


def prepare_jit_inputs():

    input_ids_example = torch.randint(200, size=MODEL_INPUT_SIZE['input_ids'])
    pixel_values_example = torch.randn(MODEL_INPUT_SIZE['pixel_values'])
    attention_mask_example = torch.randint(1, size=MODEL_INPUT_SIZE['attention_mask'])
        
    return input_ids_example, pixel_values_example, attention_mask_example


def apply_jit_trace(model):
    
    logging.info("using jit trace for acceleration...")
    input_ids_example, pixel_values_example, attention_mask_example = prepare_jit_inputs()
    example_inputs = {"input_ids": input_ids_example, 
                        "pixel_values": pixel_values_example,
                        "attention_mask": attention_mask_example
                    } 
    
    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
        model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False)
    
    model = torch.jit.freeze(model.eval())
    
    model(**example_inputs)
    model(**example_inputs)
                    
    return model 


def optimize_with_ipex(model):

    logging.info("using ipex optimize for acceleration...")
    
    input_ids_example, pixel_values_example, attention_mask_example = prepare_jit_inputs()
    
    model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True, sample_input=(input_ids_example, pixel_values_example, attention_mask_example))

    return model 


def apply_torch_compile(model):

    logging.info("using torch compile for acceleration...")
    model = torch.compile(model, backend="ipex")
    
    return model 
 
 
def get_model_dtype(model_id):
    model_dtype = MODEL_DTYPE[model_id]
    if model_dtype == 'bf16':
        return torch.bfloat16
    elif model_dtype == 'fp32':
        return torch.float32
    else:
        raise ValueError("please use either bf16 or float32 for model_dtype.")    


if __name__ == '__main__':
     
    args = get_args()
    model_id = args.model_id 
    use_bf16 = args.bf16
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile
    
    model_dtype = get_model_dtype(model_id)
    
    model, processor = load_model(model_id, SEED, model_dtype)

    image = Image.open(requests.get(IMG_URL, stream=True).raw)
    inputs = processor(text=TEXT, images=image, return_tensors="pt", padding=True)
    
    if use_ipex_optimize:
        model = optimize_with_ipex(model)
    if use_jit:
        model = apply_jit_trace(model)
    if use_torch_compile:
        model = apply_torch_compile(model) 
        
    if use_bf16:          
        logging.info("using BF16 for acceleration...")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            elapsed_time = benchmark(model, inputs, SEED, 10)
    else:
        elapsed_time = benchmark(model, inputs, SEED, 10)    
        
    logging.info(f"total time: {elapsed_time}")
    logging.info(f"average time: {sum(elapsed_time[3:])/len(elapsed_time[3:])}")
     