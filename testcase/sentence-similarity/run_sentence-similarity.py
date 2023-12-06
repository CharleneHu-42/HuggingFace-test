from transformers import AutoTokenizer, AutoModel
import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
import time
import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO)
SEED = 20
SENTENCES = ["This is an example sentence", "Each sentence is converted"]
CHI_SENTENCES = ["如何更换花呗绑定银行卡", "花呗更改绑定银行卡"]

MODEL_INPUT_SIZE = {
    "input_ids": (1, 7),
    "token_type_ids": (1, 7),
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
    args = parser.parse_args()
    return args


def load_model(model_id, seed, model_dtype):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype=model_dtype)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=model_dtype, return_dict=False
    )

    return model, tokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def benchmark(model, encoded_input, seed, nb_pass, model_id):
    elapsed_time = []
    for _ in range(nb_pass):
        start = time.time()
        torch.manual_seed(seed)
        model_output = model(**encoded_input)
        duration = time.time() - start
        elapsed_time.append(duration)
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        if model_id != "shibing624/text2vec-base-chinese":
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        logging.info(sentence_embeddings[0][:10])

    return elapsed_time


def prepare_jit_inputs(model_id):
    input_ids_example = torch.randint(6000, size=MODEL_INPUT_SIZE["input_ids"])
    attention_mask_example = torch.randint(1, size=MODEL_INPUT_SIZE["attention_mask"])

    if model_id == "sentence-transformers/all-mpnet-base-v2":
        example_inputs = {
            "input_ids": input_ids_example,
            "attention_mask": attention_mask_example,
        }
    else:
        token_type_ids_example = torch.randint(
            1, size=MODEL_INPUT_SIZE["token_type_ids"]
        )
        example_inputs = {
            "input_ids": input_ids_example,
            "token_type_ids": token_type_ids_example,
            "attention_mask": attention_mask_example,
        }
    return example_inputs


def apply_jit_trace(model, model_id, dtype):
    logging.info("using jit trace for acceleration...")
    example_inputs = prepare_jit_inputs(model_id)

    with torch.cpu.amp.autocast(
        enabled=True if dtype == torch.bfloat16 else False, dtype=dtype
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
    model = ipex.optimize(model, dtype=dtype)

    return model


def apply_torch_compile(model):
    logging.info("using torch compile for acceleration...")
    model = torch.compile(model, backend="ipex")

    return model


if __name__ == "__main__":
    args = get_args()
    logging.info("args={args}")
    model_id = args.model_id
    use_bf16 = args.bf16
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile

    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32

    if "shibing624/text2vec-base-chinese" in model_id:
        sentences = CHI_SENTENCES
    else:
        sentences = SENTENCES

    model, tokenizer = load_model(model_id, SEED, model_dtype=torch_dtype)
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    if use_ipex_optimize:
        model = optimize_with_ipex(model, dtype=dtype)
    if use_jit:
        model = apply_jit_trace(model, model_id, dtype=dtype)
    if use_torch_compile:
        model = apply_torch_compile(model)

    if use_bf16:
        logging.info("using BF16 for acceleration...")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            elapsed_time = benchmark(model, encoded_input, SEED, 10, model_id)
    else:
        elapsed_time = benchmark(model, encoded_input, SEED, 10, model_id)

    logging.info(f"total time: {elapsed_time}")
    logging.info(f"average time: {sum(elapsed_time[3:])/len(elapsed_time[3:])}")
