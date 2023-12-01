from transformers import AutoTokenizer, AutoModel
import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
import time
import sys 
import argparse
import logging 

SEED = 20
SENTENCES = ['This is an example sentence', 'Each sentence is converted']
CHI_SENTENCES = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
MODEL_ID_NAME = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "chin": "shibing624/text2vec-base-chinese"
}
MODEL_DTYPE = {
    "minilm": "fp32",
    "mpnet": "fp32",
    "chin": "fp32"
}


MODEL_INPUT_SIZE = {"input_ids": (1, 7), "token_type_ids": (1, 7), "attention_mask": (1, 7)}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default='minilm',
        help="model id for the respective model, choose one from (minilm, mpnet, chin)"
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=model_dtype)
    model = AutoModel.from_pretrained(model_name, torch_dtype=model_dtype, return_dict=False)
    
    return model, tokenizer  


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def benchmark(model, encoded_input, seed, nb_pass, model_id):
    elapsed_time = []
    for _ in range(nb_pass):
        start = time.time()
        torch.manual_seed(seed)
        model_output = model(**encoded_input)
        duration = time.time() - start
        elapsed_time.append(duration)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        if model_id != 'chin':
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        print(sentence_embeddings[0][:10])
        
    return elapsed_time


def prepare_jit_inputs(model_id):
    
    input_ids_example = torch.randint(6000, size=MODEL_INPUT_SIZE['input_ids'])
    attention_mask_example = torch.randint(1, size=MODEL_INPUT_SIZE['attention_mask'])
        
    if model_id == 'mpnet':    
        example_inputs = {"input_ids": input_ids_example,
                              "attention_mask": attention_mask_example } 
    else:        
        token_type_ids_example = torch.randint(1, size=MODEL_INPUT_SIZE['token_type_ids'])
        example_inputs = {"input_ids": input_ids_example,
                              "token_type_ids": token_type_ids_example,
                              "attention_mask": attention_mask_example } 
    return example_inputs


def apply_jit_trace(model, model_id):
    
    logging.info("using jit trace for acceleration...")
    example_inputs = prepare_jit_inputs(model_id)
    
    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
        model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False)
    
    model = torch.jit.freeze(model.eval())
    
    model(**example_inputs)
    model(**example_inputs)
    
    return model 

def optimize_with_ipex(model):
    
    logging.info("using ipex optimize for acceleration...")
    model = ipex.optimize(model, dtype=torch.bfloat16)

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
    
    if model_id == 'chin':
        sentences = CHI_SENTENCES
    else:
        sentences = SENTENCES
    
    model, tokenizer = load_model(model_id, SEED, model_dtype)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    if use_ipex_optimize:
        model = optimize_with_ipex(model)
    if use_jit:
        model = apply_jit_trace(model, model_id)
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
    