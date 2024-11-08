## Bitsandbytes

## Envs Setup
### Install BNB
#### CPU
Install bitsandbytes with CPU backend in your working directory:
```bash
git clone --branch multi-backend-refactor https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cpu -S .
make
pip install .
```

#### XPU
<TBF>

### Install HF Transformers
Please make sure `transformers >= 4.45.0`

## use case
Go to tests/workloads directory.
### Inference
#### bf16(baseline)
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16
```
#### int8
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --bitsandbytes int8
```
#### nf4
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --bitsandbytes nf4
```
#### fp4
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --bitsandbytes fp4
```

### Finetune
#### bf16 LoRA(baseline)
```
./run.sh -t fine-tune
```
#### int8 LoRA
```
./run.sh -t fine-tune --bitsandbytes int8
```
#### nf4 QLoRA
```
./run.sh -t fine-tune --bitsandbytes nf4
```
#### fp4 QLoRA
```
./run.sh -t fine-tune --bitsandbytes fp4
```
