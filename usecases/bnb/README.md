## Bitsandbytes

## Envs Setup
### Install BNB

Install bitsandbytes in your working directory:
```bash
$ git clone --branch multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git
$ cd bitsandbytes/
$ pip install -r requirements-dev.txt
$ pip install .
```
Please make sure `transformers >= 4.45.0`

## use case
Go to `tests/workloads` directory.
### Inference
The following commands defaultly use CPU, please add flag: `--device xpu --model_dtype float16` if you use XPU.
#### bf16(baseline)
```bash
$ bash ./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16
```
#### int8
```bash
$ bash ./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --quant_algo bitsandbytes --quant_dtype int8
```
#### nf4
```bash
$ bash ./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --quant_algo bitsandbytes --quant_dtype nf4
```
#### fp4
```bash
$ bash ./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --quant_algo bitsandbytes --quant_dtype fp4
```

### Finetune
The following commands defaultly use CPU, please add flag: `--device xpu` if you use XPU.
#### bf16 LoRA(baseline)
```bash
$ bash ./run.sh -t fine-tune
```
#### int8 LoRA
```bash
$ bash ./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --quant_algo bitsandbytes --quant_dtype int8
```
#### nf4 QLoRA
```bash
$ bash ./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --quant_algo bitsandbytes --quant_dtype nf4
```
#### fp4 QLoRA
```bash
$ bash ./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --quant_algo bitsandbytes --quant_dtype fp4
```
