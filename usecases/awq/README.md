# AutoAWQ

## Envs Setup

### Install AWQ

```bash
$ pip install git+https://github.com/casper-hansen/AutoAWQ.git
```

## use case
Go to `tests/workloads` directory.

### Inference
The following commands defaultly use CPU, please add flag: `--device xpu --model_dtype float16` if you use XPU.

#### bf16(baseline)

```bash
$ bash ./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16
```
#### int4
```bash
$ bash ./run.sh -t text-generation -m TheBloke/firefly-llama2-7B-chat-AWQ --model_dtype bfloat16 --quant_algo autoawq --quant_dtype int4
```

### Finetune

The following commands defaultly use CPU, please add flag: `--device xpu` if you use XPU.
#### bf16 LoRA(baseline)

```bash
bash ./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf
```
#### int4 LoRA
```bash
$ bash ./run.sh -t fine-tune -m TheBloke/firefly-llama2-7B-chat-AWQ --quant_algo autoawq --quant_dtype int4
```
