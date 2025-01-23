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
$ bash ./run.sh -t text-generation -m meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16
```
#### int4
```bash
$ bash ./run.sh -t text-generation -m hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --model_dtype bfloat16 --quant_algo autoawq --quant_dtype int4
```

### Finetune

The following commands defaultly use CPU, please add flag: `--device xpu` if you use XPU.
#### bf16 LoRA(baseline)

```bash
$ bash ./run.sh -t llm-lora -m meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16
```
#### int4 LoRA
```bash
$ bash ./run.sh -t llm-lora -m hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --model_dtype bfloat16
```
