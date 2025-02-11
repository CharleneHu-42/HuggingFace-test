## GPTQModel

## Envs Setup
### Install GPTQModel
#### CPU
Install GPTQModel with CPU backend in your working directory:
```bash
$ pip install intel_extension_for_pytorch optimum
$ pip install -v --no-build-isolation gptqmodel[ipex]
```

## use case
Go to tests/workloads directory.
### Inference
The following commands defaultly use CPU, please add flag: `--device xpu --model_dtype float16` if you use XPU.
#### bf16(baseline)
```bash
$ bash ./run.sh -t text-generation -m meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16
```
#### Int4
```bash
$ bash ./run.sh -t text-generation -m hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --model_dtype bfloat16
```

### Finetune
The following commands defaultly use CPU, please add flag: `--device xpu` if you use XPU.
#### bf16 LoRA(baseline)
```bash
$ bash ./run.sh -t llm-lora -m meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16
```
#### Int4 LoRA
```bash
$ bash ./run.sh -t llm-lora -m hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --model_dtype bfloat16
```
