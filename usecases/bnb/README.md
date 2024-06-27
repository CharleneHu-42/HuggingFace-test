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
Please notice that the original transformers may not work, it depends on the [PR](https://github.com/huggingface/transformers/pull/31098).
You can install the transformers in a proper folder by the following command:
```bash
git clone --branch bnb_cpu https://github.com/jiqing-feng/transformers.git && cd transformers/ && pip install .
```

## use case
Go to tests/workloads directory.
### Inference
#### bf16(baseline)
```
sh run.sh -t text-generation --model_dtype bfloat16
```
#### int8
```
sh run.sh -t text-generation --model_dtype bfloat16 --quant_type int8
```
#### nf4
```
sh run.sh -t text-generation --model_dtype bfloat16 --quant_type nf4
```
#### fp4
```
sh run.sh -t text-generation --model_dtype bfloat16 --quant_type fp4
```

### Finetune
#### bf16 LoRA(baseline)
```
sh run.sh -t fine-tune
```
#### int8 LoRA
```
sh run.sh -t fine-tune --quant_type int8
```
#### nf4 QLoRA
```
sh run.sh -t fine-tune --quant_type nf4
```
#### fp4 QLoRA
```
sh run.sh -t fine-tune --quant_type fp4
```
