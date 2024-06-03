## Bitsandbytes

## Envs
Install bitsandbytes with CPU backend in a proper folder:
```bash
git clone --branch multi-backend-refactor https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cpu -S .
make
pip install .
```

Please notice that the original transformers may not work, it depends on the [PR](https://github.com/huggingface/transformers/pull/31098).
You can install the transformers in a proper folder by the following command:
```bash
git clone --branch bnb_cpu https://github.com/jiqing-feng/transformers.git && cd bitsandbytes/ && pip install .
```

Go to the tests/workloads folder.
## Inference
Running inference by `sh run.sh -t text-generation --model_dtype bfloat16` and you can add the flag `--quant_type` to set quantization type, including `int8`, `nf4` and `fp4`, for example:
Run int8 inference: `sh run.sh -t text-generation --model_dtype bfloat16 --quant_type int8`
Run nf4 inference: `sh run.sh -t text-generation --model_dtype bfloat16 --quant_type nf4`

## Finetune
Running lora finetune by `sh run.sh -t fine-tune`, also use `--quant_type` to set quantization type, for example:
Run int8 lora: `sh run.sh -t fine-tune --quant_type int8`
