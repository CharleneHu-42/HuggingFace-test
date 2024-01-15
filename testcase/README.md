# TEST GUIDE

## Envs on CPU
Please use conda env.
We recommend you to use the preview version of [pytorch](https://pytorch.org/get-started/locally/), and build [ipex](https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu) from source by (use gcc-11):
```bash
git clone --recursive https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu.git
git submodule sync && git submodule update --init --recursive
source /opt/rh/gcc-toolset-11/enable
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
cd frameworks.ai.pytorch.ipex-cpu/
python setup.py develop
```

## Model
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

## Test commandline
### All tasks

```bash 
sh run.sh --task task_name --model_id model_name
```
note: `task_name` should be the same with folders name. For example: `text-generation`

To accelerate inference with bfloat16, ipex_optimize and jit, use the following command
```
sh run.sh --task task_name --model_id model_name --model_dtype bfloat16 --compute_dtype bfloat16 --ipex_optimize True --jit True
```
**___Note: The ipex and jit optimizations may failed in some tasks.___**

For text-genetation task: You can control the input and output and use greedy search by add the follwing flags:
```
--batch_size 1 --num_beams 1 --input_tokens 1024 --output_tokens 32
```
**___Note: Default values are batch_size=1, num_beams=4, input_tokens=32, output_tokens=32.___**
You can also use ipex optimize transformers by adding the flag `--ipex_optimize_transformers True`, but it doesn't work for now.

For more options, run 
```bash
sh run.sh -h 
```

you could also run the all_task using
```bash
sh run_all_task.sh -h
```
To accelerate all task with bfloat16, ipex_optimize and jit, use the following command
```
sh run_all_task.sh --compute_dtype bfloat16 --model_dtype bfloat16 --ipex_optimize True --jit True
```

To run the test scripts on XPU, please activate the oneAPI environment, e.g. 
```bash
source env.sh
sh run.sh --task task_name --model_id model_name --device xpu
```

### Finetune
We defaultly use `fp16` and `ipex` on XPU and GPU, `bf16` on CPU. If you want to run finetune separately and use `ipex` on CPU, please run the follwing script:
```
sh run.sh -t fine-tune --ipex_optimize True
``` 
You can also add `--gradient_checkpointing` to use gradient checkpointing.
**___Note: IPEX has bug with bf16 on the newest version(2.3).___**

## Test data
For text prompts and speech demos can be found [here](https://drive.google.com/drive/folders/1PbGjFGuPgSxTqK3tC1UKP7sF0cib1pyd?usp=drive_link)

