# TEST GUIDE

## Model
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

## Test commandline
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

## Test data
For text prompts and speech demos can be found [here](https://drive.google.com/drive/folders/1PbGjFGuPgSxTqK3tC1UKP7sF0cib1pyd?usp=drive_link)

