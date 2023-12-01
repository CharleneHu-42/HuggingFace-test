# TEST GUIDE

## Model
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

## Test commandline
```bash 
sh run.sh --task task_name --model model_name
```
note: `task_name` should be the same with folders name. For example: `text-generation`

To accelerate inference with bfloat16, ipex_optimize and jit, use the following command
```
sh run.sh --task task_name --model model_name --bf16 --ipex_optimize --jit
```

For more options, run 
```bash
sh run.sh -h 
```

## Test data
For text prompts and speech demos can be found [here](https://drive.google.com/drive/folders/1PbGjFGuPgSxTqK3tC1UKP7sF0cib1pyd?usp=drive_link)
