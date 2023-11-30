# TEST GUIDE

## Model
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

## Test commandline
```
sh run.sh task_name model_name
```
note: `task_name` should be the same with folders name. For example: `text-generation`


For some models that can be accelerated with bfloat16, ipex_optimize and jit, use the following command
```
# use bf16
sh run.sh task_name model_name bf16 
# use bf16 + ipex_optimize
sh run.sh task_name model_name bf16 ipex_opt 
# use bf16 + ipex_optimize + jit
sh run.sh task_name model_name bf16 ipex_opt jit
```

## Test data
For text prompts and speech demos can be found [here](https://drive.google.com/drive/folders/1PbGjFGuPgSxTqK3tC1UKP7sF0cib1pyd?usp=drive_link)
