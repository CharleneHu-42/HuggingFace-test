# TEST GUIDE

## Inference

### Intel Native Experience 
#### CPU

#### XPU
For XPU, you first need to activate the oneAPI environment and then run the test script:
```bash
source env.sh
./run_all_task_xpu.sh --model_dtype float16 
```
If you want to compare the performance with NV GPU, just add the flag `--device cuda` to the command above.  


### Intel Intermediate Expeirnece 
#### CPU

#### XPU
```bash 
source env.sh
./run_all_task_xpu.sh --model_dtype float16 --ipex_optimize True
```


## Fine-tune
### CPU

### XPU 
```bash
./run.sh --task fine-tune --device xpu
```


## Notes
### Connection Error
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

### Command Usage 
To run individual task:
```bash 
sh run.sh --task task_name --model_id model_name
```
note: `task_name` should be the same with folders name. For example: `text-generation`

To accelerate inference with bfloat16, ipex_optimize and jit, use the following command
```
sh run.sh --task task_name --model_id model_name --model_dtype bfloat16 --compute_dtype bfloat16 --ipex_optimize True --jit True
```
**___Note: The ipex and jit optimizations may failed in some tasks.___**


### Text-Generation
For text-genetation task, you can control the input and output and use greedy search by add the follwing flags:
```
--batch_size 1 --num_beams 1 --input_tokens 1024 --output_tokens 32
```
**___Note: Default values are batch_size=1, num_beams=4, input_tokens=32, output_tokens=32.___**
You can also use ipex optimize transformers by adding the flag `--ipex_optimize_transformers True`, but it doesn't work for now.

### Finetune
We use `accelerate launch` to start finetune on XPU and GPU, use `mpirun python` on CPU to enable distributed finetune(need to install one-ccl on CPU). 
The finetune task defaultly use `fp16` and `ipex` on XPU and GPU, `bf16` on CPU. If you want to run finetune separately and use `ipex` on CPU, please run the follwing script:
```
sh run.sh --task_name fine-tune --ipex_optimize True --device cpu 
``` 
You can also add `--gradient_checkpointing True` to use gradient checkpointing.


### Test Data 
For text prompts and speech demos can be found [here](https://drive.google.com/drive/folders/1PbGjFGuPgSxTqK3tC1UKP7sF0cib1pyd?usp=drive_link)



