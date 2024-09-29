# TRL on XPU&GPU
This repository includes examples and training recipes to fine-tune large language modles using the ORPO and KTO algorithm with TRL.


## ORPO

### Installtion 
To install the necessary dependencies, run the following command:
```bash
pip install -U transformers datasets accelerate peft trl wandb
```

### Usage 
You can run the ORPO training script with custom arguments. Here is an example:
```bash
python run_orpo.py --base_model meta-llama/Meta-Llama-3-8B --model_save_dir OrpoLlama-3-8B --device_map xpu --attn_type eager
```
To find out more options, run:
```bash
python run_orpo.py -h
```

## KTO 



