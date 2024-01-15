"""
This script is adapted from the following official alpaca-loca fine-tuning code:
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
"""

import os
from typing import List
import sys
import fire
import torch
import transformers

from datasets import load_dataset
import time

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from utils import Prompter
import logging

logging.basicConfig(level=logging.INFO)


def get_dataset_and_collator(
    data_path,
    prompt_template_name,
    base_model,
    cutoff_len,
    add_eos_token,
    train_on_inputs,
    split_ratio,
):
    data = load_dataset(data_path)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), prompt_template_name)
    prompter = Prompter(template_path)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    train_val = data["train"].train_test_split(
        train_size=split_ratio, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

    collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    return train_data, val_data, collator


def get_lora_model(base_model, lora_r, lora_alpha, lora_target_modules, lora_dropout):
    model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def set_wandb_env():
    os.environ["WANDB_PROJECT"] = "lora-fine-tune"
    os.environ["WANDB_LOG_MODEL"] = "all"


def train(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    max_steps: int = 200,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    split_ratio: float = 0.8,
    gradient_checkpointing: bool = False,
    compile: bool = False,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    use_wandb: bool = False,
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    **kwargs
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"  base_model: {base_model}\n"
            f"  data_path: {data_path}\n"
            f"  output_dir: {output_dir}\n"
            f"  batch_size: {batch_size}\n"
            f"  micro_batch_size: {micro_batch_size}\n"
            f"  num_epochs: {num_epochs}\n"
            f"  max_steps: {max_steps}\n"
            f"  gradient_checkpointing: {gradient_checkpointing}\n"
            f"  compile: {compile}\n"
            f"  learning_rate: {learning_rate}\n"
            f"  cutoff_len: {cutoff_len}\n"
            f"  split_ratio: {split_ratio}\n"
            f"  lora_r: {lora_r}\n"
            f"  lora_alpha: {lora_alpha}\n"
            f"  lora_dropout: {lora_dropout}\n"
            f"  lora_target_modules: {lora_target_modules}\n"
            f"  train_on_inputs: {train_on_inputs}\n"
            f"  add_eos_token: {add_eos_token}\n"
            f"  group_by_length: {group_by_length}\n"
            f"  prompt template: {prompt_template_name}\n"
            f"  use_wandb: {use_wandb}\n"
        )

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if use_wandb:
        set_wandb_env()

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        gradient_checkpointing=gradient_checkpointing,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else "none",
        **kwargs
    )

    model = get_lora_model(
        base_model, lora_r, lora_alpha, lora_target_modules, lora_dropout
    )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    train_data, val_data, collator = get_dataset_and_collator(
        data_path,
        prompt_template_name,
        base_model,
        cutoff_len,
        add_eos_token,
        train_on_inputs,
        split_ratio,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=collator,
    )

    model.config.use_cache = False
    old_state_dict = model.state_dict

    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if compile and torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    start = time.time()
    trainer.train()
    logging.info(f"training time: {time.time() - start}s")

    # model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
