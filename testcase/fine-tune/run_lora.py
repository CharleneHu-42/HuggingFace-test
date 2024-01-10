import os
import time
import logging

# os.environ["WANDB_PROJECT"] = "peft_tweets" # log to your project
# os.environ["WANDB_LOG_MODEL"] = "all" # log your models
from copy import deepcopy
from argparse import ArgumentParser
from datasets import load_dataset
import evaluate
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
import torch

logging.basicConfig(level=logging.INFO)
# used to calculate the loss for unbalanced labels
# more details see https://huggingface.co/blog/Lora-for-sequence-classification-with-Roberta-Llama-Mistral
POS_WEIGHT, NEG_WEIGHT = (1.1637114032405993, 0.8766697374481806)
TEST_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with PEFT")
    parser.add_argument(
        "--data_name",
        type=str,
        default="mehdiiraqui/twitter_disaster",
        required=False,
        help="Huggingface dataset name",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mistral-lora-token-classification",
        required=False,
        help="Path to store the fine-tuned model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        required=False,
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Eval batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument("--lora_rank", type=int, default=2, help="Lora rank")
    parser.add_argument("--lora_alpha", type=float, default=16, help="Lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Lora dropout")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices={"lora_only", "none", "all"},
        help="Layers to add learnable bias",
    )

    arguments = parser.parse_args()
    return arguments


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)[
        "precision"
    ]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
    }


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


def get_dataset_and_collator(
    data_name,
    model_checkpoints,
    add_prefix_space=True,
    max_length=512,
    truncation=True,
):
    dataset = load_dataset(data_name)
    data = dataset["train"].train_test_split(train_size=0.8, seed=42)
    data["val"] = data.pop("test")
    data["test"] = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoints, add_prefix_space=add_prefix_space
    )

    tokenizer.pad_token = tokenizer.eos_token

    def _preprocesscing_function(examples):
        return tokenizer(examples["text"], truncation=truncation, max_length=max_length)

    col_to_delete = ["id", "keyword", "location", "text"]
    tokenized_datasets = data.map(_preprocesscing_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(col_to_delete)
    tokenized_datasets = tokenized_datasets.rename_column("target", "label")
    tokenized_datasets.set_format("torch")

    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets, padding_collator


def get_lora_model(
    model_checkpoints, num_labels=2, rank=4, alpha=16, lora_dropout=0.1, bias="none"
):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_checkpoints,
        num_labels=num_labels,
        low_cpu_mem_usage=True,
        offload_folder="offload",
        trust_remote_code=True,
    )
    if (
        model_checkpoints == "mistralai/Mistral-7B-v0.1"
        or model_checkpoints == "meta-llama/Llama-2-7b-hf"
    ):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def get_weighted_trainer(pos_weight, neg_weight):
    class _WeightedBCELossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(
                    [neg_weight, pos_weight], device=labels.device, dtype=logits.dtype
                )
            )
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    return _WeightedBCELossTrainer


def main(args):
    """
    Training function
    """

    dataset, collator = get_dataset_and_collator(
        args.data_name,
        args.model_name,
        max_length=args.max_length,
        add_prefix_space=True,
        truncation=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="none",
        max_grad_norm=0.3,
    )

    model = get_lora_model(
        args.model_name,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    model.config.pad_token_id = model.config.eos_token_id

    weighted_trainer = get_weighted_trainer(POS_WEIGHT, NEG_WEIGHT)

    trainer = weighted_trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(CustomCallback(trainer))

    start = time.time()
    trainer.train()
    logging.info(f"training time: {time.time() - start}s")


if __name__ == "__main__":
    args = get_args()
    main(args)
