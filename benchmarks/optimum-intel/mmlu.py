"""Adapted from https://github.com/declare-lab/instruct-eval
Helper script to benchmark TRTLLM, HF, IPEX and Optimum-Intel models on the MMLU dataset.
Example usage:
    mkdir data; wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
    tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
    python mmlu.py --model_name <HF model path> --device xpu --eval_mode optimum-intel
"""

import argparse
import os
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import set_seed

from benchmark_utils import BenchmarkPipeline

logging.basicConfig(level=logging.INFO)

RAND_SEED = 1234
set_seed(RAND_SEED)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DTYPE_STR_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(get_choices()[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def prepare_prompt(test_df, dev_df, subject, row, ntrain, pipeline):
    k = ntrain
    prompt_end = format_example(test_df, row, include_answer=False)
    train_prompt = gen_prompt(dev_df, subject, k)
    prompt = train_prompt + prompt_end

    while not pipeline.check_valid_length(prompt) and k > 0:
        k -= 1
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

    return prompt


def evaluate(pipeline, subject, batch_size, ntrain, dev_df, test_df, warm_up_samples):
    # warm-up
    for row in range(warm_up_samples):
        prompt = prepare_prompt(test_df, dev_df, subject, row, ntrain, pipeline)
        _ = pipeline([prompt])

    num_samples = test_df.shape[0]
    num_iter = num_samples // batch_size
    all_labels = []
    all_preds = []
    for i in range(num_iter + 1):
        batch_prompt = []
        start_row = i * batch_size
        if start_row >= num_samples:
            break
        end_row = (i + 1) * batch_size
        for row in range(start_row, end_row):
            if row >= num_samples:
                break
            prompt = prepare_prompt(test_df, dev_df, subject, row, ntrain, pipeline)
            batch_prompt.append(prompt)
        batch_labels = test_df.iloc[start_row:end_row, test_df.shape[1] - 1].to_list()
        batch_preds = pipeline(batch_prompt)
        all_labels.append(batch_labels)
        all_preds.append(batch_preds)

    all_labels = [label for batch_labels in all_labels for label in batch_labels]
    all_preds = [pred for batch_preds in all_preds for pred in batch_preds]
    cors = [
        pred.strip().startswith(label) for pred, label in zip(all_preds, all_labels)
    ]

    acc = np.mean(cors)
    cors = np.array(cors)

    logging.info("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=None, help="huggingface model name"
    )
    parser.add_argument("--engine_dir", type=str, default=None, help="trt-llm only")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="directory to save the benchmark result",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu",
        help=(
            "Path to the data directory. If not available, "
            "download https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        ),
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5,
        help="number of examples to be included in the prompt",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
        default="fp16",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "xpu"],
        default="xpu",
    )
    parser.add_argument("--warm_up_samples", type=int, default=10)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["trt-llm", "optimum-intel", "hf", "ipex"],
        default="hf",
    )
    parser.add_argument("--check_accuracy", action="store_true")
    parser.add_argument("--accuracy_threshold", type=float, default=0.3)

    args = parser.parse_args()
    return args


def main(args):
    pipeline = BenchmarkPipeline(args)

    data_fullpath = os.path.join(args.data_dir, "test")
    subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(data_fullpath) if "_test.csv" in f]
    )

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in get_subcategories().values()
        for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        cors, _ = evaluate(
            pipeline,
            subject,
            args.batch_size,
            args.ntrain,
            dev_df,
            test_df,
            args.warm_up_samples,
        )
        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories().keys():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        logging.info("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        logging.info("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    logging.info("Average accuracy: {:.3f}".format(weighted_acc))
    if args.check_accuracy:
        assert (
            weighted_acc >= args.accuracy_threshold
        ), f"Expected accuracy >= {args.accuracy_threshold} while got {weighted_acc}"

    pipeline.report(args.save_dir, weighted_acc)

    return weighted_acc


if __name__ == "__main__":
    args = parse_args()
    main(args)
