# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Adapted from https://github.com/declare-lab/instruct-eval
Helper script to compare TRTLLM and HF models on the MMLU dataset.
Example usage:
    mkdir data; wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
    tar -xf data/mmlu.tar -C data && mv data/data data/mmlu

    python mmlu.py --hf_model_dir <HF model path> --engine_dir <TRTLLM engine path> --test_trt_llm
    python mmlu.py --hf_model_dir <HF model path> --engine_dir <TRTLLM engine path> --test_hf
"""
import argparse
import os
import random
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer,
                          GenerationConfig)
from typing import Optional
import intel_extension_for_pytorch as ipex
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DTYPE_STR_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
RAND_SEED = 1234


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


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    elif model_name == 'GemmaForCausalLM' or model_name == 'RecurrentGemmaForCausalLM':
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file,
                                   padding_side='left',
                                   truncation_side='left',
                                   legacy=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left',
                                legacy=False)

    if model_name == 'QWenForCausalLM' and model_version == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config['pad_token_id']
        end_id = gen_config['eos_token_id']
    elif model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


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
        format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate(args, subject, pipeline, dev_df, test_df, output_len):
    cors = []
    all_probs = []
    for i in range(test_df.shape[0]):
        if i >= args.max_ite:
            break
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while not pipeline.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        pred = pipeline(prompt, output_len)

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def get_tokenizer(ckpt_path, max_seq_len):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class Pipeline:

    def __init__(self, tokenizer, model, model_name, pad_id, end_id,
                 max_attention_window_size, mode):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.pad_id = pad_id
        self.end_id = end_id
        self.max_attention_window_size = max_attention_window_size
        self.mode = mode
        self.input_lens = []
        self.tpot = []

    def __call__(self, prompt, output_len):
        # Run the model in batch size 1 and beam size 1
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        batch_input_ids = [inputs]

        # For multi-choice tasks like MMLU, we don't need to adjust following parameters
        top_k = 1
        top_p = 0.0
        do_sample = False

        input_lengths = [x.size(0) for x in batch_input_ids]
        self.input_lens.append(input_lengths[0])

        with torch.no_grad():
            if self.mode == 'hf':
                # Left padding for HF
                max_length = max(input_lengths)
                paddings = [
                    torch.ones(max_length - l, dtype=torch.int32) * self.pad_id
                    for l in input_lengths
                ]
                batch_input_ids = [
                    torch.cat([pad, x])
                    for x, pad in zip(batch_input_ids, paddings)
                ]
                batch_input_ids = torch.stack(batch_input_ids)
                batch_input_ids = batch_input_ids.xpu()
                with torch.no_grad():
                    # Use default temperature and top_k
                    start = time.time()
                    outputs = self.model.generate(batch_input_ids,
                                                  max_new_tokens=output_len,
                                                  top_k=top_k, top_p=top_p, do_sample=do_sample)
                    end = time.time()
                    self.tpot.append((end-start)/output_len*1000)
                    output_ids = outputs[0, input_lengths[0]:]
            elif self.mode == 'ipex':
                max_length = max(input_lengths)
                paddings = [
                    torch.ones(max_length - l, dtype=torch.int32) * self.pad_id
                    for l in input_lengths
                ]
                batch_input_ids = [
                    torch.cat([pad, x])
                    for x, pad in zip(batch_input_ids, paddings)
                ]
                batch_input_ids = torch.stack(batch_input_ids)
                batch_input_ids = batch_input_ids.xpu()

                # Use default temperature and top_k
                start = time.time()
                outputs = self.model.generate(batch_input_ids,
                                                max_new_tokens=output_len,
                                                top_k=top_k, top_p=top_p, do_sample=do_sample)
                end = time.time()
                self.tpot.append((end-start)/output_len*1000)
                torch.xpu.synchronize() 
                output_ids = outputs[0, input_lengths[0]:]
            elif self.mode == 'tensorrt':
                start = time.time()
                outputs = self.model.generate(
                    batch_input_ids,
                    max_new_tokens=output_len,
                    max_attention_window_size=self.max_attention_window_size,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                end = time.time()
                self.tpot.append((end-start)/output_len*1000)
                torch.cuda.synchronize()
                output_ids = outputs[0, 0, input_lengths[0]:]
            elif self.mode == 'optimum-intel':
                max_length = max(input_lengths)
                paddings = [
                    torch.ones(max_length - l, dtype=torch.int32) * self.pad_id
                    for l in input_lengths
                ]
                batch_input_ids = [
                    torch.cat([pad, x])
                    for x, pad in zip(batch_input_ids, paddings)
                ]
                batch_input_ids = torch.stack(batch_input_ids)
                batch_input_ids = batch_input_ids.to("xpu")
                with torch.no_grad():
                    # Use default temperature and top_k
                    start = time.time()
                    outputs = self.model.generate(batch_input_ids,
                                                    max_new_tokens=output_len,
                                                    top_k=top_k, top_p=top_p, do_sample=do_sample)
                    end = time.time()
                    self.tpot.append((end-start)/output_len*1000)
                    output_ids = outputs[0, input_lengths[0]:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def check_valid_length(self, prompt):
        if isinstance(self.model, nn.Module):
            return True
        return len(self.tokenizer.encode(prompt)) <= self.model.max_input_len

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir", type=str, default=None)
    parser.add_argument("--engine_dir", type=str, default=None)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu",
        help=("Path to the data directory. If not available, "
              "download https://people.eecs.berkeley.edu/~hendrycks/data.tar"),
    )
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
        default="fp16",
    )
    parser.add_argument(
        "--debug_mode",
        default=False,
        action="store_true",
        help="Whether or not to turn on the debug mode",
    )
    parser.add_argument(
        "--hf_device_map_auto",
        action="store_true",
        help=("Use device map 'auto' to load a pretrained HF model. This may "
              "help to test a large model that cannot fit into a singlue GPU."),
    )
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behavior'
    )
    parser.add_argument(
        '--output_len',
        type=int,
        default=2,
        help=
        'The maximum output length'
    )
    parser.add_argument(
        '--tokenizer_dir',
        default=None,
        help='tokenizer path; defaults to hf_model_dir if left unspecified')
    parser.add_argument('--vocab_file')
    parser.add_argument("--test_trt_llm", action="store_true")
    parser.add_argument("--test_hf", action="store_true")
    parser.add_argument("--test_ipex", action="store_true")
    parser.add_argument("--test_optimum", action="store_true")
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--accuracy_threshold', type=float, default=0.3)
    parser.add_argument('--max_ite', type=int, default=10000000)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    runtime_rank = 0

    data_fullpath = os.path.join(args.data_dir, "test")

    subjects = sorted([
        f.split("_test.csv")[0] for f in os.listdir(data_fullpath)
        if "_test.csv" in f
    ])

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in get_subcategories().values()
        for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    model_name = "LlamaForCausalLM"
    model_version = None
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version
    )
    output_len = args.output_len
    
    if args.test_trt_llm:
        assert not args.test_hf, "Cannot test both TRT-LLM and HF"
        from tensorrt_llm.runtime import ModelRunner
        model = ModelRunner.from_dir(args.engine_dir,
                                     rank=runtime_rank,
                                     debug_mode=args.debug_mode)
        pipeline = Pipeline(tokenizer, model, model_name, pad_id, end_id,
                            args.max_attention_window_size, "tensorrt")
    elif args.test_optimum:
        assert not args.test_hf, "Cannot test both TRT-LLM and HF"
        from optimum.intel import IPEXModelForCausalLM
        model = IPEXModelForCausalLM.from_pretrained(args.hf_model_dir, device_map="xpu", torch_dtype=torch.float16, export=True)
        pipeline = Pipeline(tokenizer, model, model_name, pad_id, end_id, 
                            args.max_attention_window_size, "optimum-intel")
    elif args.test_ipex:
        assert not args.test_hf, "Cannot test both TRT-LLM and HF"
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_dir,
            trust_remote_code=True,
            device_map="xpu",
            torch_dtype=DTYPE_STR_MAPPING[args.data_type],
        )
        model = ipex.optimize_transformers(
                    model.eval(), dtype=torch.float16, device="xpu", inplace=True
                )  
        model = model.to(memory_format=torch.channels_last)
        pipeline = Pipeline(tokenizer, model, model_name, pad_id, end_id, 
                            args.max_attention_window_size, "ipex")
    else:
        assert args.test_hf, "Must test either TRT-LLM or HF"
        import intel_extension_for_pytorch
        if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
            auto_model_cls = AutoModelForSeq2SeqLM
        elif model_name == 'ChatGLMForCausalLM' and model_version == 'chatglm':
            auto_model_cls = AutoModel
        else:
            auto_model_cls = AutoModelForCausalLM
        model = auto_model_cls.from_pretrained(
            args.hf_model_dir,
            trust_remote_code=True,
            torch_dtype=DTYPE_STR_MAPPING[args.data_type],
            device_map="xpu"
        )
        if not args.hf_device_map_auto:
            model.xpu()
        if model_name == "qwen":
            model.generation_config = GenerationConfig.from_pretrained(
                args.hf_model_dir, trust_remote_code=True)
        pipeline = Pipeline(tokenizer, model, model_name, pad_id, end_id,
                            args.max_attention_window_size, "hf")


    for subject in tqdm(subjects):
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test",
                                           subject + "_test.csv"),
                              header=None)

        cors, acc, probs = evaluate(args, subject, pipeline, dev_df, test_df, output_len)
        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories().keys():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    if args.check_accuracy:
        assert weighted_acc >= args.accuracy_threshold, f"Expected accuracy >= {args.accuracy_threshold} while got {weighted_acc}"

    input_lens = pipeline.input_lens
    tpot = pipeline.tpot

    print(f"Average Prompt Length: {np.mean(input_lens)}, tpot list size: {len(tpot)}")
    print(f"Average TPOT: {np.mean(tpot)}")

    return weighted_acc


if __name__ == "__main__":
    main()


