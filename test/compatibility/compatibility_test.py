import argparse
import csv
import json
import logging
import os

import testcase

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    level=logging.INFO,
    datefmt="[%X]",
)
logger = logging.getLogger(__name__)


def init_arguments_parser():
    parser = argparse.ArgumentParser(description="compatiblity test")
    parser.add_argument(
        "-o", "--output", default="/tmp", help="Assign output folder, default is /tmp"
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0")
    parser.add_argument(
        "-c",
        "--case",
        default="text-classification-mrpc",
        choices=["text-classification-mrpc", "text-classification-sst2"],
        help="choice test case",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="bert-base-cased",
        help="Assign model name, default is bert-base-cased",
    )
    parser.add_argument(
        "--finetune_dir",
        default="/home/wangyi/project/hugface/transformers/examples/pytorch/text-classification",
        help="huggingface finetune examples dir",
    )
    parser.add_argument(
        "--optimum_dir",
        default="/home/wangyi/project/hugface/optimum-intel/examples/neural_compressor/text-classification",
        help="Huggingface optimum-intel examples dir",
    )
    return parser


def get_finetune_case_command(arg: argparse.Namespace):
    switch = {
        "text-classification-mrpc": "run_glue.py --task_name mrpc --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --no_cuda --num_train_epochs 1 --overwrite_output_dir",
        "text-classification-sst2": "run_glue.py --task_name sst2 --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --no_cuda --num_train_epochs 1 --overwrite_output_dir",
    }
    return switch[arg.case]


def get_optimum_case_command(arg: argparse.Namespace):
    switch = {
        "text-classification-mrpc": "run_glue.py --task_name mrpc --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --no_cuda --num_train_epochs 1 --overwrite_output_dir --verify_loading",
        "text-classification-sst2": "run_glue.py --task_name sst2 --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --no_cuda --num_train_epochs 1 --overwrite_output_dir --verify_loading",
    }
    return switch[arg.case]


def main():
    parser = init_arguments_parser()
    args = parser.parse_args()
    f = open(args.output + "/summery.csv", mode="w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(
        f,
        fieldnames=[
            "casename",
            "optimum-intel",
            "subname",
            "eval_f1",
            "eval_samples_per_second",
            "train_samples_per_second",
            "train_loss",
        ],
    )
    csv_writer.writeheader()
    finetune_base_cmd = get_finetune_case_command(args)
    optimum_base_cmd = get_optimum_case_command(args)
    testcase.run_ipex_bf16_finetune_evaluate(
        args.case, finetune_base_cmd, args, csv_writer
    )
    testcase.run_ipex_fp32_finetune_evaluate(
        args.case, finetune_base_cmd, args, csv_writer
    )
    testcase.run_torch_bf16_finetune_evaluate(
        args.case, finetune_base_cmd, args, csv_writer
    )
    testcase.run_torch_fp32_finetune_evaluate(
        args.case, finetune_base_cmd, args, csv_writer
    )

    testcase.run_ipex_bf16_finetune_quantization(
        args.case, finetune_base_cmd, optimum_base_cmd, args, csv_writer
    )
    testcase.run_ipex_fp32_finetune_quantization(
        args.case, finetune_base_cmd, optimum_base_cmd, args, csv_writer
    )
    testcase.run_torch_bf16_finetune_quantization(
        args.case, finetune_base_cmd, optimum_base_cmd, args, csv_writer
    )
    testcase.run_torch_fp32_finetune_quantization(
        args.case, finetune_base_cmd, optimum_base_cmd, args, csv_writer
    )


if __name__ == "__main__":
    main()
