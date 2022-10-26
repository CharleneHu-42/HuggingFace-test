import argparse
import copy
import csv
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def get_json_map(path: str):
    import json

    results = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results


def run_finetune(
    case_name: str,
    subname: str,
    basecmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    import psutil
    os.chdir(args.finetune_dir)
    output_dir = args.output + "/" + case_name + "-" + subname
    cmdstr = (
        basecmd
        + " --output_dir "
        + output_dir
        + " --model_name_or_path "
        + args.model
        + " --do_eval --do_train"
    )
    bf16 = False
    ipex = False
    if "bf16" in subname:
        cmdstr += " --bf16"
        bf16 = True
    if "ipex" in subname:
        cmdstr += " --use_ipex"
        ipex = True
    cmdstr = "numactl --cpunodebind 0 --membind 0 python3 " + cmdstr
    logger.info(f"{cmdstr}")
    exit, output = subprocess.getstatusoutput(cmdstr)
    a = {
        "optimum-intel": "wo",
        "casename": None,
        "subname": None,
        "eval_f1": None,
        "eval_samples_per_second": None,
        "train_samples_per_second": None,
        "train_loss": None,
        "lcores": int(psutil.cpu_count(logical=True) / 2),
        "jit_mode": False,
        "bf16": bf16,
        "use_ipex": ipex,
        "model": args.model,
    }
    a["casename"] = case_name
    a["subname"] = subname + "-finetune"
    if exit == 0:
        results = get_json_map(output_dir + "/all_results.json")
        a["eval_f1"] = results["eval_f1"]
        a["eval_samples_per_second"] = results["eval_samples_per_second"]
        a["train_loss"] = results["train_loss"]
        a["train_samples_per_second"] = results["train_samples_per_second"]
    else:
        logger.error(f"{output}")
    csv_writer.writerow(a)
    return


def run_inference(
    case_name: str,
    subname: str,
    inf_subname: str,
    basecmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    os.chdir(args.finetune_dir)
    model_dir = args.output + "/" + case_name + "-" + subname
    output_dir = args.output + "/" + case_name + "-" + subname + "/" + inf_subname
    cmdstr = (
        basecmd
        + " --output_dir "
        + output_dir
        + " --model_name_or_path "
        + model_dir
        + " --do_eval"
    )
    jit = False
    bf16 = False
    ipex = False
    if "bf16" in inf_subname:
        cmdstr += " --bf16"
        bf16 = True
    if "ipex" in inf_subname:
        cmdstr += " --use_ipex"
        ipex = True
    if "jit" in inf_subname:
        cmdstr += " --jit_mode_eval"
        jit = True
    cmdstr = "taskset -c 0-3 python3 " + cmdstr
    logger.info(f"{cmdstr}")
    exit, output = subprocess.getstatusoutput(cmdstr)
    a = {
        "optimum-intel": "wo",
        "casename": None,
        "subname": None,
        "eval_f1": None,
        "eval_samples_per_second": None,
        "train_samples_per_second": None,
        "train_loss": None,
        "lcores": 4,
        "jit_mode": jit,
        "bf16": bf16,
        "use_ipex": ipex,
        "model": args.model,
    }
    a["casename"] = case_name
    a["subname"] = subname + "-finetune-model-inf-only-" + inf_subname
    if exit == 0:
        results = get_json_map(output_dir + "/eval_results.json")
        a["eval_f1"] = results["eval_f1"]
        a["eval_samples_per_second"] = results["eval_samples_per_second"]
    else:
        logger.error(f"{output}")
    csv_writer.writerow(a)
    return


def run_optimum_intel_quantization(
    case_name: str,
    subname: str,
    qt_subname: str,
    basecmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    os.chdir(args.optimum_dir)
    model_dir = args.output + "/" + case_name + "-" + subname
    output_dir = args.output + "/" + case_name + "-" + subname + "/ptq-" + qt_subname
    cmdstr = (
        basecmd
        + " --output_dir "
        + output_dir
        + " --model_name_or_path "
        + model_dir
        + " --do_eval --do_train"
    )
    if "bf16" in qt_subname:
        cmdstr += " --bf16"
    if "ipex" in qt_subname:
        cmdstr += " --use_ipex"
    if "static" in qt_subname:
        cmdstr += " --apply_quantization --quantization_approach static"
    if "dynamic" in qt_subname:
        cmdstr += " --apply_quantization --quantization_approach dynamic"
    if "aware_training" in qt_subname:
        cmdstr += " --apply_quantization --quantization_approach aware_training"

    cmdstr = "taskset -c 0-3 python3 " + cmdstr
    logger.info(f"{cmdstr}")
    exit, output = subprocess.getstatusoutput(cmdstr)
    a = {
        "optimum-intel": "w",
        "casename": None,
        "subname": None,
        "eval_f1": None,
        "eval_samples_per_second": None,
        "train_samples_per_second": None,
        "train_loss": None,
        "lcores": 4,
        "jit_mode": False,
        "bf16": False,
        "use_ipex": False,
        "model": args.model,
    }
    a["casename"] = case_name
    a["subname"] = subname + "-finetune-model-only-qt-" + qt_subname
    if exit == 0:
        results = get_json_map(output_dir + "/eval_results.json")
        if results:
            a["eval_f1"] = results["eval_f1"]
            a["eval_samples_per_second"] = results["eval_samples_per_second"]
    else:
        logger.error(f"{output}")
    csv_writer.writerow(a)
    return


def run_ipex_bf16_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "ipex-bf16", basecmd, args, csv_writer)
    # step2: run bf16 or fp32 inference on the finetune output model
    run_inference(case_name, "ipex-bf16", "ipex-bf16", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-bf16", "ipex-fp32", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-bf16", "fp32", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-bf16", "bf16", basecmd, args, csv_writer)
    # step3: run bf16 or fp32 inference on the finetune output model using jit
    run_inference(case_name, "ipex-bf16", "ipex-bf16-jit", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-bf16", "ipex-fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-bf16", "fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-bf16", "bf16-jit", basecmd, args, csv_writer)
    return


def run_ipex_fp32_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "ipex-fp32", basecmd, args, csv_writer)
    # step2: run bf16 or fp32 inference on the finetune output model
    run_inference(case_name, "ipex-fp32", "ipex-bf16", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-fp32", "ipex-fp32", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-fp32", "fp32", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-fp32", "bf16", basecmd, args, csv_writer)
    # step3: run bf16 or fp32 inference on the finetune output model using jit
    run_inference(case_name, "ipex-fp32", "ipex-bf16-jit", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-fp32", "ipex-fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-fp32", "fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "ipex-fp32", "bf16-jit", basecmd, args, csv_writer)
    return


def run_torch_bf16_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "bf16", basecmd, args, csv_writer)
    # step2: run bf16 or fp32 inference on the finetune output model
    run_inference(case_name, "bf16", "ipex-bf16", basecmd, args, csv_writer)
    run_inference(case_name, "bf16", "ipex-fp32", basecmd, args, csv_writer)
    run_inference(case_name, "bf16", "fp32", basecmd, args, csv_writer)
    run_inference(case_name, "bf16", "bf16", basecmd, args, csv_writer)
    # step3: run bf16 or fp32 inference on the finetune output model using jit
    run_inference(case_name, "bf16", "ipex-bf16-jit", basecmd, args, csv_writer)
    run_inference(case_name, "bf16", "ipex-fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "bf16", "fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "bf16", "bf16-jit", basecmd, args, csv_writer)
    return


def run_torch_fp32_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "fp32", basecmd, args, csv_writer)
    # step2: run bf16 or fp32 inference on the finetune output model
    run_inference(case_name, "fp32", "ipex-bf16", basecmd, args, csv_writer)
    run_inference(case_name, "fp32", "ipex-fp32", basecmd, args, csv_writer)
    run_inference(case_name, "fp32", "fp32", basecmd, args, csv_writer)
    run_inference(case_name, "fp32", "bf16", basecmd, args, csv_writer)
    # step3: run bf16 or fp32 inference on the finetune output model using jit
    run_inference(case_name, "fp32", "ipex-bf16-jit", basecmd, args, csv_writer)
    run_inference(case_name, "fp32", "ipex-fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "fp32", "fp32-jit", basecmd, args, csv_writer)
    run_inference(case_name, "fp32", "bf16-jit", basecmd, args, csv_writer)
    return


def run_ipex_bf16_finetune_quantization(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "ipex-bf16-1", finetune_cmd, args, csv_writer)
    # step2: quantization using optimum-intel
    run_optimum_intel_quantization(
        case_name, "ipex-bf16-1", "static", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "ipex-bf16-1", "dynamic", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "ipex-bf16-1", "aware_training", optimum_cmd, args, csv_writer
    )
    return


def run_ipex_fp32_finetune_quantization(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "ipex-fp32-1", finetune_cmd, args, csv_writer)
    # step2: quantization using optimum-intel
    run_optimum_intel_quantization(
        case_name, "ipex-fp32-1", "static", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "ipex-fp32-1", "dynamic", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "ipex-fp32-1", "aware_training", optimum_cmd, args, csv_writer
    )
    return


def run_torch_bf16_finetune_quantization(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "bf16-1", finetune_cmd, args, csv_writer)
    # step2: quantization using optimum-intel
    run_optimum_intel_quantization(
        case_name, "bf16-1", "static", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "bf16-1", "dynamic", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "bf16-1", "aware_training", optimum_cmd, args, csv_writer
    )
    return


def run_torch_fp32_finetune_quantization(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    # step1: finetune to get the model firstly
    run_finetune(case_name, "fp32-1", finetune_cmd, args, csv_writer)
    # step2: quantization using optimum-intel
    run_optimum_intel_quantization(
        case_name, "fp32-1", "static", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "fp32-1", "dynamic", optimum_cmd, args, csv_writer
    )
    run_optimum_intel_quantization(
        case_name, "fp32-1", "aware_training", optimum_cmd, args, csv_writer
    )
    return
