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
    a["subname"] = subname + "-finetune-" + inf_subname + "-inf"
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
    output_dir = args.output + "/" + case_name + "-" + subname + "/" + qt_subname
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
    if "static-ptq" in qt_subname:
        cmdstr += " --apply_quantization --quantization_approach static"
    if "dyn-ptq" in qt_subname:
        cmdstr += " --apply_quantization --quantization_approach dynamic"
    if "qat" in qt_subname:
        cmdstr += " --apply_quantization --quantization_approach aware_training"
    if "ipex" in qt_subname:
        cmdstr += " --backend ipex"

    cmdstr = "numactl --cpunodebind 0 --membind 0 python3 " + cmdstr
    logger.info(f"{cmdstr}")
    exit, output = subprocess.getstatusoutput(cmdstr)
    if exit != 0:
        logger.error(f"{output}")
    return


def run_optimum_intel_quantization_deploy(
    case_name: str,
    subname: str,
    qt_subname: str,
    deploy_subname: str,
    basecmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    os.chdir(args.optimum_dir)
    model_dir = args.output + "/" + case_name + "-" + subname
    loading_dir = args.output + "/" + case_name + "-" + subname + "/" + qt_subname
    output_dir = (
        args.output
        + "/"
        + case_name
        + "-"
        + subname
        + "/"
        + qt_subname
        + "/"
        + deploy_subname
    )
    cmdstr = (
        basecmd
        + " --output_dir "
        + output_dir
        + " --model_name_or_path "
        + model_dir
        + " --do_eval"
        + " --loading_dir "
        + loading_dir
    )
    jit = False
    bf16 = False
    ipex = False
    if "bf16" in deploy_subname:
        cmdstr += " --bf16"
        bf16 = True
    if "ipex" in deploy_subname:
        cmdstr += " --use_ipex"
        ipex = True
    if "jit" in deploy_subname:
        cmdstr += " --jit_mode_eval"
        jit = True
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
        "jit_mode": jit,
        "bf16": bf16,
        "use_ipex": ipex,
        "model": args.model,
    }
    a["casename"] = case_name
    a["subname"] = subname + "-finetune-" + qt_subname + "-" + deploy_subname + "-inf"
    if exit == 0:
        results = get_json_map(output_dir + "/eval_results.json")
        if results:
            a["eval_f1"] = results["eval_f1"]
            a["eval_samples_per_second"] = results["eval_samples_per_second"]
    else:
        logger.error(f"{output}")
    csv_writer.writerow(a)
    return


def run_finetune_evaluation(
    finetune_case_name: str,
    case_name: str,
    basecmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    infer_cases = [
        "ipex-eager-bf16",
        "ipex-eager-fp32",
        "pt-eager-fp32",
        "pt-eager-bf16",
        "ipex-jit-bf16",
        "ipex-jit-fp32",
        "pt-jit-fp32",
        "pt-jit-bf16",
    ]
    # step1: finetune to get the model firstly
    run_finetune(case_name, finetune_case_name, basecmd, args, csv_writer)
    for infer_case in infer_cases:
        # step2: run bf16 or fp32 inference on the finetune output model
        run_inference(
            case_name, finetune_case_name, infer_case, basecmd, args, csv_writer
        )


def run_ipex_bf16_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    finetune_case_name = "0-ipex-bf16"
    run_finetune_evaluation(finetune_case_name, case_name, basecmd, args, csv_writer)
    return


def run_ipex_fp32_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    finetune_case_name = "0-ipex-fp32"
    run_finetune_evaluation(finetune_case_name, case_name, basecmd, args, csv_writer)
    return


def run_pt_bf16_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    finetune_case_name = "0-pt-bf16"
    run_finetune_evaluation(finetune_case_name, case_name, basecmd, args, csv_writer)
    return


def run_pt_fp32_finetune_evaluate(
    case_name: str, basecmd: str, args: argparse.Namespace, csv_writer: csv.DictWriter
):
    finetune_case_name = "0-pt-fp32"
    run_finetune_evaluation(finetune_case_name, case_name, basecmd, args, csv_writer)
    return


def get_infer_cases(qt_case: str):
    infer_cases = []
    if "pt-fx-fp32int8" in qt_case:
        infer_cases = [
            "ipex-eager-fp32int8",
            "pt-eager-fp32int8",
            #    "ipex-jit-fp32int8", #no-work
            #    "pt-jit-fp32int8",   #no-work
        ]
    elif "ipex-fp32int8" in qt_case:
        infer_cases = [
            "ipex-eager-fp32int8",
            "pt-eager-fp32int8",
        ]
    elif "ipex-bf16int8" in qt_case:
        infer_cases = [
            "ipex-eager-fp32int8",
            "pt-eager-fp32int8",
            "ipex-eager-bf16int8",
            "pt-eager-bf16int8",
        ]
    if "pt-eager-fp32int8" in qt_case:
        infer_cases = [
            "ipex-eager-fp32int8",
            "pt-eager-fp32int8",
            "ipex-jit-fp32int8",
            "pt-jit-fp32int8",
        ]
    return infer_cases


def run_finetune_quantization_deploy(
    finetune_case_name: str,
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    deploy_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    qt_cases = [
        "pt-fx-fp32int8-static-ptq",
        "pt-eager-fp32int8-dyn-ptq",
        "pt-fx-fp32int8-qat",
        "ipex-fp32int8-static-ptq",
    ]
    # step1: finetune to get the model firstly
    run_finetune(case_name, finetune_case_name, finetune_cmd, args, csv_writer)
    for qt_case in qt_cases:
        # step2: use optimum-intel to quantitize the model
        run_optimum_intel_quantization(
            case_name, finetune_case_name, qt_case, optimum_cmd, args, csv_writer
        )
        infer_cases = get_infer_cases(qt_case)
        for infer_case in infer_cases:
            # step3: use optimum-intel to deploy the quantized model
            run_optimum_intel_quantization_deploy(
                case_name,
                finetune_case_name,
                qt_case,
                infer_case,
                deploy_cmd,
                args,
                csv_writer,
            )
    return


def run_ipex_bf16_finetune_quantization_deploy(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    deploy_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    finetune_case_name = "1-ipex-bf16"
    run_finetune_quantization_deploy(
        finetune_case_name,
        case_name,
        finetune_cmd,
        optimum_cmd,
        deploy_cmd,
        args,
        csv_writer,
    )


def run_ipex_fp32_finetune_quantization_deploy(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    deploy_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    finetune_case_name = "1-ipex-fp32"
    run_finetune_quantization_deploy(
        finetune_case_name,
        case_name,
        finetune_cmd,
        optimum_cmd,
        deploy_cmd,
        args,
        csv_writer,
    )


def run_pt_bf16_finetune_quantization_deploy(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    deploy_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    finetune_case_name = "1-pt-bf16"
    run_finetune_quantization_deploy(
        finetune_case_name,
        case_name,
        finetune_cmd,
        optimum_cmd,
        deploy_cmd,
        args,
        csv_writer,
    )
    return


def run_pt_fp32_finetune_quantization_deploy(
    case_name: str,
    finetune_cmd: str,
    optimum_cmd: str,
    deploy_cmd: str,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter,
):
    finetune_case_name = "1-pt-fp32"
    run_finetune_quantization_deploy(
        finetune_case_name,
        case_name,
        finetune_cmd,
        optimum_cmd,
        deploy_cmd,
        args,
        csv_writer,
    )
