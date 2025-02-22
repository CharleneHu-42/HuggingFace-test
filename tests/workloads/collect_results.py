import os
import csv
import pandas as pd
import re
import json
import argparse

parser = argparse.ArgumentParser("Generate csv file of benchmark results based on logs")
parser.add_argument(
    "-l",
    "--log-dir",
    type=str,
    help="log directory path",
)
parser.add_argument(
    "-o",
    "--output-file",
    type=str,
    help="output file path",
)
parser.add_argument(
    "-p",
    "--log-pattern",
    # action='append',
    type=str,
    help="log filename pattern to be processed",
)
args = parser.parse_args()


log_dir = os.path.normpath(args.log_dir)
text_gen = (
    True
    if "text-generation" in log_dir or "summarization" in log_dir or "translation" in log_dir or "autotp" in log_dir
    else False
)
field_names = [
    "model_id",
    "dtype",
    "quant",
    "input/output tokens",
    "batch_size",
    "beam_size",
    "do_sample",
    "1st token latency (ms)",
    "2nd+ token avg latency (ms)",
    # "2nd+ token p90 latency (ms)",
    # "2nd+ token p99 latency (ms)",
    "inference latency (ms)",
    "mem (GB)",
    "pipeline latency (ms)",
    "forward latency (ms)",
] if text_gen else [
    "model_id",
    "dtype",
    "quant",
    "batch_size",
    "mem (GB)",
    "pipeline latency (ms)",
    "model latency (ms)",
    "forward latency (ms)",
]
full_data = {
    "stock_eager": [],
    "stock_compile": [],
    "ipex": [],
    "ipex_compile": [],
    "optimum-intel": []
}
if text_gen:
    log_list = [file for file in os.listdir(log_dir) if re.search(r'\d\.log$', file)]
    # log_list_w_time = [(f, os.path.getmtime(os.path.join(log_dir, f))) for f in os.listdir(log_dir) if re.search(r'\d\.log$', file)]
else:
    log_list = [file for file in os.listdir(log_dir) if file.endswith('.log')]
    # log_list_w_time = [(f, os.path.getmtime(os.path.join(log_dir, f))) for f in os.listdir(log_dir) if file.endswith('.log')]
if args.log_pattern:
    combined_pattern = "|".join(args.log_pattern)
    log_list = [file for file in log_list if re.search(rf"{args.log_pattern}", file)]
print(log_list)
# for log in log_list:
#     print(log)
# print(int(log.split('_')[-2]))
log_files_with_mtime = [(file, os.path.getmtime(os.path.join(log_dir, file))) for file in log_list]
# Sort files by modification time (oldest first)
log_files_with_mtime.sort(key=lambda x: x[1])
log_list = [file for file, _ in log_files_with_mtime]
# if text_gen:
#     log_list = sorted(log_list, key=lambda filename: int(filename.split('_')[-2]))
# log_list = sorted(log_list, key=lambda filename: filename.split('_')[0])
print(log_list)
for log_name in log_list:
    print(f"log: {log_name}")
    log_info = {field: "" for field in field_names}

    fields = log_name.split(".log")[0].split("_")
    log_info["model_id"] = fields[0]
    log_info["dtype"] = fields[1]
    if "bnb" in os.path.join(log_dir, log_name):
        log_info["quant"] = "bnb"
    elif "awq" in os.path.join(log_dir, log_name):
        log_info["quant"] = "awq"
    if "optimum-intel" in log_name:
        impl = "optimum-intel"
        fields = log_name.replace("optimum-intel_", "").split(".log")[0].split("_")
    else:
        if fields[3] == "False":
            impl = "stock_eager" if fields[5] == "False" else "stock_compile"
        else:
            impl = "ipex" if fields[5] == "False" else "ipex_compile"
    if "bs" in log_name:
        log_info["batch_size"] = fields[7]        
    if text_gen:
        log_info["beam_size"] = fields[-5]
        log_info["do_sample"] = fields[-3]
        log_info["input/output tokens"] = fields[-2] + "/" + fields[-1]

    with open(os.path.join(log_dir,log_name), "r") as file:
        for line in file:
            if "1st token" in line or "First token" in line:
                log_info["1st token latency (ms)"] = line.split()[-2]
            if "2nd+ token" in line or "Average 2" in line:
                log_info["2nd+ token avg latency (ms)"] = line.split()[-2]
            if "pipeline average" in line:
                log_info["pipeline latency (ms)"] = line.split()[4].rstrip(",")
                log_info["forward latency (ms)"] = line.split()[-1]
            if "model average" in line:
                log_info["model latency (ms)"] = line.split()[-1]
            if "Inference latency" in line:
                log_info["inference latency (ms)"] = line.split()[-2]
            if "Maximum resident set size" in line:
                log_info["mem (GB)"] = round(float(line.split()[-1])/1024/1024, 2)
    
    if full_data.get(impl) == None: 
        print(f"Implementation {impl} not supported, skipped.")
        continue
    full_data[impl].append(log_info)

base_dir = os.path.basename(log_dir)
import socket
hostname = socket.gethostname()
for impl, data in full_data.items():
    if len(data) > 0:
        output_csv = hostname + "-" + impl + "-" + base_dir + "-results.csv"
        df = pd.DataFrame(data, columns=field_names)
        df.to_csv(os.path.join(log_dir, output_csv), mode="a", index=False)
