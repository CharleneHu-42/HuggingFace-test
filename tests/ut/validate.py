import pandas as pd
import re
import glob
import os
import sys
from pathlib import Path
from utils import *

lib_target = sys.argv[1]

txt_files_glob = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), lib_target, "*.txt"))
)
pattern = r"(.*) tests"

rerun_cases = []
for txt_file in txt_files_glob:
    txt_file_name = os.path.basename(txt_file).split(".")[0]
    excel_file_path = txt_file.replace("txt", "xlsx")

    if not os.path.exists(excel_file_path):
        rerun_cases.append(txt_file_name)
        continue

    with open(txt_file, "r") as file:
        lines = file.readlines()
        last_line = lines[-1] if lines else None
        match = re.match(pattern, last_line)
        true_case_num = match.group(1)

    if "/" in true_case_num:
        true_case_num = true_case_num.split("/")[0]

    df = pd.read_excel(excel_file_path)

    real_case_num = df.shape[0]

    if int(true_case_num) != real_case_num:
        rerun_cases.append(txt_file_name)
        save_dir = os.path.join(os.path.dirname(__file__), lib_target)
        save_cases_to_txt(df, os.path.join(save_dir, f"{txt_file_name}_real.txt"))
        extract_short_cases(
            txt_file, os.path.join(save_dir, f"{txt_file_name}_true.txt")
        )

    print("+", end="", flush=True)  # Print + in the same line
print(f"\n{rerun_cases} need double-check.")
