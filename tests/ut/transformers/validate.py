import pandas as pd
import re
import glob
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *

excel_files_glob = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.xlsx")))
pattern = r"(.*) tests"

rerun_cases = []
for excel_file in excel_files_glob:
    excel_file_name = os.path.basename(excel_file).split(".")[0]
    txt_file = excel_file.replace("xlsx", "txt")

    with open(txt_file, "r") as file:
        lines = file.readlines()
        last_line = lines[-1] if lines else None
        match = re.match(pattern, last_line)
        true_case_num = match.group(1)

    if "/" in true_case_num:
        true_case_num = true_case_num.split("/")[0]

    df = pd.read_excel(excel_file)
    real_case_num = df.shape[0]

    if int(true_case_num) != real_case_num:
        rerun_cases.append(excel_file_name)
        save_cases_to_txt(df, f"{excel_file_name}_real.txt")
        extract_short_cases(txt_file, f"{excel_file_name}_true.txt")

    print("+", end="", flush=True)  # Print + in the same line
print(f"\n{rerun_cases} need double-check.")
