import fire
import pandas as pd
import glob
import os

from utils import *

def main(
    file_name: str = "",
    output_dir: str = "",
):
    os.makedirs(output_dir, exist_ok=True)
    tests_df = pd.read_excel(file_name)
    
    tests_df = tests_df[tests_df["ignore"] != 1]
    print("========Overview========")
    print(f"#TOTAL UT: {tests_df.shape[0]}")
    
    passed_df = tests_df[tests_df["result"] == "PASSED"]
    print(f"#PASSED UT: {passed_df.shape[0]}")
    
    failed_df = tests_df[(tests_df["result"] == "FAILED")]
    print(f"#FAILED UT: {failed_df.shape[0]}")

    skipped_df = tests_df[(tests_df["result"] == "SKIPPED")]
    print(f"#SKIPPED UT: {skipped_df.shape[0]}")
    
    print("========FAILED========")
    cuda_also_fails = failed_df[failed_df["cuda also failed?"] == 1]
    print(f"Cuda also fails: {cuda_also_fails.shape[0]}")
    save_cases_to_txt(cuda_also_fails, os.path.join(output_dir, "cuda_also_fails.txt") )
    
    to_debug = failed_df[failed_df["cuda also failed?"] != 1]
    print(f"To debug: {to_debug.shape[0]}")
    to_debug[RELEVANT_COLS].to_excel(os.path.join(output_dir, "fail_to_debug.xlsx"), index=False)
    
    print("========SKIPPED========")
    cuda_also_skipped = skipped_df[skipped_df["cuda also skipped?"] == 1]
    print(f"Cuda also skips: {cuda_also_fails.shape[0]}")
    save_cases_to_txt(cuda_also_skipped, os.path.join(output_dir, "cuda_also_skips.txt") )
    
    other_skipped = skipped_df[skipped_df["cuda also skipped?"] != 1]
    xpu_missing_features = other_skipped[other_skipped["xpu missing features?"] == 1]
    print(f"XPU missing features: {xpu_missing_features.shape[0]}")
    
    to_investigate = other_skipped[other_skipped["xpu missing features?"] != 1]
    print(f"To investigate: {to_investigate.shape[0]}")
    to_investigate[RELEVANT_COLS].to_excel(os.path.join(output_dir, "skip_to_investigate.xlsx"), index=False)

if __name__ == "__main__":
    fire.Fire(main)
