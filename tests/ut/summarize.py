import fire
import pandas as pd
import os

from utils import *

def main(
    lib_target: str = "",
    cuda_path: str = "",
    xpu_path: str = "",
    output_dir: str = "",
):
    os.makedirs(output_dir, exist_ok=True)
    cuda_df = pd.read_excel(cuda_path)
    xpu_df = pd.read_excel(xpu_path)

    if lib_target == "transformers":
        cuda_df = cuda_df[cuda_df["cuda should only?"] != 1]
        xpu_df = xpu_df[xpu_df["cuda should only?"] != 1]
        cuda_df = cuda_df[cuda_df["cuda shouldnot only?"] != 1]
        xpu_df = xpu_df[xpu_df["cuda shouldnot only?"] != 1]
        cuda_df = cuda_df[cuda_df["xpu missing features?"] != 1]
        xpu_df = xpu_df[xpu_df["xpu missing features?"] != 1]
    
    save_ut_results_to_txt(cuda_df, output_dir, "cuda")
    save_ut_results_to_txt(xpu_df, output_dir, "xpu")
    
    def print_ut_stats(tests_df):
        print("========Overview========")
        print(f"TOTAL: {tests_df.shape[0]}")
        
        passed_df = tests_df[tests_df["result"] == "PASSED"]
        print(f"PASSED: {passed_df.shape[0]}")
        
        failed_df = tests_df[(tests_df["result"] == "FAILED")]
        print(f"FAILED: {failed_df.shape[0]}")

        skipped_df = tests_df[(tests_df["result"] == "SKIPPED")]
        print(f"SKIPPED: {skipped_df.shape[0]}")
        
        return passed_df, failed_df, skipped_df 
    
    print(f"+++++++++++++++++CUDA+++++++++++++++++")
    _, _, _ = print_ut_stats(cuda_df)
    print(f"+++++++++++++++++XPU+++++++++++++++++")
    _, failed_df, skipped_df = print_ut_stats(xpu_df)
    
    print("========FAILED========")
    cuda_also_fails = failed_df[failed_df["cuda also failed?"] == 1]
    print(f"Cuda also fails: {cuda_also_fails.shape[0]}")
    save_cases_to_txt(cuda_also_fails, os.path.join(output_dir, "cuda_also_fails.txt") )
    
    to_debug = failed_df[failed_df["cuda also failed?"] != 1]
    print(f"To debug: {to_debug.shape[0]}")
    to_debug[RELEVANT_COLS].to_excel(os.path.join(output_dir, "fail_to_debug.xlsx"), index=False)
    
    print("========SKIPPED========")
    cuda_also_skipped = skipped_df[skipped_df["cuda also skipped?"] == 1]
    print(f"Cuda also skips: {cuda_also_skipped.shape[0]}")
    save_cases_to_txt(cuda_also_skipped, os.path.join(output_dir, "cuda_also_skips.txt") )
    
    other_skipped = skipped_df[skipped_df["cuda also skipped?"] != 1]
    xpu_missing_features = other_skipped[other_skipped["xpu missing features?"] == 1]
    print(f"XPU missing features: {xpu_missing_features.shape[0]}")
    
    to_investigate = other_skipped[other_skipped["xpu missing features?"] != 1]
    print(f"To investigate: {to_investigate.shape[0]}")
    to_investigate[RELEVANT_COLS].to_excel(os.path.join(output_dir, "skip_to_investigate.xlsx"), index=False)


if __name__ == "__main__":
    fire.Fire(main)
