import fire
import pandas as pd
import glob
import os

from utils import read_txt_to_list, print_ut_stats, replace_unittests

XPU_MISSING_FEATURES = [
    "test requires natten",
    "test requires apex",
    "test requires aqlm",
    "test requires bitsandbytes and torch",
    "test requires auto-gptq",
    "test requires autoawq",
    "test requires quanto",
    "test requires gguf",
    "test requires LOMO",
    "test requires `detectron2`",
    "test requires Flash Attention",
]

GPU_ONLY = [
    "test requires Torch-TensorRT FX",
    "test requires PyTorch Quantization Toolkit",
    "test requires TorchXLA",
    "test requires JAX & Flax",
    "test requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0",
    "test requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7",
    "test requires GaLore",
]
    
    
def main(
    excel_dir: str = "",
    ignore_path: str = "",
    output_dir: str = "",
):
    os.makedirs(output_dir, exist_ok=True)

    cuda_also_failed = os.path.join(ignore_path, "cuda_also_failed.txt")
    cuda_also_skipped = os.path.join(ignore_path, "cuda_also_skipped.txt")
    cuda_only = os.path.join(ignore_path, "cuda_only.txt")
    memory_api = os.path.join(ignore_path, "available_memory_api.txt")

    cuda_failed_cases = read_txt_to_list(cuda_also_failed)
    cuda_skipped_cases = read_txt_to_list(cuda_also_skipped)
    cuda_only_cases = read_txt_to_list(cuda_only)
    memory_api_cases = read_txt_to_list(memory_api)
    
    xpu_missing_files = glob.glob(os.path.join(ignore_path, "xpu_missing_*.txt"))
    all_files = [read_txt_to_list(file) for file in xpu_missing_files]
    xpu_missing_cases = [case for file in all_files for case in file]
    
    all_test_files = glob.glob(os.path.join(excel_dir, "*.xlsx"))

    tests_df = pd.concat(
        pd.read_excel(excel_file) for excel_file in all_test_files
    ).reset_index()
    tests_df.drop(columns=["index"], inplace=True)

    tests_df = tests_df[
        ["file_name", "suite_name", "test_name", "result", "message", "duration"]
    ]
    
    # if the file name contains `unittest`, we will need to manually replace it with the actual file name
    tests_df = replace_unittests(tests_df)
    
    tests_df["same with gpu?"] = [0] * tests_df.shape[0]

    same_with_cuda = cuda_failed_cases + cuda_skipped_cases
    
    for index, row in tests_df.iterrows():
        file_name = row["file_name"]
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        if f"{suite_name}::{test_name}" in same_with_cuda or f"{file_name}::{suite_name}::{test_name}" in same_with_cuda:
            tests_df.iloc[index, -1] = 1
    
    tests_df["cuda only?"] = [0] * tests_df.shape[0]
    
    for index, row in tests_df.iterrows():
        file_name = row["file_name"]
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        if f"{suite_name}::{test_name}" in cuda_only_cases or f"{file_name}::{suite_name}::{test_name}" in cuda_only_cases:
            tests_df.iloc[index, -1] = 1
    
    tests_df["available memory api?"] = [0] * tests_df.shape[0]
    
    for index, row in tests_df.iterrows():
        file_name = row["file_name"]
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        if f"{suite_name}::{test_name}" in memory_api_cases or f"{file_name}::{suite_name}::{test_name}" in memory_api_cases:
            tests_df.iloc[index, -1] = 1
            
    tests_df["xpu missing features?"] = [0] * tests_df.shape[0]
    
    for index, row in tests_df.iterrows():
        file_name = row["file_name"]
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        if f"{suite_name}::{test_name}" in xpu_missing_cases or f"{file_name}::{suite_name}::{test_name}" in xpu_missing_cases:
            tests_df.iloc[index, -1] = 1
    
    SKIP_MESSAGES = XPU_MISSING_FEATURES + GPU_ONLY
    tests_df["other skips?"] = [0] * tests_df.shape[0]
    
    for index, row in tests_df.iterrows():
        if row["message"] in SKIP_MESSAGES:
            tests_df.iloc[index, -1] = 1

    tests_df.to_excel(os.path.join(output_dir, "raw_test_results.xlsx"), index=False)

    skip_stats = (
        tests_df[tests_df["result"] == "SKIPPED"]["message"]
        .value_counts()
        .reset_index()
    )
    skip_stats.to_excel(
        os.path.join(output_dir, "skipped_tests_stats.xlsx"), index=False
    )
    failed_stats = (
        tests_df[tests_df["result"] == "FAILED"]["message"].value_counts().reset_index()
    )
    failed_stats.to_excel(
        os.path.join(output_dir, "failed_tests_stats.xlsx"), index=False
    )

    print_ut_stats(tests_df)
    

if __name__ == "__main__":
    fire.Fire(main)
