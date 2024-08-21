import fire
import pandas as pd
import glob
import os

from utils import read_txt_to_list, print_ut_stats, replace_unittests, save_skipped_stats_to_excel, save_failed_stats_to_excel

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

    memory_api = os.path.join(ignore_path, "available_memory_api.txt")
    memory_api_cases = read_txt_to_list(memory_api)
    
    cuda_also_failed_file = os.path.join(".", "cuda_also_failed.txt")
    cuda_also_skipped_file = os.path.join(".", "cuda_also_skipped.txt")
    cuda_also_failed = read_txt_to_list(cuda_also_failed_file)
    cuda_also_skipped = read_txt_to_list(cuda_also_skipped_file)
    
    only_files = glob.glob(os.path.join(ignore_path, "*_only.txt"))
    all_only_files = [read_txt_to_list(file) for file in only_files]
    only_cases = [case for file in all_only_files for case in file]
    
    xpu_missing_files = glob.glob(os.path.join(ignore_path, "xpu_missing_*.txt"))
    all_missing_files = [read_txt_to_list(file) for file in xpu_missing_files]
    xpu_missing_cases = [case for file in all_missing_files for case in file]
    
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
    
    def add_column_and_mark_with_case_list(df, new_column, case_list):
        df[new_column] = [0] * df.shape[0]
        
        for index, row in df.iterrows():
            file_name = row["file_name"]
            suite_name = row["suite_name"]
            test_name = row["test_name"]
            if f"{suite_name}::{test_name}" in case_list or f"{file_name}::{suite_name}::{test_name}" in case_list:
                df.iloc[index, -1] = 1
        return df 
    
    tests_df = add_column_and_mark_with_case_list(tests_df, "cuda also failed?", cuda_also_failed)
    tests_df = add_column_and_mark_with_case_list(tests_df, "cuda also skipped?", cuda_also_skipped)
    tests_df = add_column_and_mark_with_case_list(tests_df, "cuda/cpu/tpu only?", only_cases)                     
    tests_df = add_column_and_mark_with_case_list(tests_df, "available memory api?", memory_api_cases)            
    tests_df = add_column_and_mark_with_case_list(tests_df, "xpu missing features?", xpu_missing_cases)            

    SKIP_MESSAGES = XPU_MISSING_FEATURES + GPU_ONLY
    tests_df["other skips?"] = [0] * tests_df.shape[0]
    
    for index, row in tests_df.iterrows():
        if row["message"] in SKIP_MESSAGES:
            tests_df.iloc[index, -1] = 1

    tests_df.to_excel(os.path.join(output_dir, "raw_test_results.xlsx"), index=False)

    save_skipped_stats_to_excel(tests_df, "skipped_tests_stats.xlsx")
    save_failed_stats_to_excel(tests_df, "failed_tests_stats.xlsx")

    print_ut_stats(tests_df)
    

if __name__ == "__main__":
    fire.Fire(main)
