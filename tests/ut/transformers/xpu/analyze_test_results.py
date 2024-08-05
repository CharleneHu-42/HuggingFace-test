import fire
import pandas as pd
import glob
import os

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
]

def replace_unittests(df):
    for i, v in df["file_name"].items():
        if "/unittest/" in str(v):
            suite_name = df.loc[i, "suite_name"]
            new_df = (
                df.groupby(["suite_name", "file_name"], as_index=False)
                .size()
                .sort_values(["size"], ascending=False)
            )
            new_value = new_df[new_df["suite_name"] == suite_name]["file_name"].iloc[0]
            df.loc[i, "file_name"] = new_value
    return df


def main(
    excel_dir: str = "",
    gpu_failed_path: str = "",
    gpu_skipped_path: str = "",
    output_dir: str = "",
):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(gpu_failed_path):
        with open(gpu_failed_path, "r") as f:
            gpu_cannot_run = [line.strip() for line in f.readlines()]

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

    if os.path.isfile(gpu_failed_path):
        tests_df["same as gpu?"] = [0] * tests_df.shape[0]
        df_tmp = tests_df[tests_df["result"] == "FAILED"]

        for index, row in df_tmp.iterrows():
            suite_name = row["suite_name"]
            test_name = row["test_name"]
            if f"{suite_name}::{test_name}" in gpu_cannot_run:
                tests_df.iloc[index, -1] = 1

    SKIP_MESSAGES = XPU_MISSING_FEATURES + GPU_ONLY
    tests_df["xpu-irrelevant"] = [0] * tests_df.shape[0]
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

    result_stats = tests_df["result"].value_counts()
    pass_rate = result_stats["PASSED"] / sum(result_stats)
    print(f"=====UT PASS RATE=====\n{pass_rate}")
    print(f"=====DETAILS=====\n{result_stats}")


if __name__ == "__main__":
    fire.Fire(main)
