import fire
import pandas as pd
import glob
import os


SKIP_MESSAGES = [
    "test requires natten",
    "test requires Flash Attention",
    "test requires TensorFlow",
    "test requires JAX & Flax",
    "test requires PyTorch Quantization Toolkit",
    "test requires multiple GPUs",
    "test requires 0 or 1 GPU",
    "test requires 0 or 1 or 2 GPUs",
    "test requires TorchXLA",
    "test requires PyTorch NeuronCore",
    "test requires PyTorch NPU",
    "test requires multiple NPUs",
    "test requires Torch-TensorRT FX",
    "test requires CUDA",
    "test requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0",
    "test requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7",
    "test requires `detectron2`",
    "test requires Ray/tune",
    "test requires apex",
    "test requires aqlm",
    "test requires bitsandbytes and torch",
    "test requires auto-gptq",
    "test requires autoawq",
    "test requires quanto",
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


def transform_df(df, col_name):
    values = df[col_name].unique().tolist()
    passed = []
    skipped = []
    failed = []
    error = []
    for i, value in enumerate(values):
        tmp_df = df[df[col_name] == value]
        passed.append(0)
        skipped.append(0)
        failed.append(0)
        error.append(0)
        for i, v in tmp_df["result"].items():
            if v == "PASSED":
                passed[-1] = tmp_df["size"][i]
            elif v == "SKIPPED":
                skipped[-1] = tmp_df["size"][i]
            elif v == "ERROR":
                error[-1] = tmp_df["size"][i]
            else:
                failed[-1] = tmp_df["size"][i]

    new_df = pd.DataFrame(
        {
            "Test Vector": values,
            "PASS": passed,
            "FAIL": failed,
            "SKIP": skipped,
            "ERROR": error,
        }
    )
    new_df["TOTAL"] = new_df["PASS"] + new_df["FAIL"] + new_df["SKIP"] + new_df["ERROR"]

    return new_df


def extract_category(all_df):
    category = []
    for i, v in all_df["Test Vector"].items():
        end = v.split("/")
        if len(end) == 2:
            category.append(end[-1])
        else:
            category.append(end[1])
    all_df["Category"] = category
    return all_df


def main(
    excel_dir: str = "",
    gpu_failed_path: str = "",
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

    tests_df["xpu-irrelevant"] = [0] * tests_df.shape[0]
    for index, row in tests_df.iterrows():
        if row["message"] in SKIP_MESSAGES:
            tests_df.iloc[index, -1] = 1

    tests_df.to_excel(os.path.join(output_dir, "raw_test_results.xlsx"), index=False)
    # aggregate the results by file_name and result
    tests_df_agg = tests_df.groupby(["file_name", "result"], as_index=False).size()
    # transform the result values to individual columns
    tests_df_agg = transform_df(tests_df_agg, "file_name")
    # extract the test category from the file name
    tests_df_agg = extract_category(tests_df_agg)

    tests_stats = tests_df_agg[
        ["Category", "Test Vector", "PASS", "FAIL", "SKIP", "TOTAL"]
    ]
    tests_stats.to_excel(
        os.path.join(output_dir, "test_stats_by_category_and_file.xlsx"), index=False
    )
    # aggregate test results by category
    tests_stats2 = (
        tests_df_agg.groupby(["Category"])
        .agg({"PASS": "sum", "FAIL": "sum", "SKIP": "sum", "TOTAL": "sum"})
        .reset_index()
    )
    tests_stats2.to_excel(
        os.path.join(output_dir, "test_stats_by_category.xlsx"), index=False
    )

    skip_stats = (
        tests_df[tests_df["result"] == "SKIPPED"]["message"]
        .value_counts()
        .reset_index()
    )
    skip_stats.to_excel(
        os.path.join(output_dir, "skipped_tests_stats.xlsx"), index=False
    )


if __name__ == "__main__":
    fire.Fire(main)
