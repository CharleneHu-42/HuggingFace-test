import glob
import os
import pandas as pd

RELEVANT_COLS = [
    "file_name",
    "suite_name",
    "test_name",
    "result",
    "message",
    "duration",
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


def read_txt_to_list(ignore_file):
    cases_list = []
    if os.path.isfile(ignore_file):
        with open(ignore_file, "r") as f:
            cases_list.append([line.strip() for line in f.readlines()])

    cases = [case for cases in cases_list for case in cases]
    return cases


def save_list_to_txt(list, output_file):
    list = sorted(list)
    with open(output_file, "w") as file:
        for case in list:
            file.write(case + "\n")

def reorder_txt_cases(file):            
    cases = read_txt_to_list(file)
    save_list_to_txt(cases, file)
    
def save_cases_to_bash(df, output_file):
    df = df.sort_values(by=["suite_name", "test_name"])

    cases = []

    for _, row in df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        cases.append(
            f"pytest -rA tests -k '{suite_name} and {test_name}' --excelreport RERUN/{suite_name+test_name}.xlsx"
        )

    save_list_to_txt(cases, output_file)


def save_cases_to_txt(df, output_file):

    df = df.sort_values(by=["suite_name", "test_name"])

    cases = []
    for _, row in df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        cases.append(f"{suite_name}::{test_name}")

    save_list_to_txt(cases, output_file)


def save_cases_with_empty_messages(file_name):
    df = pd.read_excel(file_name)

    skipped = df[(df["result"] == "SKIPPED") | (df["result"] == "FAILED")]
    rerun = skipped[skipped["message"].isnull()]

    save_cases_to_bash(rerun, "rerun.sh")


def merge_excel_files_to_one(input_path, output_file):

    all_test_files = glob.glob(os.path.join(input_path, "*.xlsx"))

    df_merged = pd.concat(
        pd.read_excel(excel_file) for excel_file in all_test_files
    ).reset_index()
    df_merged.drop(columns=["index"], inplace=True)

    df_merged.to_excel(output_file, index=False)


def save_skipped_cases_to_txt(input_file, output_file):

    df = pd.read_excel(input_file)
    df_skipped = df[df["result"] == "SKIPPED"]

    save_cases_to_txt(df_skipped, output_file)


def save_failed_cases_to_txt(input_file, output_file):

    df = pd.read_excel(input_file)
    df_skipped = df[df["result"] == "FAILED"]

    save_cases_to_txt(df_skipped, output_file)


def save_skipped_stats_to_excel(df, output_file):
    skipped_df = df[df["result"] == "SKIPPED"]["message"].value_counts().reset_index()
    skipped_df.to_excel(output_file, index=False)


def save_failed_stats_to_excel(df, output_file):
    skipped_df = df[df["result"] == "FAILED"]["message"].value_counts().reset_index()
    skipped_df.to_excel(output_file, index=False)


def save_skipped_stats_to_excel_from_excel(input_file, output_file):
    df = pd.read_excel(input_file)
    save_skipped_stats_to_excel(df, output_file)


def save_failed_stats_to_excel_from_excel(input_file, output_file):
    df = pd.read_excel(input_file)
    save_failed_stats_to_excel(df, output_file)


def mark_cuda_failed_skipped_cases(xpu_file, cuda_file, output_file):
    xpu_df = pd.read_excel(xpu_file)
    cuda_df = pd.read_excel(cuda_file)

    xpu_df["cuda also"] = [0] * xpu_df.shape[0]

    pvc_skipped = xpu_df[xpu_df["result"] == "SKIPPED"]
    cuda_skipped = cuda_df[cuda_df["result"] == "SKIPPED"]

    def mark_same_with_cuda(cuda_df, xpu_df, xpu_df_all):

        for _, row in cuda_df.iterrows():
            suite_name = row["suite_name"]
            test_name = row["test_name"]
            target = xpu_df[
                (xpu_df["suite_name"] == suite_name)
                & (xpu_df["test_name"] == test_name)
            ]
            xpu_df_all.loc[target.index, "cuda also"] = 1

        return xpu_df_all

    xpu_df = mark_same_with_cuda(cuda_skipped, pvc_skipped, xpu_df)

    pvc_failed = xpu_df[xpu_df["result"] == "FAILED"]
    cuda_failed = cuda_df[cuda_df["result"] == "FAILED"]

    xpu_df = mark_same_with_cuda(cuda_failed, pvc_failed, xpu_df)

    xpu_df.to_excel(output_file, index=False)


def print_ut_stats(tests_df):
    result_stats = tests_df["result"].value_counts()
    pass_rate = result_stats["PASSED"] / sum(result_stats)
    print(f"=====UT PASS RATE=====\n{pass_rate}")
    print(f"=====TOTAL UT=====\n{tests_df.shape[0]}")
    print(f"=====DETAILS=====\n{result_stats}")


def remove_cases_duplicated(input_txt, output_txt):
    cases_list = read_txt_to_list(input_txt)
    cases_list = ["::".join(case.split("::")[-2:]) for case in cases_list]

    cased_unique = list(set(cases_list))
    cased_unique = sorted(cased_unique)

    save_list_to_txt(cased_unique, output_txt)


def merge_excel_files_to_df(excel_path):
    all_excel_files = glob.glob(os.path.join(excel_path, "*.xlsx"))

    df = pd.concat(
        pd.read_excel(excel_file) for excel_file in all_excel_files
    ).reset_index()
    df.drop(columns=["index"], inplace=True)

    return df


def update_ut_result_after_rerun(rerun_excel_path, ori_excel_file, output_file):
    rerun_df = merge_excel_files_to_df(rerun_excel_path)
    rerun_df = rerun_df[RELEVANT_COLS]

    ori_df = pd.read_excel(ori_excel_file)
    ori_df = ori_df[RELEVANT_COLS]

    ori_df = ori_df.sort_values(by=["file_name", "suite_name", "test_name"])
    rerun_df = rerun_df.sort_values(by=["file_name", "suite_name", "test_name"])

    for _, row in rerun_df.iterrows():
        file_name = row["file_name"]
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        result = row["result"]
        message = row["message"]

        sample = ori_df[
            (ori_df["file_name"] == file_name)
            & (ori_df["suite_name"] == suite_name)
            & (ori_df["test_name"] == test_name)
        ]
        ori_df.loc[sample.index, "result"] = result
        ori_df.loc[sample.index, "message"] = message

    ori_df.to_excel(output_file, index=False)
    
    
def add_column_and_mark_with_case_list(df, new_column, case_list):
    df[new_column] = [0] * df.shape[0]

    for index, row in df.iterrows():
        file_name = row["file_name"]
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        if (
            f"{suite_name}::{test_name}" in case_list
            or f"{file_name}::{suite_name}::{test_name}" in case_list
            or f"{file_name}::{test_name}" in case_list
        ):
            df.iloc[index, -1] = 1
    return df