import fire
import pandas as pd
import glob
import os

from utils import *


def main(
    excel_dir: str = "",
    file_name: str = "",
):
    # read xpu ut excel files
    tests_df = merge_excel_files_to_df(excel_dir)
    tests_df = tests_df[RELEVANT_COLS]

    tests_df.to_excel(file_name, index=False)

    save_skipped_stats_to_excel(tests_df, f"{file_name.split('.')[0]}_skipped.xlsx")
    save_failed_stats_to_excel(tests_df, f"{file_name.split('.')[0]}_failed.xlsx")

    print_ut_stats(tests_df)


if __name__ == "__main__":
    fire.Fire(main)
