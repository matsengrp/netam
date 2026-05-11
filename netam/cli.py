import os
import shutil

import fire
import pandas as pd
import pypdf

from netam.pretrained import PRETRAINED_DIR


def clear_model_cache():
    """This function clears the cache of pre-trained models."""
    if os.path.exists(PRETRAINED_DIR):
        print(f"Removing {PRETRAINED_DIR}")
        shutil.rmtree(PRETRAINED_DIR)


def concatenate_csvs(
    input_csvs_str: str,
    output_csv: str,
    is_tsv: bool = False,
    record_path: bool = False,
):
    """This function concatenates multiple CSV or TSV files into one CSV file.

    Args:
        input_csvs: A string of paths to the input CSV or TSV files separated by commas.
        output_csv: Path to the output CSV file.
        is_tsv: A boolean flag that determines whether the input files are TSV.
        record_path: A boolean flag that adds a column recording the path of the input_csv.
    """
    input_csvs = input_csvs_str.split(",")
    dfs = []

    for csv in input_csvs:
        df = pd.read_csv(csv, delimiter="\t" if is_tsv else ",")
        if record_path:
            df["input_file_path"] = csv
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)

    result_df.to_csv(output_csv, index=False)


def concatenate_pdfs(
    input_pdfs_str: str,
    output_pdf: str,
):
    """This function concatenates multiple PDF files into one PDF file.

    Args:
        input_pdfs_str: A string of paths to the input PDF files separated by commas.
        output_pdf: Path to the output PDF file.
    """
    input_pdfs = input_pdfs_str.split(",")
    writer = pypdf.PdfWriter()

    for pdf in input_pdfs:
        if not os.path.isfile(pdf):
            raise FileNotFoundError(f"Input PDF not found: {pdf}")
        reader = pypdf.PdfReader(pdf)
        for page in reader.pages:
            writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)


def main():
    fire.Fire()
