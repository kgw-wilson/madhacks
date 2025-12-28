from pathlib import Path
import ast
import requests
import pandas as pd
from data_processing.constants import (
    SAMPLE_ID_COL,
    PROMPT_COL,
    MODEL_PERFORMANCE_COLS,
    RB_DATA_PATH,
    FILTERED_CSV_PATH,
    RESPONSE_SUFFIX,
)

RB_DATA_DOWNLOAD_URL = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl?download=true"

RB_PKL_PATH = RB_DATA_PATH / "unprocessed_0shot.pkl"


def download_routerbench() -> None:
    """Downloads the .pkl file from HuggingFace and saves it."""

    print(f"Beginning download for 0-shot RouterBench data...")

    with requests.get(RB_DATA_DOWNLOAD_URL, stream=True) as r:
        r.raise_for_status()
        with open(RB_PKL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Downloaded RouterBench data to: {RB_PKL_PATH}")


def filter_routerbench_data_and_save_csv() -> None:
    """
    Filters the pickled routerbench data to only include known model names.

    What follows is rationale for what columns are kept/cut.
    We need unique identifiers for each sample so that we can tell them
    apart when debugging. We need prompts and the responses from each model so
    we can use them to generate embeddings later for use in our experiments.
    We need which model was the best (cheapest and correct) for each prompt
    so that we can predict it with supervised learning. Everything else,
    including cost and dataset name is excluded to reduce cognitive load.
    """

    print(f"Beginning filtering 0-shot RouterBench data...")

    df = pd.read_pickle(RB_PKL_PATH)

    # Establish which columns to keep in our filtered data (see docstring above)
    response_cols = [col for col in df.columns if col.endswith(RESPONSE_SUFFIX)]
    columns_to_keep = [
        SAMPLE_ID_COL,
        PROMPT_COL,
        *MODEL_PERFORMANCE_COLS,
        *response_cols,
    ]

    filtered_columns_df = df[columns_to_keep].copy()

    def _safe_parse_and_join(value: str) -> str:
        """Cleans RouterBench text data at the cell level

        Model responses are in a weird string of list of strings format so
        the code below simplifies them into just strings with leading/trailing
        whitespace removed. Also handles missing data and numeric values.
        """
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return str(value).strip()
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return " ".join(str(item).strip() for item in parsed)
            else:
                return str(parsed).strip()
        except (ValueError, SyntaxError):
            return str(value).strip()

    for col in [PROMPT_COL, *response_cols]:
        df[col] = df[col].apply(_safe_parse_and_join)

    # Save the resulting DataFrame (expecting len=36497, all samples from RouterBench)
    # CSV format is chosen to easily view data and troubleshoot any issues
    filtered_columns_df.to_csv(FILTERED_CSV_PATH, index=False)
    print(
        f"Filtered RouterBench data with len {len(filtered_columns_df)} saved to: {FILTERED_CSV_PATH}"
    )


if __name__ == "__main__":
    if RB_PKL_PATH.exists():
        print(f"Routerbench data already exists at: {RB_PKL_PATH}")
    else:
        download_routerbench()
    if FILTERED_CSV_PATH.exists():
        print(f"Filtered RouterBench data already exists at: {FILTERED_CSV_PATH}")
    else:
        filter_routerbench_data_and_save_csv()
