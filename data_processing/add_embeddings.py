import pandas as pd
from tqdm import tqdm
from data_processing.constants import (
    PROMPT_COL,
    PROMPT_EMBEDDING_COL,
    RESPONSE_SUFFIX,
    RESPONSE_EMBEDDING_SUFFIX,
    FILTERED_WITH_EMBEDDINGS_PATH,
    FILTERED_CSV_PATH,
    embedding_model,
)


def update_data_with_prompt_response_embeddings() -> None:
    """
    Creates a new DataFrame with columns for prompt and response embeddings

    All embeddings are generated in batches for speed and memory efficiency.
    """

    df = pd.read_csv(FILTERED_CSV_PATH)

    print("Encoding all prompts...")

    prompt_embeddings = embedding_model.encode(
        df[PROMPT_COL].to_list(), batch_size=64, show_progress_bar=True
    )

    df[PROMPT_EMBEDDING_COL] = list(prompt_embeddings)

    print("Encoding all responses...")

    response_cols = [col for col in df.columns if col.endswith(RESPONSE_SUFFIX)]

    for col in tqdm(response_cols, desc="Percentage of total responses encoded"):
        response_embeddings = embedding_model.encode(
            df[col].tolist(), batch_size=64, show_progress_bar=True
        )
        embedding_col_name = col.split(RESPONSE_SUFFIX)[0] + RESPONSE_EMBEDDING_SUFFIX
        df[embedding_col_name] = list(response_embeddings)

    # Save the updated DataFrame as a .pkl file, pickling is done instead of
    # CSV for file size and because embeddings are not human readable anyway
    df.to_pickle(FILTERED_WITH_EMBEDDINGS_PATH)
    print(f"Updated dataset saved to {FILTERED_WITH_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    if FILTERED_WITH_EMBEDDINGS_PATH.exists():
        print(
            f"Filtered RouterBench data with prompt and response embeddings exists at: {FILTERED_WITH_EMBEDDINGS_PATH}"
        )
    else:
        update_data_with_prompt_response_embeddings()
