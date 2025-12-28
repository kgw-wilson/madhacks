from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from data_processing.constants import PROMPT_EMBEDDING_COL, RESULTS_CSV_FILENAME
from utils import (
    predict_top_k,
    evaluate_predictions,
    TEST_SPLIT_SIZE,
    K_VALUES,
    RANDOM_STATE,
)
from data_processing.add_embeddings import FILTERED_WITH_EMBEDDINGS_PATH


# Directory for exp1 - (where to save results)
EXP1_PATH = Path(__file__).parent.parent / "exp1"
RESULTS_CSV_PATH = EXP1_PATH / RESULTS_CSV_FILENAME


def _run_experiment() -> None:
    """Run a kNN style evaluation using prompt embeddings"""

    df = pd.read_pickle(FILTERED_WITH_EMBEDDINGS_PATH)
    train_df, eval_df = train_test_split(
        df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    # List for saving experiment results will be turned into DataFrame later
    results_rows = []

    # Convert promp embeddings to arrays
    train_embeddings = np.stack(train_df[PROMPT_EMBEDDING_COL])
    eval_embeddings = np.stack(eval_df[PROMPT_EMBEDDING_COL])

    # Normalize embeddings for better nearest neighbors performance
    train_embeddings = normalize(train_embeddings)
    eval_embeddings = normalize(eval_embeddings)

    for k in K_VALUES:

        print(f"Running evaluation for {k=}")
        start = time.time()

        predictions = predict_top_k(
            eval_embeddings,
            train_embeddings,
            train_df,
            k,
        )

        accuracy, correct, total = evaluate_predictions(eval_df, predictions)

        elapsed = time.time() - start

        results_rows.append(
            {
                "k": k,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "time_sec": elapsed,
            }
        )

    print(f"Saving experiment results to: {RESULTS_CSV_PATH}")
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)


if __name__ == "__main__":
    if RESULTS_CSV_PATH.exists():
        print(f"Experiment 1 results already exist at: {RESULTS_CSV_PATH}")
    else:
        _run_experiment()
