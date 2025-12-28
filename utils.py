import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from data_processing.constants import MODEL_PERFORMANCE_COLS


# How many neighbors to consider in the training data when doing evaluation
# These are the values tested in the RouterBench paper, see Appendix
# B. Extended Experimental Settings
K_VALUES = [5, 10, 40]

# Safe split because we still get enough training data to find nearest neighbors
# and the eval set is large enough to get stable accuracy estimates. This is
# the value they use in the RouterBench paper
TEST_SPLIT_SIZE = 0.3

# Ensures repeatability between runs
# We couldn't find a consistent seed used in the RouterBench code, so 42 was chosen
RANDOM_STATE = 42


def predict_top_k(
    eval_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    k: int,
) -> list[str]:
    """
    Predict best model for each eval embedding using most similar training embeddings

    sklearn is used because it's clean and optimized, better than computing a whole similarity
    matrix and sorting it. Let sklearn decide the best algorithm with "auto" based on the
    values passed to .fit() for simplicity, and set n_jobs to -1 to use all available CPU cores
    for speed. Cosine similarity metric isused because it's the metric SBERT was
    trained to minimize so it should be the most semantically descriptive.
    """

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="auto", n_jobs=-1)

    nn.fit(train_embeddings)
    _, indices = nn.kneighbors(eval_embeddings)

    predictions = []

    for neighbor_idxs in indices:
        # Grab k training rows with most similar embeddings to the eval embeddings
        neighbors = train_df.iloc[neighbor_idxs]

        # Mean performance per model
        mean_performance = neighbors[MODEL_PERFORMANCE_COLS].mean()

        # Pick model with highest expected performance
        best_model = mean_performance.idxmax()

        predictions.append(best_model)

    return predictions


def evaluate_predictions(
    eval_df: pd.DataFrame,
    predictions: list[str],
) -> tuple[float, int, int]:
    """
    Gets statistics about quality of predictions

    A prediction is considered correct if the model with the highest average performance
    across an evaluation sample's nearest neighbors in the training data is able to
    answer that evaluation prompt correctly.

    RouterBench's code seems to say "correct" at the sample level means any value
    over 0.0, so that's what is done here.
    """

    correct = 0
    total = 0

    # Extra enumerate done to not depend on DataFrame indices
    for i, (_, row) in enumerate(eval_df.iterrows()):
        predicted_model = predictions[i]
    # for idx, row in eval_df.iterrows():
    #     predicted_model = predictions[idx]
        was_model_correct = row[predicted_model] > 0.0
        if was_model_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total else 0

    return accuracy, correct, total
