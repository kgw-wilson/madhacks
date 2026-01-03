# Try clustering on input compared to kNN, compare both to clustering on differences using FLAN
# This experiment is just english data


import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    cosine_similarity,
    cosine_distances,
)


# Create dfs
file_path = "rb_data_with_flan_and_task.pkl"
df = pd.read_pickle(file_path)

# Flan response was obtained only for english examples
df = df[
    (df["flan_s_response"].notna())
    & (df["flan_s_response_embedding"].notna())
    & (df["task_description_embedding"].notna())
]
# filter df to try to only get good task descriptions
exclude_keywords = ["Answer", "Choice", "Choose"]
df = df[
    (df["task_description"].str.len() >= 10)  # Check minimum length
    & ~df["task_description"].str.contains(
        "|".join(exclude_keywords), case=False, na=False
    )  # Exclude rows with keywords
]


# Train-test split (20% eval, 80% train)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare the training embeddings and oracle models
train_embeddings = np.stack(train_df["task_description_embedding"])
train_oracle_models = train_df["oracle_model_to_route_to"]


"""
Accuracy at k=1: 72.25% (992/1373)
Accuracy at k=2: 63.58% (873/1373)
Accuracy at k=3: 77.13% (1059/1373)
Accuracy at k=5: 78.66% (1080/1373)
Accuracy at k=10: 81.06% (1113/1373)
Accuracy at k=20: 81.65% (1121/1373)
Accuracy at k=40: 81.65% (1121/1373)
"""


print("BEGINNING kNN")

# Number of nearest neighbors to consider (40 was best according to paper)
k_schedule = [
    1,
    2,
    3,
    5,
    10,
    20,
    40,
]


def predict_oracle_model_knn(eval_embedding, train_embeddings, train_oracle_models):
    """Predict the best model based on the nearest neighbors in input embedding space"""

    # Compute cosine similarity
    similarities = cosine_similarity([eval_embedding], train_embeddings)[0]

    # Get indices of the top-k closest embeddings
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # Get the corresponding oracle models
    top_k_models = train_oracle_models.iloc[top_k_indices]

    # Return the most frequent oracle model in the top-k (majority vote)
    predicted_model = pd.Series(top_k_models).mode().iloc[0]
    return predicted_model


for k in k_schedule:

    # Initialize variables to track performance
    total_predictions = 0
    correct_predictions = 0

    # Loop through each row in the evaluation set
    for _, eval_row in eval_df.iterrows():
        eval_embedding = eval_row["task_description_embedding"]

        actual_oracle_model = eval_row["oracle_model_to_route_to"]

        # Predict the oracle model for the current evaluation row
        predicted_model = predict_oracle_model_knn(
            eval_embedding, train_embeddings, train_oracle_models
        )

        # Check if the prediction matches the actual model
        if predicted_model == actual_oracle_model:
            correct_predictions += 1
        total_predictions += 1

    # Calculate and display accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(
        f"Accuracy at {k=}: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})"
    )


"""
Accuracy at k=1: 4.66% (64/1373)
Accuracy at k=5: 47.20% (648/1373)
Accuracy at k=10: 58.70% (806/1373)
Accuracy at k=20: 74.14% (1018/1373)
Accuracy at k=50: 60.60% (832/1373)
Accuracy at k=100: 67.59% (928/1373)
Accuracy at k=200: 70.07% (962/1373)
Accuracy at k=500: 71.38% (980/1373)
Accuracy at k=1000: 73.63% (1011/1373)
"""

print("BEGIN clustering")

# Set the number of clusters in kMeans
k_schedule = [
    1,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    # len(train_df) // 2, # takes too long
]


# Define the function to predict based on medoids
def predict_oracle_model_with_cluster(eval_embedding):
    """
    Predict the best oracle model based on the closest medoid.

    Args:
        eval_embedding (numpy array): The embedding of the evaluation example

    Returns:
        dict: The oracle model and metadata of the closest medoid.
    """

    similarities = {}
    for cluster, prompt_idx in closest_prompt_idx_by_cluster.items():
        similarities[cluster] = cosine_similarity(
            # [eval_embedding], [train_df.loc[prompt_idx]["prompt_embedding"]]
            [eval_embedding],
            [train_df.loc[prompt_idx]["task_description_embedding"]],
        )[0, 0]

    # Find the cluster with the highest similarity
    best_cluster = max(similarities, key=similarities.get)
    best_prompt_idx = closest_prompt_idx_by_cluster[best_cluster]

    # Return the oracle model for the closest cluster
    return train_df.loc[best_prompt_idx]["oracle_model_to_route_to"]


for k in k_schedule:

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_embeddings)

    # Initialize variables to track performance
    total_predictions = 0
    correct_predictions = 0

    # Create new dictionary mapping each cluster label to the index in train_df of
    # the prompt in that cluster with the closest embedding to the cluster center
    closest_prompt_idx_by_cluster = {k_id: None for k_id in range(k)}

    # Loop through the dataframe and assign each prompt to a cluster_label
    train_df["cluster_label"] = kmeans.predict(train_embeddings)

    # Find the closest prompt embedding to the cluster center for each cluster
    for cluster in range(k):

        # Get all points in the current cluster
        cluster_indices = train_df[train_df["cluster_label"] == cluster].index.values
        cluster_points = np.stack(
            train_df["task_description_embedding"][cluster_indices]
        )

        # Compute pairwise cosine distances within the cluster to the center
        distances = cosine_distances(
            cluster_points, [kmeans.cluster_centers_[cluster]]
        ).flatten()

        # Find the index of the medoid (point with minimum sum of distances to other points)
        medoid_idx_in_cluster = np.argmin(distances)
        medoid_idx_global = cluster_indices[medoid_idx_in_cluster]

        closest_prompt_idx_by_cluster[cluster] = medoid_idx_global

    # Loop through each row in the evaluation set
    for _, eval_row in eval_df.iterrows():
        eval_embedding = eval_row["task_description_embedding"]
        actual_oracle_model = eval_row["oracle_model_to_route_to"]

        # Predict the oracle model for the current evaluation row
        predicted_model = predict_oracle_model_with_cluster(eval_embedding)

        # Check if the prediction matches the actual model
        if predicted_model == actual_oracle_model:
            correct_predictions += 1
        total_predictions += 1

    # Calculate and display accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(
        f"Accuracy at {k=}: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})"
    )
