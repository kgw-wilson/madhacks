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
file_path = "rb_data_with_flan.pkl"
df = pd.read_pickle(file_path)

# Flan response was obtained only for english examples
df = df[(df["flan_s_response"].notna()) & (df["flan_s_response_embedding"].notna())]

# Create new df column for differences between input embeddings and flan response embeddings
# df["embedding_diff"] = df["response_embedding"] - df["prompt_embedding"]
# df["embedding_diff"] = df["flan_s_response_embedding"] - df["prompt_embedding"] # first try
# df["embedding_diff"] = df["mixtral_response_embedding"] - df["prompt_embedding"]
df["embedding_diff"] = df["gpt4_response_embedding"] - df["prompt_embedding"]

# Train-test split (20% eval, 80% train)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare the training embeddings and oracle models
# train_embeddings = np.stack(train_df["prompt_embedding"])
train_embeddings = np.stack(train_df["flan_s_response_embedding"])
# train_embeddings = np.stack(train_df["mixtral_response_embedding"])
# train_embeddings = np.stack(train_df["gpt4_response_embedding"])
train_oracle_models = train_df["oracle_model_to_route_to"]

# try kNN using embedding diffs
# train_embeddings = np.stack(train_df["embedding_diff"])

"""
============================================================
Repeat exp1 (kNN on input embeddings) with just English data, compares prompt embedding in training data
to prompt embedding in eval data to find most similar, then gets oracle model for that row (requires knowing
all candidate model responses and suitability for every example in training data)
Accuracy at k=1: 74.17% (1783/2404)
Accuracy at k=2: 65.60% (1577/2404)
Accuracy at k=3: 80.32% (1931/2404)
Accuracy at k=5: 82.03% (1972/2404)
Accuracy at k=10: 83.90% (2017/2404)
Accuracy at k=20: 84.11% (2022/2404)
Accuracy at k=40: 84.28% (2026/2404)

Take 2: use FLAN, trained and evaluted using flan_s_response_embedding. Still need to know the oracle model
for all the training data, but is a proof of concept that using the response embedding instead of the input
embedding does work.
Accuracy at k=1: 81.16% (1951/2404)
Accuracy at k=2: 77.91% (1873/2404)
Accuracy at k=3: 82.65% (1987/2404)
Accuracy at k=5: 82.70% (1988/2404)
Accuracy at k=10: 83.32% (2003/2404)
Accuracy at k=20: 83.28% (2002/2404)
Accuracy at k=40: 83.28% (2002/2404)

Same as Take 2 but with mixtral
Accuracy at k=1: 70.30% (1690/2404)
Accuracy at k=2: 59.98% (1442/2404)
Accuracy at k=3: 72.05% (1732/2404)
Accuracy at k=5: 73.46% (1766/2404)
Accuracy at k=10: 83.28% (2002/2404)
Accuracy at k=20: 83.19% (2000/2404)
Accuracy at k=40: 83.28% (2002/2404)

Same as take 2/3 but with gpt4
Accuracy at k=1: 77.75% (1869/2404)
Accuracy at k=2: 69.13% (1662/2404)
Accuracy at k=3: 75.00% (1803/2404)
Accuracy at k=5: 79.16% (1903/2404)
Accuracy at k=10: 83.32% (2003/2404)
Accuracy at k=20: 83.32% (2003/2404)
Accuracy at k=40: 83.32% (2003/2404)
============================================================
"""


"""On Embedding diffs

Experiment 1: Based on oracle model's response embedding (max, knows about correct output)
Accuracy at k=1: 84.94% (2042/2404)
Accuracy at k=2: 81.95% (1970/2404)
Accuracy at k=3: 86.31% (2075/2404)
Accuracy at k=5: 87.40% (2101/2404)
Accuracy at k=10: 87.98% (2115/2404)
Accuracy at k=20: 87.81% (2111/2404)
Accuracy at k=40: 87.35% (2100/2404)

FLAN-t5-small
Accuracy at k=1: 74.13% (1782/2404)
Accuracy at k=2: 67.05% (1612/2404)
Accuracy at k=3: 79.62% (1914/2404)
Accuracy at k=5: 82.24% (1977/2404)
Accuracy at k=10: 83.49% (2007/2404)
Accuracy at k=20: 83.69% (2012/2404)
Accuracy at k=40: 83.78% (2014/2404)

Mixtral
Accuracy at k=1: 75.37% (1812/2404)
Accuracy at k=2: 67.05% (1612/2404)
Accuracy at k=3: 79.37% (1908/2404)
Accuracy at k=5: 81.66% (1963/2404)
Accuracy at k=10: 83.28% (2002/2404)
Accuracy at k=20: 83.74% (2013/2404)
Accuracy at k=40: 83.86% (2016/2404)

GPT4
Accuracy at k=1: 80.62% (1938/2404)
Accuracy at k=2: 75.71% (1820/2404)
Accuracy at k=3: 83.24% (2001/2404)
Accuracy at k=5: 84.57% (2033/2404)
Accuracy at k=10: 85.40% (2053/2404)
Accuracy at k=20: 86.06% (2069/2404)
Accuracy at k=40: 85.73% (2061/2404)

"""

print("BEGINNING kNN")

# Number of nearest neighbors to consider (40 was best according to paper)
k_schedule = [
    # 1,
    # 2,
    # 3,
    # 5,
    # 10,
    # 20,
    # 40,
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
        # eval_embedding = eval_row["prompt_embedding"]
        # eval_embedding = eval_row["flan_s_response_embedding"]
        # eval_embedding = eval_row["mixtral_response_embedding"]
        # eval_embedding = eval_row["gpt4_response_embedding"]
        eval_embedding = eval_row["embedding_diff"]

        
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
================================================================
Repeat exp2 (cluster on input embeddings) with just English data, knows oracle model for cluster centers.
Prediction is done based on prompt embeddings in the eval set. This is similar to the experiment above 
but uses clustering rather than kNN
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 54.78% (1317/2404)
Accuracy at k=10: 53.16% (1278/2404)
Accuracy at k=20: 61.94% (1489/2404)
Accuracy at k=50: 72.46% (1742/2404)
Accuracy at k=100: 70.63% (1698/2404)
Accuracy at k=200: 72.46% (1742/2404)
Accuracy at k=500: 72.71% (1748/2404)
Accuracy at k=1000: 74.33% (1787/2404)

Take 2: Clustering on FLAN response embeddings
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 73.21% (1760/2404)
Accuracy at k=10: 73.38% (1764/2404)
Accuracy at k=20: 71.80% (1726/2404)
Accuracy at k=50: 72.25% (1737/2404)
Accuracy at k=100: 71.42% (1717/2404)
Accuracy at k=200: 71.55% (1720/2404)
Accuracy at k=500: 71.67% (1723/2404)
Accuracy at k=1000: 71.67% (1723/2404)

Take 3: mixtral, similar to take 2
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 83.28% (2002/2404)
Accuracy at k=10: 80.82% (1943/2404)
Accuracy at k=20: 79.33% (1907/2404)
Accuracy at k=50: 75.25% (1809/2404)
Accuracy at k=100: 72.63% (1746/2404)
Accuracy at k=200: 74.63% (1794/2404)
Accuracy at k=500: 74.67% (1795/2404)
Accuracy at k=1000: 73.88% (1776/2404)

Take 4: gpt4
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 83.28% (2002/2404)
Accuracy at k=10: 71.51% (1719/2404)
Accuracy at k=20: 70.72% (1700/2404)
Accuracy at k=50: 75.17% (1807/2404)
Accuracy at k=100: 74.08% (1781/2404)
Accuracy at k=200: 75.79% (1822/2404)
Accuracy at k=500: 76.66% (1843/2404)
Accuracy at k=1000: 76.12% (1830/2404)
================================================================
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
            [eval_embedding], [train_df.loc[prompt_idx]["flan_s_response_embedding"]]
            # [eval_embedding], [train_df.loc[prompt_idx]["mixtral_response_embedding"]]
            # [eval_embedding], [train_df.loc[prompt_idx]["gpt4_response_embedding"]]
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
        # cluster_points = np.stack(train_df["prompt_embedding"][cluster_indices])
        cluster_points = np.stack(train_df["flan_s_response_embedding"][cluster_indices])
        # cluster_points = np.stack(train_df["mixtral_response_embedding"][cluster_indices])
        # cluster_points = np.stack(train_df["gpt4_response_embedding"][cluster_indices])

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
        # eval_embedding = eval_row["prompt_embedding"]
        eval_embedding = eval_row["flan_s_response_embedding"]
        # eval_embedding = eval_row["mixtral_response_embedding"]
        # eval_embedding = eval_row["gpt4_response_embedding"]
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


"""
================================================================================
exp8 (cluster on differences between input embeddings and FLAN output embeddings

First try (Using FLAN small embeddings)
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 65.39% (1572/2404)
Accuracy at k=10: 34.32% (825/2404)
Accuracy at k=20: 50.04% (1203/2404)
Accuracy at k=50: 45.63% (1097/2404)
Accuracy at k=100: 76.71% (1844/2404)
Accuracy at k=200: 73.25% (1761/2404)
Accuracy at k=500: 72.59% (1745/2404)
Accuracy at k=1000: 76.12% (1830/2404)
Accuracy at k=4806: 74.92% (1801/2404)

Second try (Using embeddings on same exact dataset as above, but using embedding diff from oracle model's best response embedding)
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 83.28% (2002/2404)
Accuracy at k=10: 82.28% (1978/2404)
Accuracy at k=20: 82.03% (1972/2404)
Accuracy at k=50: 85.32% (2051/2404)
Accuracy at k=100: 82.32% (1979/2404)
Accuracy at k=200: 83.11% (1998/2404)
Accuracy at k=500: 83.90% (2017/2404)
Accuracy at k=1000: 83.86% (2016/2404)
Accuracy at k=4806: 84.90% (2041/2404)

Third try (same as first try but using Mixtral 7b 8x or whatever)
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 68.43% (1645/2404)
Accuracy at k=10: 65.72% (1580/2404)
Accuracy at k=20: 68.97% (1658/2404)
Accuracy at k=50: 74.29% (1786/2404)
Accuracy at k=100: 76.29% (1834/2404)
Accuracy at k=200: 76.91% (1849/2404)
Accuracy at k=500: 77.25% (1857/2404)
Accuracy at k=1000: 74.96% (1802/2404)
Accuracy at k=4806: 74.63% (1794/2404)

Last try (using best/most correct model) gpt-4-1106-preview
Accuracy at k=1: 83.28% (2002/2404)
Accuracy at k=5: 75.37% (1812/2404)
Accuracy at k=10: 80.74% (1941/2404)
Accuracy at k=20: 81.16% (1951/2404)
Accuracy at k=50: 78.99% (1899/2404)
Accuracy at k=100: 81.36% (1956/2404)
Accuracy at k=200: 77.75% (1869/2404)
Accuracy at k=500: 79.12% (1902/2404)
Accuracy at k=1000: 79.70% (1916/2404)
Accuracy at k=4806: 79.03% (1900/2404)

================================================================================
"""

print(f"BEGIN clustering on diffs with FLAN")


def predict_oracle_model_with_diff_cluster(embedding_diff):
    """
    Predict what oracle router would route to based on medoids for difference clusters

    Args:
        embedding_diff (numpy array): The difference between the prompt embedding and
            the FLAN reponsee mbedding of the evaluation example.

    Returns:
        dict: The oracle model and metadata of the closest medoid.
    """

    similarities = {}
    for cluster, prompt_idx in closest_prompt_idx_by_cluster.items():
        similarities[cluster] = cosine_similarity(
            [embedding_diff], [train_df.loc[prompt_idx]["embedding_diff"]]
        )[0, 0]

    # Find the cluster with the highest similarity
    best_cluster = max(similarities, key=similarities.get)
    best_prompt_idx = closest_prompt_idx_by_cluster[best_cluster]

    # Return the oracle model for the closest cluster
    return train_df.loc[best_prompt_idx]["oracle_model_to_route_to"]


# Get numpy arrays for the embedding diffs
embedding_diffs_train = np.stack(train_df["embedding_diff"])
embedding_diffs_val = np.stack(eval_df["embedding_diff"])


# Set the number of clusters in kMeans
k_schedule = [
    # 1,
    # 5,
    # 10,
    # 20,
    # 50,
    # 100,
    # 200,
    # 500,
    # 1000,
    # len(train_df) // 2,
]

for k in k_schedule:

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embedding_diffs_train)

    # Initialize variables to track performance
    total_predictions = 0
    correct_predictions = 0

    # Create new dictionary mapping each cluster label to the index in train_df of
    # the prompt in that cluster with the closest embedding to the cluster center
    closest_prompt_idx_by_cluster = {k_id: None for k_id in range(k)}

    # Loop through the dataframe and assign each prompt to a cluster_label
    train_df["cluster_label"] = kmeans.predict(embedding_diffs_train)

    # Find the closest prompt embedding to the cluster center for each cluster
    for cluster in range(k):

        # Get all points in the current cluster
        cluster_indices = train_df[train_df["cluster_label"] == cluster].index.values
        cluster_points = np.stack(train_df["embedding_diff"][cluster_indices])

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
        eval_embedding = eval_row["embedding_diff"]
        actual_oracle_model = eval_row["oracle_model_to_route_to"]

        # Predict the oracle model for the current evaluation row
        predicted_model = predict_oracle_model_with_diff_cluster(eval_embedding)

        # Check if the prediction matches the actual model
        if predicted_model == actual_oracle_model:
            correct_predictions += 1
        total_predictions += 1

    # Calculate and display accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(
        f"Accuracy at {k=}: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})"
    )
