import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    cosine_similarity,
    cosine_distances,
)  # SBERT (Sentence-BERT) was trained to minimize cosine distance

# Load the dataset
file_path = "cleaned_data_with_embeddings.pkl"
df = pd.read_pickle(file_path)

# Train-test split (30% eval, 70% train)
train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

# Stack the embeddings for clustering (you can also cluster them separately if desired)
train_embeddings = np.stack(
    train_df["prompt_embedding"]
)  # Use prompt embeddings or stack prompt + response embeddings
train_oracle_models = train_df["oracle_model_to_route_to"]

# Set the number of clusters
k_schedule = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
# k_schedule = [50, 75, 100, 200, 500, 1000]
# k_schedule = [len(train_df)//2]
"""
['Accuracy at k=1: 82.50% (3032/3675)',
'Accuracy at k=2: 82.50% (3032/3675)',
'Accuracy at k=3: 22.34% (821/3675)',
'Accuracy at k=5: 81.52% (2996/3675)',
'Accuracy at k=7: 79.76% (2931/3675)',
'Accuracy at k=10: 75.73% (2783/3675)',
'Accuracy at k=15: 77.12% (2834/3675)',
'Accuracy at k=20: 77.22% (2838/3675)',
'Accuracy at k=30: 76.22% (2801/3675)',
'Accuracy at k=40: 54.48% (2002/3675)',
['Accuracy at k=50: 54.26% (1994/3675)',
'Accuracy at k=75: 77.90% (2863/3675)',
'Accuracy at k=100: 79.59% (2925/3675)',
'Accuracy at k=200: 76.60% (2815/3675)',
'Accuracy at k=500: 78.56% (2887/3675)',
'Accuracy at k=1000: 79.21% (2911/3675)',
'Accuracy at k=4287: 79.86% (2935/3675)]
"""


exp_output = []
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
    prompt_embedding_np = np.stack(train_df["prompt_embedding"].to_numpy())
    train_df["cluster_label"] = kmeans.predict(prompt_embedding_np)

    # Find the closest prompt embedding to the cluster center for each cluster
    for cluster in range(k):

        # Get all points in the current cluster
        cluster_indices = train_df[train_df["cluster_label"] == cluster].index.values
        cluster_points = np.stack(train_df["prompt_embedding"][cluster_indices])

        # Compute pairwise cosine distances within the cluster to the center
        distances = cosine_distances(
            cluster_points, [kmeans.cluster_centers_[cluster]]
        ).flatten()

        # Find the index of the medoid (point with minimum sum of distances to other points)
        medoid_idx_in_cluster = np.argmin(distances)
        medoid_idx_global = cluster_indices[medoid_idx_in_cluster]

        closest_prompt_idx_by_cluster[cluster] = medoid_idx_global

    # Define the function to predict based on medoids
    def predict_oracle_model(eval_embedding):
        """
        Predict the best oracle model based on the closest medoid.

        Args:
            eval_embedding (numpy array): The embedding of the evaluation example.
            medoids (dict): A dictionary where each key is a cluster and each value contains the medoid embedding,
                            oracle model, and additional metadata.

        Returns:
            dict: The oracle model and metadata of the closest medoid.
        """

        # Compute cosine similarity between eval_embedding and each medoid
        # test_prompt_idx = closest_prompt_idx_by_cluster[0]

        # raise RuntimeError(f'{eval_embedding.shape=}, {train_df.iloc[test_prompt_idx]["prompt_embedding"]=}')
        # raise RuntimeError(f'{cosine_similarity([eval_embedding], [train_df.iloc[test_prompt_idx]["prompt_embedding"]])=}')
        similarities = {}
        for cluster, prompt_idx in closest_prompt_idx_by_cluster.items():
            similarities[cluster] = cosine_similarity(
                [eval_embedding], [train_df.loc[prompt_idx]["prompt_embedding"]]
            )[0, 0]

        # Find the cluster with the highest similarity
        best_cluster = max(similarities, key=similarities.get)
        best_prompt_idx = closest_prompt_idx_by_cluster[best_cluster]

        # Return the oracle model for the closest cluster
        return train_df.loc[best_prompt_idx]["oracle_model_to_route_to"]

    # Loop through each row in the evaluation set
    for _, eval_row in eval_df.iterrows():
        eval_embedding = eval_row["response_embedding"]
        actual_oracle_model = eval_row["oracle_model_to_route_to"]

        start = time.time()
        # Predict the oracle model for the current evaluation row
        predicted_model = predict_oracle_model(eval_embedding)
        end = time.time()

        # Check if the prediction matches the actual model
        if predicted_model == actual_oracle_model:
            correct_predictions += 1
        total_predictions += 1

        # print(
        #     f"Processed eval row: Predicted={predicted_model}, Actual={actual_oracle_model}. Took {end-start} seconds."
        # )

    # Calculate and display accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    exp_output.append(
        f"Accuracy at {k=}: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})"
    )

print(exp_output)
