"""This is different from exp3 because now we're not training to predict the cluster in the difference space but
rather the cluster in the output space. Should expect to see slightly higher accuracy than in exp3. 

Overall, the results are slightly better than in exp3, which is what I expected.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    cosine_similarity,
    cosine_distances,
) 
from mlp_model import MLP

# Load the dataset
file_path = "cleaned_data_with_embeddings.pkl"
df = pd.read_pickle(file_path)

# Train-test split (30% eval, 70% train)
train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

# Make new column in df, embedding_diff
train_df["embedding_diff"] = train_df["prompt_embedding"] - train_df["response_embedding"]
response_embeddings = np.stack(
    train_df["response_embedding"]
)

k_schedule = [100]

exp_outputs = []

for k in k_schedule:

    # Perform KMeans clustering on the embedding differences
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(response_embeddings)

    # Train classifier to map input embeddings to diff clusters
    # Example usage

    input_size = 384 # Embedding size of all-MiniLM-L6-v2
    hidden_layers = [1024] * 7

    """Results at k = 100

    [100] * 2
    Saving. Epoch 1/100, Loss: 2.8476, Eval Accuracy: 0.5254
    Saving. Epoch 11/100, Loss: 1.6489, Eval Accuracy: 0.5668

    ---

    [256, 128]
    Saving. Epoch 1/100, Loss: 2.6263, Eval Accuracy: 0.5347
    Saving. Epoch 11/100, Loss: 1.5958, Eval Accuracy: 0.5698

    ---

    [1024] * 7
    Saving. Epoch 1/100, Loss: 2.6156, Eval Accuracy: 0.3151
    Saving. Epoch 11/100, Loss: 1.4814, Eval Accuracy: 0.5284
    Saving. Epoch 51/100, Loss: 0.9573, Eval Accuracy: 0.5306

    """

    output_size = k

    X_train = np.stack(train_df["prompt_embedding"])
    y_train = kmeans.predict(response_embeddings)

    # Train MLP to map prompt embeddings to output clusters
    X_eval = np.stack(eval_df["prompt_embedding"])
    eval_output_embeddings = np.stack(eval_df["response_embedding"])
    y_eval = kmeans.predict(eval_output_embeddings)

    # Initialize and train the model
    save_path = f"models/{'-'.join([str(l) for l in hidden_layers])}_at_k_{k}.ckpt"
    if os.path.isfile(save_path):
        raise RuntimeError(f"File already exists at {save_path=}")
    
    model = MLP(input_size, hidden_layers, output_size, save_path=save_path)
    model.train_model(X_train, y_train, X_eval, y_eval, epochs=100, batch_size=64)

    # Record experiment output
    exp_outputs.append(f"Fit!")

print(exp_outputs)


"""
Next:
Try classifier to predict output embeddings instead of difference
Finetune SBERT to classify based on clusters
"""