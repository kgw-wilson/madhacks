"""Notes from Fred, 1 is kNN (them), 2 is kMeans (us), 3 is diffs (us)

1. Lots of fast kNN search -> great, real drawback is that pre-defined tasks and model performance is inflexible and expensive

2. Cutting down, treat. Could be a bug. Model performance in a cluster changes. Could measure accuracy as you get farther from the cluster center? 

Geometry of space?

3. Take the average to find best prompt for cluster, play around with that though, depends on space geometry


Look at relative accuracy as you move around in the space, very smooth in that space or nonsmooth?

Predictive router should be close to zero router. Not obvious what the comparision should be for predictive routers
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
diff_embeddings = np.stack(
    train_df["embedding_diff"]
)

k_schedule = [100]

exp_outputs = []

for k in k_schedule:

    # Perform KMeans clustering on the embedding differences
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(diff_embeddings)

    # Train classifier to map input embeddings to diff clusters
    # Example usage

    input_size = 384 # Embedding size of all-MiniLM-L6-v2
    hidden_layers = [1024] * 7

    """Results

    [100] * 2
    Saving. Epoch 1/100, Loss: 3.9555, Eval Accuracy: 0.2027
    Saving. Epoch 11/100, Loss: 1.4730, Eval Accuracy: 0.5208
    Saving. Epoch 21/100, Loss: 1.2961, Eval Accuracy: 0.5347

    ---

    [256, 128]
    Saving. Epoch 1/100, Loss: 3.7120, Eval Accuracy: 0.2525
    Saving. Epoch 11/100, Loss: 1.3738, Eval Accuracy: 0.5295
    Saving. Epoch 21/100, Loss: 1.1774, Eval Accuracy: 0.5358

    ---

    [1024] * 7
    Saving. Epoch 1/100, Loss: 3.9455, Eval Accuracy: 0.0454
    Saving. Epoch 11/100, Loss: 1.3923, Eval Accuracy: 0.4846
    Saving. Epoch 21/100, Loss: 1.0357, Eval Accuracy: 0.5012
    
    """

    output_size = k

    X_train = np.stack(train_df["prompt_embedding"])
    y_train = kmeans.predict(diff_embeddings)

    # Clusters exist in difference space, train MLP to map prompt embeddings to clusters
    X_eval = np.stack(eval_df["prompt_embedding"])
    eval_diff = np.stack(eval_df["prompt_embedding"] - eval_df["response_embedding"])
    y_eval = kmeans.predict(eval_diff)

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