"""Given a prompt, we try to predict the model output embedding with a model trained on 
English language examples from lmsys-chat-1m
"""

import os
import ast
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    cosine_similarity,
    cosine_distances,
)
from mini_lm_model import EmbeddingFineTuner, train_model, FinetuningDataset, CosineLoss
from transformers import AdamW
from torch.utils.data import DataLoader


# Load the dataset
file_path = "lmsys-embeddings.pkl"
df = pd.read_pickle(file_path)

# Train-test split (30% eval, 70% train) is what's used in the paper for RouterBench
# train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)
train_df, eval_df = train_test_split(
    df, test_size=0.2, random_state=42
)  # training an encoder 20-80 is probably better


# Cut down datasize to test code
fraction_to_keep = 0.5  # 0.5%

"""RESULTS

-- fraction_to_keep == 0.01:
Saving. Eval loss: 0.6930
Epoch 1/5, Loss: 0.8269.
Saving. Eval loss: 0.5953
Epoch 2/5, Loss: 0.6217.
Saving. Eval loss: 0.5460
Epoch 3/5, Loss: 0.5475.
Saving. Eval loss: 0.5128
Epoch 4/5, Loss: 0.5019.
Saving. Eval loss: 0.4904
Epoch 5/5, Loss: 0.4704.

-- fraction_to_keep == 0.1:
Saving. Eval loss: 0.4137
Epoch 1/5, Loss: 0.5198.
Saving. Eval loss: 0.3756
Epoch 2/5, Loss: 0.3879.
Saving. Eval loss: 0.3613
Epoch 3/5, Loss: 0.3643.
Saving. Eval loss: 0.3544
Epoch 4/5, Loss: 0.3539.
Saving. Eval loss: 0.3502
Epoch 5/5, Loss: 0.3479.

-- fraction_to_keep == 0.5:
Saving. Eval loss: 0.3500
Epoch 1/5, Loss: 0.3978.
Saving. Eval loss: 0.3393
Epoch 2/5, Loss: 0.3434.
Saving. Eval loss: 0.3357
Epoch 3/5, Loss: 0.3370.
Saving. Eval loss: 0.3332
Epoch 4/5, Loss: 0.3338.
Saving. Eval loss: 0.3317
Epoch 5/5, Loss: 0.3318.
"""




train_df = train_df.sample(frac=fraction_to_keep, random_state=42)
eval_df = eval_df.sample(frac=fraction_to_keep, random_state=42)


# Hyperparameters
batch_size = 32
learning_rate = 1e-4
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_every = 1
max_evals_without_improvement = 5


# Create dataloaders
train_output_embeddings = train_df["response_embedding"]
train_dataset = FinetuningDataset(
    list(train_df["prompt"]), np.stack(train_output_embeddings)
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_output_embeddings = eval_df["response_embedding"]
eval_dataset = FinetuningDataset(list(eval_df["prompt"]), np.stack(eval_output_embeddings))
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


# Train and evaluate
model = EmbeddingFineTuner(
    pretrained_model_name="all-MiniLM-L6-v2",  # same as in RouterBench and other experiments
    save_path=os.path.join("exp7_models", f"e{epochs}_finetuned_embedder.ckpt"),
    device=device,
    output_dim=384,
    hidden_dim=384,
)
criterion = CosineLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

train_model(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    criterion,
    epochs=epochs,
    device=device,
    eval_every=eval_every,
    max_evals_without_improvement=max_evals_without_improvement,
)


"""
What to try next:

See if running smaller model on prompt, getting embedding, and routing to cluster based on that 
- This would be similar to exp2 except clustering isn't done on the prompt embeddings but instead on output 
embeddings made by FLAN and passed through mini-l6-v2. 
- If I do this on lmsys data, I'd need a baseline, that would have to be clustering on the input embeddings there

Train instruction-tuned model to output embeddings instead of tokens, see if that reduces validation loss compared to exp5 and exp7

"""