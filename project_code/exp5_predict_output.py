"""This is a new approach, given a prompt, we try to predict the model output embedding.

Note that the embeddings included the brackets and extra quotes, but that was cleaned up
for exp5.

Training loss is really good, which shows the concept is there. Validation loss is high
which means the trianing needs to be tuned and we need more data

70-30 train eval split:
Saving. Eval loss: 0.3482
Epoch 1/5, Loss: 0.5320.
Epoch 2/5, Loss: 0.2697.
Epoch 3/5, Loss: 0.1700.
Epoch 4/5, Loss: 0.1167.
Epoch 5/5, Loss: 0.0849.

---

80-20 train eval split:
Saving. Eval loss: 0.3157
Epoch 1/5, Loss: 0.4988.
Epoch 2/5, Loss: 0.2405.
Epoch 3/5, Loss: 0.1485.
Epoch 4/5, Loss: 0.1002.
Epoch 5/5, Loss: 0.0721.

---

Since the validation loss is high, I'll try adding more prompt/output embedding pairs from
other datasets.

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
file_path = "cleaned_data_with_embeddings.pkl"
df = pd.read_pickle(file_path)

# Train-test split (30% eval, 70% train) is what's used in the paper for RouterBench
# train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42) # training an encoder 20-80 is probably better


# Hyperparameters
batch_size = 32
learning_rate = 1e-4
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# Clean prompt data
train_prompts = []
for i, row in train_df.iterrows():
    raw_prompt = row["prompt"]  # In weird string/list format
    prompt_list = ast.literal_eval(raw_prompt)
    clean_prompt = " ".join(prompt_list)
    train_prompts.append(clean_prompt)
eval_prompts = []
for i, row in eval_df.iterrows():
    raw_prompt = row["prompt"]  # In weird string/list format
    prompt_list = ast.literal_eval(raw_prompt)
    clean_prompt = " ".join(prompt_list)
    eval_prompts.append(clean_prompt)

# Create dataloaders
train_output_embeddings = train_df["prompt_embedding"]
train_dataset = FinetuningDataset(train_prompts, np.stack(train_output_embeddings))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_output_embeddings = eval_df["prompt_embedding"]
eval_dataset = FinetuningDataset(eval_prompts, np.stack(eval_output_embeddings))
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


# Train and evaluate
model = EmbeddingFineTuner(
    pretrained_model_name="all-MiniLM-L6-v2",  # same as in RouterBench and other experiments
    save_path=os.path.join("exp5_models", "finetuned_embedder.ckpt"),
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
    epochs=5,
    device="cpu",
    eval_every=10,
    max_evals_without_improvement=5,
)
