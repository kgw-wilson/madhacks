import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # SBERT (Sentence-BERT) was trained to minimize cosine distance

# Load SBERT model
# from sentence_transformers import SentenceTransformer
# model_name = "all-MiniLM-L6-v2"  # Takes about 0.02 - 0.1 seconds on CPU, not bad
# model = SentenceTransformer(model_name)

# Load the dataset
file_path = "cleaned_data_with_embeddings.pkl"
df = pd.read_pickle(file_path)

# Train-test split (30% eval, 70% train)
train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

# Number of nearest neighbors to consider
# Accuracy at 40: 82.75%
# Accuracy at 20: 82.72%
# Accuracy at 10: 82.61%
# Accuracy at 5: 81.52%
# Accuracy at 3: 80.84%
# Accuracy at 2: 76.93%
# Accuracy at 1: 79.70%
k = 1

# Initialize variables to track performance
total_predictions = 0
correct_predictions = 0


def predict_oracle_model(eval_embedding, train_embeddings, train_oracle_models):
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


# Prepare the training embeddings and oracle models
train_embeddings = np.stack(train_df["prompt_embedding"])
train_oracle_models = train_df["oracle_model_to_route_to"]

# Loop through each row in the evaluation set
for _, eval_row in eval_df.iterrows():
    eval_embedding = eval_row["response_embedding"]
    actual_oracle_model = eval_row["oracle_model_to_route_to"]

    start = time.time()
    # Predict the oracle model for the current evaluation row
    predicted_model = predict_oracle_model(
        eval_embedding, train_embeddings, train_oracle_models
    )
    end = time.time()

    # Check if the prediction matches the actual model
    if predicted_model == actual_oracle_model:
        correct_predictions += 1
    total_predictions += 1

    print(
        f"Processed eval row: Predicted={predicted_model}, Actual={actual_oracle_model}. Took {end-start} seconds."
    )

# Calculate and display accuracy
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})")
