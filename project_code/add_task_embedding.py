# For exp8, adds flan responses for each prompt and embedding for that response

import time
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load embedding model
embedding_model_name = (
    "all-MiniLM-L6-v2"  # Takes about 0.15 - 0.3 seconds on CPU, not bad
)
embedding_model = SentenceTransformer(embedding_model_name)


# Load the datasets
file_path = "rb_data_with_flan_and_task.pkl"
df = pd.read_pickle(file_path)
df = df[df["task_description"].notna()]

# Initialize columns for embeddings
new_emb_col = "task_description_embedding"
df[new_emb_col] = None

# Loop through each row in the DataFrame
for index, row in df.iterrows():

    task = row["task_description"]
    if task:
        task_embedding = embedding_model.encode(task)
        df.at[index, new_emb_col] = task_embedding


# Save the updated DataFrame
output_file_path = "rb_data_with_flan_and_task.pkl"
df.to_pickle(output_file_path)
print(f"Updated dataset saved to {output_file_path}")
