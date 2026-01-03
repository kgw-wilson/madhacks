# For exp8, adds flan responses for each prompt and embedding for that response

import time
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load embedding model
embedding_model_name = "all-MiniLM-L6-v2"  # Takes about 0.15 - 0.3 seconds on CPU, not bad
embedding_model = SentenceTransformer(embedding_model_name)


# Load the datasets
file_path = "rb_data_with_flan.pkl"
df = pd.read_pickle(file_path)
full_rb_df = pd.read_csv("routerbench_output.csv")

# Initialize columns for embeddings
new_res_col = "gpt4_response"
new_emb_col = new_res_col + "_embedding"
df[new_res_col] = None
df[new_emb_col] = None

# Loop through each row in the DataFrame
for index, row in df.iterrows():

    # Find the mixtral response from the full routerbench_output df
    full_row = full_rb_df.loc[full_rb_df["sample_id"] == row["sample_id"]].iloc[0]

    # Extract the input prompt and mixtral model's response
    raw_response = full_row["gpt-4-1106-preview|model_response"]  # full_row["mistralai/mixtral-8x7b-chat|model_response"]  # In weird string/list format
    response_list = ast.literal_eval(raw_response)
    mixtral_response = " ".join(response_list)
    mixtral_response_embedding = embedding_model.encode(mixtral_response)

    # Save
    df.at[index, new_res_col] = mixtral_response
    df.at[index, new_emb_col] = mixtral_response_embedding


# Save the updated DataFrame
output_file_path = "rb_data_with_flan.pkl"
df.to_pickle(output_file_path)
print(f"Updated dataset saved to {output_file_path}")


# Question: if we cluster on the differences between input and output embeddings, when we get a 
# new prompt, we can just encode the prompt and run the prompt through FLAN to get the output embedding?