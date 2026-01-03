# For exp8, adds flan responses for each prompt and embedding for that response

import time
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load embedding model
embedding_model_name = "all-MiniLM-L6-v2"  # Takes about 0.15 - 0.3 seconds on CPU, not bad
embedding_model = SentenceTransformer(embedding_model_name)

# Load FLAN model
response_model_name = "google/flan-t5-xl"
cache_dir = "./model_cache"
tokenizer = AutoTokenizer.from_pretrained(response_model_name, cache_dir=cache_dir)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(response_model_name, cache_dir=cache_dir)

# Load the dataset
file_path = "lmsys-embeddings.pkl"
df = pd.read_pickle(file_path)

# Initialize columns for embeddings
new_res_col = "flan_s_response"
new_emb_col = new_res_col + "_embedding"
df[new_res_col] = None
df[new_emb_col] = None

# Loop through each row in the DataFrame
for index, row in df.iterrows():

    # Exclusion criteria - FLAN does not appear to do well on Chinese characters, could be something wrong with the tokenizer too
    if "chinese" in row["eval_name"].lower():
        continue

    # Extract the input prompt and oracle model's response
    raw_prompt = row["prompt"]  # In weird string/list format
    prompt_list = ast.literal_eval(raw_prompt)
    clean_prompt = " ".join(prompt_list)

    # Run the FLAN model on the prompt
    outputs = flan_model.generate(**tokenizer(clean_prompt, return_tensors="pt"), max_new_tokens=100)
    flan_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"{index=} : {clean_prompt[:5]=} : {flan_response[:5]=}")
    flan_response_embedding = embedding_model.encode(flan_response)

    # Save
    df.at[index, new_res_col] = flan_response
    df.at[index, new_emb_col] = flan_response_embedding


# Save the updated DataFrame
output_file_path = "rb_data_with_flan.pkl"
df.to_pickle(output_file_path)
print(f"Updated dataset saved to {output_file_path}")


# Question: if we cluster on the differences between input and output embeddings, when we get a 
# new prompt, we can just encode the prompt and run the prompt through FLAN to get the output embedding?