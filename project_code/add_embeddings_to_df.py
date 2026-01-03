import time
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load SBERT model
model_name = "all-MiniLM-L6-v2"  # Takes about 0.15 - 0.3 seconds on CPU, not bad
model = SentenceTransformer(model_name)

# Load the dataset
file_path = "cleaned_data.csv"  # Path to your CSV file
df = pd.read_csv(file_path)

# Initialize columns for embeddings
df["prompt_embedding"] = None
df["response_embedding"] = None

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    # Extract the input prompt and oracle model's response
    raw_prompt = row["prompt"]  # In weird string/list format
    prompt_list = ast.literal_eval(raw_prompt)
    clean_prompt = " ".join(prompt_list)

    oracle_model = row["oracle_model_to_route_to"]
    response_column = f"{oracle_model}|model_response"

    # Handle potential missing values or inconsistencies
    if response_column not in df.columns or pd.isna(row[response_column]):
        print(
            f"Skipping row {index} due to missing response for oracle model: {oracle_model}"
        )
        continue

    raw_response = row[response_column]
    model_response_list = ast.literal_eval(raw_response)
    clean_response = " ".join(model_response_list)

    response = row[response_column]

    # Generate embeddings
    try:
        start = time.time()
        prompt_embedding = model.encode(clean_prompt)
        response_embedding = model.encode(clean_response)
        end = time.time()

        print(f"Processed row {index} in {end - start:.2f} seconds.")

        # Update DataFrame with embeddings
        df.at[index, "prompt_embedding"] = prompt_embedding
        df.at[index, "response_embedding"] = response_embedding

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        continue

# Save the updated DataFrame
output_file_path = "cleaned_data_with_embeddings.pkl"
df.to_pickle(output_file_path)
print(f"Updated dataset saved to {output_file_path}")
