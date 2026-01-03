import time
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the dataset
file_path = "cleaned_data_with_embeddings.pkl"
df = pd.read_pickle(file_path)

print(f'{df["prompt"][:5]=}')
print(f'{type(df["prompt"])=}')
print(f'{len(df["prompt"])=}')
print(f'{type(list(df["prompt"]))=}')
print(f'{len(list(df["prompt"]))=}')

raise RuntimeError(f'{list(df["prompt"])[0]=}')

# Initialize columns for embeddings
df["prompt_embedding"] = None
df["response_embedding"] = None

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    # Extract the input prompt and oracle model's response
    prompt = row["prompt"]
    oracle_model = row["oracle_model_to_route_to"]
    response_column = f"{oracle_model}|model_response"
    
    # Handle potential missing values or inconsistencies
    if response_column not in df.columns or pd.isna(row[response_column]):
        print(f"Skipping row {index} due to missing response for oracle model: {oracle_model}")
        continue

    response = row[response_column]

    # Generate embeddings
    try:
        start = time.time()
        prompt_embedding = model.encode(prompt)
        response_embedding = model.encode(response)
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