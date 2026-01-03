import pandas as pd

# This is cleaning the 0-shot dataset from
# https://huggingface.co/datasets/withmartian/routerbench

# The models from the RouterBench paper we can access through OpenRouter
known_model_names = [
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-instruct",
]

# Define the list of substrings to search for
substrings = ["sample_id", "prompt", "eval_name", "oracle_model_to_route_to"] + known_model_names

# Create a regex pattern that matches any of the substrings
pattern = "|".join(substrings)

df = pd.read_csv("routerbench_output.csv")
filtered_df = df.loc[:, df.columns.str.contains(pattern)]

# Filter rows where 'oracle_model_to_route_to' is among 'known_model_names'
result_df = filtered_df[filtered_df['oracle_model_to_route_to'].isin(known_model_names)]

# Save the resulting DataFrame (len=12250)
result_df.to_csv("cleaned_data.csv", index=False)