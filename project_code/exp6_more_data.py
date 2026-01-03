"""Notes

access token: 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

Each example has these keys
['conversation_id', 'model', 'conversation', 'turn', 'language', 'openai_moderation', 'redacted']

Every example matches this criteria
example["conversation"][0]["role"] == "user" and  example["conversation"][1]["role"] == "assistant":

Create dataset of 
model, prompt, response, prompt_embedding, response_embedding
"""

from datasets import load_from_disk
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load SBERT model
model_name = "all-MiniLM-L6-v2"  # Takes about 0.15 - 0.3 seconds on CPU, not bad
model = SentenceTransformer(model_name)

ds = load_from_disk("./lmsys-data")

# List to accumulate data
rows = []

for example in ds["train"]:  # There's only a train split

    if example["language"] != "English":  # skip non-english languages for now
        continue

    prompt = example["conversation"][0]["content"]
    response = example["conversation"][1]["content"]

    prompt_embedding = model.encode(prompt)
    response_embedding = model.encode(response)

    rows.append(
        {
            "model": example["model"],
            "prompt": prompt,
            "response": response,
            "prompt_embedding": prompt_embedding,
            "response_embedding": response_embedding,
        }
    )

    if len(rows) % 100 == 0:
        print(f"{len(rows)=}")

df = pd.DataFrame(rows)
output_path = "lmsys-embeddings.pkl"
df.to_pickle(output_path)
