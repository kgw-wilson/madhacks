"""
Add Gpt-4o
"""

API_KEY = ""

import ast
from openai import OpenAI
import pandas as pd

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=API_KEY
)
system_msg = """
You are a helpful summarization agent. Any time you receive a prompt, you do not do what the prompt 
is asking you to do. Instead, you describe in 1 setence the task being described by the prompt. Start
with a verb.

Example output: "Answer a multiple choice question about the effects of parasites"
Example output: "Generate a title for a paper about solar cells based on its abstract"
"""

df = pd.read_pickle("rb_data_with_flan.pkl")
# Filter based on what was excluded for exp8
df = df[(df["flan_s_response"].notna()) & (df["flan_s_response_embedding"].notna())]

for i, row in df.iterrows():
    try:

        raw_prompt = row["prompt"]
        response_list = ast.literal_eval(raw_prompt)
        prompt = " ".join(response_list)
        content = row["prompt"]
        response = client.chat.completions.create(
            # model="gpt-4o-mini",
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg.strip()},
                {"role": "user", "content": prompt.strip()},
            ],
        )
        task_description = response.choices[0].message.content
        print(f"{i=} {task_description}")
        df.at[i, "task_description"] = task_description

    except Exception as e:
        df.to_pickle("rb_data_with_flan_and_task.pkl")
        print("An error occurred:", e)
        break

print("COMPLETED")
df.to_pickle("rb_data_with_flan_and_task.pkl")
