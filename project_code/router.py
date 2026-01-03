import time
import json
import pandas as pd
from utils import (
    MODELS,
    get_model_response,
    compare_model_to_reference,
    get_output_from_response,
)

# Load the tasks dataset
with open("tasks.json", "r") as file:
    dataset = json.load(file)

# Create empty dataframe to record experiment data
columns = []
for model in MODELS:
    for metric_name in ["price", "latency", "judge_win_rate", "human_win_rate"]:
        columns.append(f"{model.name}:{metric_name}")
df = pd.DataFrame(columns=columns)

# Run experiment
for task in dataset:
    new_row = {}
    for model in MODELS:

        # Time how long it takes this model to answer the task prompt
        start = time.time()
        response = get_model_response(model, task["prompt"])
        end = time.time()

        # Get info about model response for task
        res_json = response.json()
        model_cost = (
            model.input_cost * res_json["usage"]["prompt_tokens"]
            + model.output_cost * res_json["usage"]["completion_tokens"]
        )
        model_latency = end - start

        # Compare model to reference model
        judge_win_rate, human_win_rate = compare_model_to_reference(
            candidate_model=model,
            candidate_model_output=get_output_from_response(response),
            prompt=task["prompt"],
        )

        # Save data
        new_row[f"{model.name}:price"] = model_cost
        new_row[f"{model.name}:latency"] = model_latency
        new_row[f"{model.name}:judge_win_rate"] = judge_win_rate
        new_row[f"{model.name}:human_win_rate"] = human_win_rate

    # df = df._append(new_row, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the results to a .csv file
    df.to_csv("results.csv", mode='a', index=False)
