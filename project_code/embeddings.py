"""
Decide on task description model (i.e. LLAMA, fast claude, etc.)
Decide on embedding model (i.e. DistilBERT, AlBERT, etc.)
Perform clustering, compare k-means and DBSCAN (adaptive clustering?)
Record prompts for cluster centers
"""

import time
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from utils import Model, M_TOKENS, get_model_response, get_output_from_response

example_prompt = "Please summarize the following article into 3-5 key bullet points:\n\n'The global economy is showing signs of recovery as markets rebound and consumer confidence increases. Experts attribute this to successful vaccination programs and government stimulus packages. However, concerns remain over potential inflation and supply chain disruptions."

_TASK_FROM_INPUT_PROMPT =  """
Given a prompt, describe the main task it is trying to achieve.
Be concise, only output the task description.

Input Example: Determine the sentiment (positive, neutral, negative) of the following customer review:\n\n'I recently tried the new model and while it has some great features, it didn't meet all my expectations.

Output Example: Sentiment analysis of customer review

Input: {user_input}

Output
"""

# mistral tiny does not do well, takes a long time
# google/gemini-flash-1.5-8b is 0.5-0.6 seconds
# meta-llama/llama-3.2-1b-instruct is 0.4-0.6 seconds
# _TASK_FROM_INPUT_MODEL = Model(
#     name="meta-llama/llama-3.2-1b-instruct", # fast, lightweight model
#     input_cost=1.0 / M_TOKENS,
#     output_cost=2.0 / M_TOKENS,
# )
# start = time.time()
# example_task_from_prompt_response = get_model_response(_TASK_FROM_INPUT_MODEL, _TASK_FROM_INPUT_PROMPT.format(user_input=example_prompt))
# end = time.time()
# print(f"example_task_from_prompt_response took {end-start} seconds.")
# example_task_from_prompt = get_output_from_response(example_task_from_prompt_response)

model_name = "google/flan-t5-small" # too small # too big: "meta-llama/Llama-2-7b-chat-hf"

# Specify model name
# model_name = "EleutherAI/gpt-neo-1.3B"

# Load tokenizer and model
cache_dir = "./model_cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=cache_dir,
#     device_map="auto",  # This ensures the model is loaded to the available device (GPU/CPU)
#     # quantization_config=BitsAndBytesConfig(load_in_8bit=True)
# )

model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # For FLAN model

# Perform random generation to get model warmed up and everything loaded
model.generate(**tokenizer("hi", return_tensors="pt"), max_new_tokens=1)

raise RuntimeError

user_input = _TASK_FROM_INPUT_PROMPT.format(user_input=example_prompt)
start = time.time()
inputs = tokenizer(user_input, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=25)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
end = time.time()
print(f"response from prompt->task model: {response}, took {end-start} seconds")

# raise RuntimeError(f"{outputs=}")

# Load embedding model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Generate embedding for task
start = time.time()
inputs = tokenizer(user_input, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
end = time.time()
print(f"Getting embeddings took {end-start} seconds")
print(embeddings.shape)


"""
Questions:

How much is extra 'task' step better? How much better is clustering in that space versus prompt space?
How long does it take to find closest 5 embeddings?

Data cleaning
- Remove all prompts where best model is not available to us
- Remove all prompts where no model was correct

Generate prompt embeddings
Generate task embeddings
Measure 


4 data

So big improvements
- Moving to user data (Dolly, ShareGPT datasets)
- Offline clustering 


Summary of other options explored
- Task classifier for an incoming prompt
    Downsides are that it's inflexible to changes in the incoming prompts
    Hard to come up with classifications for all possible categories
- Nearest neighbors search on fixed dataset
    Downsides are that it's heavily dependent on the datastore, and the 
    datastore is hard to update because each example in there needs to be
    run on all candidate modles. Hard/impossible to update for new models
- Model performance benchmarking: evaluate the performance of each model on 
    each benchmark and route requests to the model that did the best on the
    associated benchmark. Extensive amount of work, static in nature, can
    overfit.
- Cascading Routing: expensive, routes one requests to multiple LLMs
- Eagle: maintains ELO score, updates routing based on recent usage
- RouteLLM: strong and weak model, routes between them


What's next for paper presentation:
Look at the experimental results, save those - DONE
Make slides

Project could involve running latency analysis of kNN with 40 neighbors on dataset, probably bad
Project could involve running their kNN system on our new examples

For project:
Get embeddings for all cleaned_data
Perform scaling analysis for approx NN search

Create the dataset for human rankings on all of our 25 tasks
Get routerbench output "oracle_model_to_route_to" for all tasks
Get our system's output
- Our system should be based on clustering on dataset of prompt embeddings
    for our new user-based dataset

"""