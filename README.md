# Intelligent Routing System for Language Models

## Project Overview

This project creates an advanced router to direct user prompts to the optimal Large Language Model (LLM) based on performance, latency, and cost considerations. The router analyzes each prompt, generates a task description using a language model, and selects the best model through a refined ranking system.

## Process

Prompt Analysis: Each user prompt is processed by a language model that distills the task into a concise description.
Task Embedding: This description is converted to an embedding via a lightweight BERT model (e.g., DistilBERT or ALBERT).
Clustering: Task embeddings are clustered using methods like DBSCAN or k-means to match prompts with their nearest cluster centers.
Model Selection: Based on user preferences, the router selects the top model for each cluster by evaluating performance, latency, and cost rankings.
Dynamic Updating: New task embeddings trigger clustering updates. Models are re-ranked by a judge model compared against a reference model, capturing costs and latencies.

## LLM Ranking Methodology

Prompt Dataset: A diverse prompt dataset is compiled across 25 distinct tasks.
Model Testing: Using OpenRouter, five models (Claude-2, ChatGPT-3.5-turbo, ChatGPT-4, Meta-LLaMA-34B, and Meta-LLaMA-2-70B) are evaluated for all prompts.
Rank Scoring: Each model’s performance is evaluated by LLM judges and human raters to establish unbiased rankings. To mitigate position bias, each judge model evaluates two LLM outputs six times per task, disregarding ties.

## Advancements Beyond Existing Methods

This system extends current LLM routing approaches by incorporating dynamic embedding-based clustering with an adaptive model-ranking mechanism. This setup continuously refines model selection as new tasks emerge, optimizing user-specific parameters in real time. This combination of task clustering, embedding adaptation, and regular performance re-evaluation enables the router to support more nuanced, responsive model deployment across a wide range of use cases.

## Results

- Compiled a dataset of 25 realworld LLM applications by hand
- Generated system for getting human rankings and judge LLM rankings for candidate models on each task
- Created prompt -> task embedding pipeline that runs in 0.38 (describe task being performed for a prompt) + 0.06 (generate task embedding) = 0.44 seconds total ON A CPU. This pipeline running on a GPU would be both accurate and extremely fast.


## Requirements
- Python 3.11 (recommended via `pyenv`)
- `pip`

## Setup

To get started, please clone the repo and `cd` into that directory.

```bash
# Ensure correct Python version (uses .python-version)
pyenv install 3.11.9
pyenv local 3.11.9

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


## Related concepts


1) Closest Concept: Task/Delta Embeddings

Even if they don’t use the exact subtraction idea, several papers consider how task information arises from changes between input and output:

🔹 Task2Vec (Achille et al., 2019)
	•	Represents a task by how a probe network’s parameters shift when learning that task.
	•	Intuitively similar: Task is defined by how the output distribution differs from what’s expected.
	•	Not embedding subtraction, but conceptually captures difference/change as a signal.

👉 Your idea (response − prompt) is in the same spirit because it focuses on what changes between input and output.

⸻

🔹 Contrastive Task Embeddings

Some meta-learning papers use contrastive learning to encode tasks:
	•	Encode tasks by differences in how models perform on them.
	•	Goal: embed tasks such that similar tasks cluster.

Not exactly subtraction, but difference becomes meaningful in the embedding space.

⸻

🧠 2) Prompt vs Response Embedding Interaction

Rather than direct subtraction, many papers look at joint representations of input and output:

🔹 Encoder–Decoder Interaction Embeddings
	•	Papers on sequence tagging / translation quality estimation sometimes train embeddings of (input, candidate output) pairs.
	•	Typically they:
	•	Concatenate input and output
	•	Pass through a joint encoder
	•	Train on quality/performance labels

This is closely related to what you want, but:
	•	They learn a joint representation, not a naive arithmetic difference.

Examples worth reading:
	•	“Learning joint representations of source and translation”
	•	“Estimate translation quality without references” (Quality Estimation field)

⸻

🧠 3) Latent Alignment & Delta Features

There are works in representation learning that treat difference vectors as meaningful

🔹 Word2Vec Analogues

In word embeddings:
	•	Vector offsets correspond to analogies (e.g., king - man + woman ≈ queen)
	•	This suggests that difference vectors capture meaningful transformations

Your task embedding (response − prompt) is analogous at the sentence level:

the transformation a model makes to solve the task

There isn’t a canonical paper that uses this exact operation for task routing, but the analogy provides motivation.

⸻

🧠 4) Response-Aware Representations

A few papers look at using generated responses (or model outputs) as signals:

🔹 “Latent Retrieval for Weak Supervision” (Lewis et al., 2022)
	•	Uses representations of model outputs (or intermediate features) as retrieval keys.
	•	Shows generated content often encodes task elements more robustly than input alone.

🔹 Chain-of-Thought / Self-Consistency Papers (Wang et al., 2022+)
	•	Response paths (and differences from prompt) reveal latent task difficulty.
	•	Not used for routing, but strongly supports the idea that response embeddings contain additional task signal.

⸻

🧠 5) Meta-Performance Prediction

A more direct line of work tries to predict model performance from some representation of input and/or output:
	•	Train a small model to predict whether a larger model will succeed.
	•	Use embeddings of (prompt + candidate output) as input to this predictor.

Examples:
	•	“Predicting model performance for task selection”
	•	“Learning to rank models based on task embeddings”

These aren’t mainstream yet, but they are exactly the meta learning version of what you’re exploring.