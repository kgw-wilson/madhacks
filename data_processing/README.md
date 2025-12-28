# Data Processing

This directory contains scripts that download and process data.

To start, run

```shell
huggingface-cli login
```

and provide HuggingFace with your access token. Without this, you won't be able to download the RouterBench data. 

Then, you can run the files in this order:

1. download_filter_rb_data.py > Downloads RouterBench 0-shot data from HuggingFace into the project and filters out columns which aren't usable.
2. add_pr_embeddings.py > Uses the saved data from step 1 to generate embeddings (512 x 512 arrays describing the content of each paragraph) for both prompts and responses.

Can pause here to run experiment 1 in the `exp1` directory.

3. add_model_response_embeddings.py > Adds response embeddings for a small, medium, and large model and saves that into a new dataframe.

## Downloading Data Rationale

We chose to consider the 0-shot data only rather than any of the other data because it meant less data (prompts no longer contain any examples before the actual task), and because the embedding generated from a prompt would be less noisy (no extra information other than the actual task).

Warning: requires about a GB of space

Might be best to exclude chinese data because it's not great with the given encoder and only 2% of the data, but because of comparability to original paper, it is kept.


## FROM CHATGPT


These preserve interpretability and are easy to compare.

1.1 Difference (you already have)

task_vec = response_emb - prompt_emb

Interpretation:
“What information was added by the model?”

Often surprisingly strong.

1.2 Weighted difference

task_vec = response_emb - α * prompt_emb

Where α ∈ [0.3, 1.0].

This can help if prompts dominate magnitude or semantic direction.

1.3 Concatenation (baseline)

task_vec = np.concatenate([prompt_emb, response_emb])

Pros:
	•	Maximum information retention
Cons:
	•	Doubles dimensionality
	•	Distance metrics become less meaningful

1.4 Difference + response

task_vec = np.concatenate([response_emb, response_emb - prompt_emb])

This preserves:
	•	What the model said
	•	How it deviated from the prompt

This often performs better than raw concatenation.


2. Similarity-based features (very underrated)

Instead of embeddings, use relationships.

2.1 Cosine similarity as a scalar feature

cos_sim = cosine(prompt_emb, response_emb)

This measures:

“How much new information did the model add?”

Low similarity often means reasoning or transformation tasks.

You can:
	•	Use it alone
	•	Append it to embeddings
	•	Bucket it into bins

2.2 Prompt–response angle space

You can construct a small feature vector:

features = [
    cosine(prompt, response),
    ||response||,
    ||response - prompt||
]

This works shockingly well for routing.


3. Neighborhood-aware representations (very strong)

This aligns closely with what you’re already doing.

3.1 Response-relative neighborhood embedding

Instead of embedding the response alone:

“How does this response relate to known responses?”

For a response r:

task_vec = mean(top_k_similar_responses(r))


You’re effectively embedding task archetypes, not text.

⸻

3.2 Cross-model response agreement

For a given prompt:
	•	Collect responses from multiple models
	•	Embed each
	•	Measure:
	•	variance
	•	mean cosine distance
	•	clustering tightness

This captures task difficulty and ambiguity.


4. Model-conditional embeddings (very relevant to routing)

Instead of one embedding per sample:

(task, model) → embedding

Examples:
	•	Response embedding per model
	•	Delta vs. reference model (e.g., GPT-4)

Then routing becomes:

“Which model historically performs best for tasks with this embedding geometry?”

This matches your current experiments very well.


5. Contrastive task fingerprints (powerful)

Construct embeddings that explicitly encode which model is better.

Example:

task_vec = embedding(model_A_response) - embedding(model_B_response)

Over many samples, this creates a model preference space.

This is extremely powerful for routing.

6. Cluster-aware representations (bridging to your future work)

Instead of raw kNN:
	1.	Cluster task embeddings (k-means, HDBSCAN)
	2.	Compute:
	•	Per-cluster model performance
	•	Confidence / entropy
	3.	Route based on cluster membership

This gives:
	•	Interpretability
	•	Stability
	•	Better generalization

⸻

7. A clean experimental progression (recommended)

If you want a principled roadmap:
	1.	Prompt embedding only (baseline)
	2.	Response embedding only
	3.	Prompt + response concat
	4.	Response − prompt
	5.	Response neighborhood mean
	6.	Clustered task embeddings

Each step adds semantic structure without massive complexity.

⸻

8. Key insight (important)

Routing doesn’t need “understanding” — it needs consistent separability.

Even noisy embeddings work if:
	•	Similar tasks cluster
	•	Different model strengths separate