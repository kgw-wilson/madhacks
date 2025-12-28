# TopK Similarity in Response Embedding Space

In Experiment 1, we showed that kNN over prompt embeddings provides a strong routing baseline.
Here, we test whether using response embeddings can further improve performance.

The intuition is that task-relevant information may be more clearly expressed in a model’s response than in the prompt itself. For example, many prompts vary widely in content, but their responses may reveal the true task structure (e.g., selecting a multiple-choice option vs. generating free-form text).

This suggests a two-stage routing setup:
	1.	A lightweight model generates an initial response.
	2.	The response embedding is used to retrieve similar examples and select the best downstream model.

If effective, this approach could enable more task-aware routing with minimal additional cost.

## Results

Overall, the experiment was successful, as we got accuracy measures that were very similar to the RouterBench paper, varying between 77% and 84% for different values of k. For more details, please see `results.csv`.

At k = 40, several response-embedding models consistently outperform the prompt-embedding baseline (0.8477 at k=40):

mistral-7b-chat: 0.8523
mixtral-8x7b-chat: 0.8523
claude-instant-v1: 0.8525
meta/llama-2-70b-chat: 0.8502
code-llama-34b: 0.8518
claude-v1: 0.8539 (best overall)
claude-v2: 0.8532

So, response embeddings can outperform prompt embeddings when k is sufficiently large (≈40) and the response comes from a reasonably strong model.

However, at k = 5 or 10, response embeddings are generally worse than prompt embeddings, with a few exceptions that are about equally good. This suggests that maybe response embeddings are noisier locally and that prompt embeddings capture coarse task similarity better at small k.

Model quality seems to matter, but it's clear increasing parameters doesn't strictly result in an increase in performance.

 