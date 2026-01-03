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