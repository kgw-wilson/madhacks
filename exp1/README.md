# Experiment 1: Top-K Similarity

## Running the Experiment

After completing all relevant data processing tasks (see the data_processing folder), run the experiment from the project root using:

```shell
python -m exp1.topk_similarity
```

## Rationale

This experiment implements the simplest baseline router described in the RouterBench paper: a k-nearest neighbors (kNN) routing approach.

This experiment's main purpose was to verify that our data processing and evaluation pipeline worked correctly.

## Results

Overall, the experiment was successful, as we got accuracy measures that were very similar to the RouterBench paper, varying between 77% and 84% for different values of k. For more details, please see `results.csv`.

## Why kNN Instead of MLP?

The main reason is that kNN supports zero-shot model insertion. MLPs require retraining when new LLMs, fine-tuned variants, or newer model generations are added. kNN can incorporate new models without relabeling or retraining, making kNN is easier to experiment with. The secondary reason is that performance between the MLP-based router and the kNN-based router is largely comparable in the paper.

## Implementation Notes

We try to reproduce RouterBench methodology where applicable (train/test split, k values). We completely ignore cost and willingness to pay for simplicity, which provides a best-case scenario for model selection and simplifies the code.