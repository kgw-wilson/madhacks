# Experiment 3: Cluster-Based Similarity

Our next idea was that given that kNN proved to be effective in the prompt embedding space and in the response embedding space, we hypothesized that we could get some cost savings over kNN by pre-computing cluster centers and at inference time using the best model for the prompt closest to the center of a cluster in the training space.

Cost saving over kNN: 
kNN dataset len D, n candidate models cost is D * n * avg_model_cost
Clustering: k clusters, n candidate models, cost is k * n * avg_model_cost
Savings are D/k