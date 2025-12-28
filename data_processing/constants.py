from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer


# Column names of interest that are native to RouterBench data
SAMPLE_ID_COL = "sample_id"
DATASET_NAME_COL = "eval_name"
PROMPT_COL = "prompt"
MODEL_PERFORMANCE_COLS = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "claude-instant-v1",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/llama-2-70b-chat",
    "meta/code-llama-instruct-34b-chat",
    "claude-v1",
    "claude-v2",
]
RESPONSE_SUFFIX = "|model_response"

# These column names are added during data processing
PROMPT_EMBEDDING_COL = "prompt_embedding"
RESPONSE_EMBEDDING_SUFFIX = "|embedding"

# Save data relative to the project root to ensure consistent paths
RB_DATA_PATH = Path(__file__).parent.parent / "routerbench_data"
FILTERED_CSV_PATH = RB_DATA_PATH / "filtered.csv"
FILTERED_WITH_EMBEDDINGS_PATH = RB_DATA_PATH / "filtered_with_embeddings.pkl"

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Load SBERT model
# SBERT is used to generate sentence embeddings because it produces
# semantically meaningful vector representations for entire sentences
# efficiently, allowing us to compare prompts and responses quickly
# without running pairwise BERT computations. This particular version is
# chosen becuase it's the one used in the RouterBench code:
# https://github.com/withmartian/routerbench/blob/cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c/evaluate_routers.py#L339C63-L339C79
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_get_device())

# The name of the .CSV file to save for results of each experiment
RESULTS_CSV_FILENAME = "results.csv"