import requests
import json
from typing import Dict, Any

M_TOKENS = 1e6


class Model:
    def __init__(self, name, input_cost, output_cost):
        self._name = name
        self._input_cost = input_cost
        self._output_cost = output_cost

    @property
    def name(self):
        return self._name

    @property
    def input_cost(self):
        return self._input_cost

    @property
    def output_cost(self):
        return self._output_cost


# From routers used in RouterBench paper:
# could not find "WizardLM/WizardLM-13B-V1.2",
# 404: "anthropic/claude-instant-1.0", # "claude-instant-v1",
# 404: "anthropic/claude-1", # "claude-v1",
# included: "anthropic/claude-2",  # "claude-v2",
# included: "openai/gpt-3.5-turbo-1106",  # "gpt-3.5-turbo-1106",
# included: "openai/gpt-4-1106-preview",  # "gpt-4-1106-preview",
# 404: "meta-llama/codellama-34b-instruct", # meta/code-llama-instruct-34b-chat
# 404: "meta-llama/llama-2-70b-chat", # meta/llama-2-70b-chat
# included: "mistralai/mistral-7b-instruct",  # "mistralai/mistral-7b-chat"
# included: "mistralai/mixtral-8x7b-instruct",  # "mistralai/mixtral-8x7b-chat"
# 404: "01-ai/yi-34b-chat" # "zero-one-ai/Yi-34B-Chat"

MODELS = [
    Model(
        name="anthropic/claude-2",
        input_cost=8.0 / M_TOKENS,
        output_cost=24.0 / M_TOKENS,
    ),
    Model(
        name="openai/gpt-3.5-turbo-1106",
        input_cost=1.0 / M_TOKENS,
        output_cost=2.0 / M_TOKENS,
    ),
    Model(
        name="openai/gpt-4-1106-preview",
        input_cost=10.0 / M_TOKENS,
        output_cost=30.0 / M_TOKENS,
    ),
    Model(
        name="mistralai/mistral-7b-instruct",
        input_cost=0.055 / M_TOKENS,
        output_cost=0.055 / M_TOKENS,
    ),
    Model(
        name="mistralai/mixtral-8x7b-instruct",
        input_cost=0.24 / M_TOKENS,
        output_cost=0.24 / M_TOKENS,
    ),
]


def get_model_response(model: Model, prompt: str) -> requests.Response:
    """
    Sends a prompt to the OpenRouter API and retrieves the response from the specified model.

    Returns:
        The HTTP response object returned by the OpenRouter API. Check result of this function
            with .json() or .status_code
    """

    return requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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"
        },
        data=json.dumps(
            {
                "model": model.name,
                "messages": [{"role": "user", "content": prompt}],
            }
        ),
    )


def get_output_from_response(response: requests.Response):
    res_json = response.json()
    return res_json["choices"][0]["message"]["content"]


# Reference model should be competitive with candidate, shouldn't always win/lose
REFERENCE_MODEL = Model(
    name="openai/gpt-3.5-turbo",
    input_cost=0.5 / M_TOKENS,
    output_cost=1.5 / M_TOKENS,
)

# Same judge model is used as in LLM-as-judge paper, should be very reliable and accurate
JUDGE_MODEL = Model(
    name="openai/gpt-4",
    input_cost=30 / M_TOKENS,
    output_cost=60 / M_TOKENS,
)


_JUDGE_MODEL_COMPARISON_PROMPT = """
Please compare the two following responses to the same prompt. In your comparison, evaluate the quality,
relevance, clarity, and completeness of the responses. Choose the response that you find superior based
on these criteria, and explain why it is better in terms of content, structure, and overall usefulness."

Model 1 Output:
{model_1_output}

Model 2 Output:
{model_2_output}
"""

_JUDGE_MODEL_OUTPUT_PROMPT = """
Given the analysis below, decide which model performed better at addressing the prompt. Your answer should
only be "Model 1" if the analysis indicates that Model 1 did a better job, "Model 2" if Model 2 performed
better, or "Tie" if you cannot decide on the winner. Please output only the name of the model with no additional text.

Analysis:
{analysis}
"""

_HUMAN_COMPARISON_PROMPT = """
======
Given the prompt below and the output of Model 1 and Model 2, type
1 and hit enter if you think Model 1 did a better job, type 2 and
hit enter if you think Model 2 did a better job, or type 3 and hit enter if you think that it is a tie. 

Prompt:
{prompt}

Model 1 Output:
{model_1_output}

Model 2 Output:
{model_2_output}
======
"""

# A hyperparameter, how many times candidate model output should be compared to the
# reference model output for each prompt. Higher number increases quality of comparison
# but increases costs
_NUM_COMPARISONS = 6


def compare_model_to_reference(
    candidate_model: Model, candidate_model_output: str, prompt: str
) -> float:
    """
    Use the judge model to compare the candidate model output to the reference model output

    It would be interesting to see the cost breakdown on how much this process costs
    and how much of a benefit we see in the quality of the performance.

    Returns:
        win_rate: the % of time the judge model prefers the candidate model's output to the reference
            model's output for the given prompt
    """

    # Get the reference model response
    reference_response = get_model_response(REFERENCE_MODEL, prompt)
    reference_output = get_output_from_response(reference_response)

    # Record the results of the comparison according to the judge model/human decision, stores
    # whether candidate model beat reference at that iteration
    judge_results: list[bool] = []
    human_results: list[bool] = []

    while len(judge_results) < _NUM_COMPARISONS:

        # Switch the order of candidate/reference model at each iteration to combat
        # order bias
        is_candidate_model_1 = len(judge_results) % 2 == 0
        model_1_output, model_2_output = (
            (candidate_model_output, reference_output)
            if is_candidate_model_1
            else (reference_output, candidate_model_output)
        )

        # Use the judge model to compare outputs and determine which is preferred
        judge_response = get_model_response(
            JUDGE_MODEL,
            _JUDGE_MODEL_COMPARISON_PROMPT.format(
                model_1_output=model_1_output, model_2_output=model_2_output
            ),
        )
        judge_response_output = get_output_from_response(judge_response)

        # Ask the judge model to simplify its comparison to just "Model 1" or "Model 2"
        simplified_response = get_model_response(
            JUDGE_MODEL,  # using a simpler model here would help save money
            _JUDGE_MODEL_OUTPUT_PROMPT.format(analysis=judge_response_output),
        )
        simplified_output = get_output_from_response(simplified_response)

        # Determine the winner from the judge model's response
        if "1" in simplified_output and "2" not in simplified_output:
            judge_winner = "1"
        elif "2" in simplified_output and "1" not in simplified_output:
            judge_winner = "2"
        else:
            judge_winner = None

        # If the judge model has decided, append the result
        if judge_winner:

            judge_results.append(
                is_candidate_model_1
                if judge_winner == "1"
                else not is_candidate_model_1
            )

    # Prompt the human for their decision
    print(_HUMAN_COMPARISON_PROMPT.format(prompt=prompt, model_1_output=model_1_output, model_2_output=model_2_output))
    human_decision = None
    while human_decision not in ["1", "2", "3"]:
        human_decision = input()
        if human_decision not in ["1", "2"]:
            print("Invalid input. Please enter '1', '2', or '3'")

    if human_decision == "1":
        human_results.append(is_candidate_model_1)
    elif human_decision == "2":
        human_results.append(not is_candidate_model_1) 

    # results store True values when candidate model won, so count True's to get win_rate
    judge_win_rate = sum(judge_results) / len(judge_results)
    human_win_rate = sum(human_results) / len(human_results)

    return judge_win_rate, human_win_rate
