# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import math
import re
from typing import Callable, Dict, Optional

from math_verify import verify, parse


def extract_answer_from_completion_tag(text):
    """
    Extract the text between <answer> and </answer> tags.

    Args:
        text (str): Input string containing <answer> tags

    Returns:
        str: The content between the tags, or None if tags aren't found
    """
    import re

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1)
    else:
        return ""

def extract_answer_from_completion(text):

    text = text.split('\n')

    return text[-1]


def accuracy_reward(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = sol

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = extract_answer_from_completion(content)
            print("Answer parsed: ", answer_parsed)
            print("Content: ", content)
            print("GOld Parsed: ", gold_parsed)
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(parse(gold_parsed), parse(answer_parsed)))
            except Exception as e:
                print(
                    f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}"
                )
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("\n") >= 1:
            count += 0.5
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]
    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(
    completions: list[Dict[str, str]], solution: list[str], **kwargs
) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = sol
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = extract_answer_from_completion(content)
        correctness.append(verify(parse(gold_parsed), parse(answer_parsed)))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = sol
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = extract_answer_from_completion(content)

            is_correct = verify(parse(gold_parsed), parse(answer_parsed))
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs


if __name__ == "__main__":
    text = "Ilé-iṣẹ́ ná $15000 lórí ìpolówó fún ọdún kan.\n Fún ọdún mìíràn, ó ná ìdá mẹ́ta iye yẹn, èyí tí ó jẹ́ $15000 * 3 = $45000.\n Àpapọ̀ iye tí ilé-iṣẹ́ ná lórí ìpolówó fún ọdún méjèèjì ni $15000 + $45000 = $60000.\nNítorí náà, àpapọ̀ iye tí ilé-iṣẹ́ ná lórí ìpolówó fún ọdún méjèèjì ni ó ná ìdá mẹ́ta iye yẹn The final answer is $20000"
    answer = "20000"
    print(accuracy_reward([[{"content": text}]], [answer]))
    print(tag_count_reward([[{"content": text}]]))
    print(reasoning_steps_reward([[{"content": text}]]))
    print(len_reward([[{"content": text}]], [answer]))
