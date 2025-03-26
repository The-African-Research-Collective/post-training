"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

import logging
import numpy as np

from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from .llm_judge_prompt import (
    gpt_judge_for_closeended_freeform,
    process_judge_response_gpt,
    JudgeLLMMathEval,
)

logger = logging.getLogger(__name__)


def prompt_fn_openaimmlu_math(line, task_name: str = None):
    query_template = """Given the following question, choose the correct answer from the choices provided.
    Question: {Question}

    Given
    Choices:
    A. {choice_0}
    B. {choice_1}
    C. {choice_2}
    D. {choice_3}

    Answer:
    """
    query = query_template.format(
        Question=line["Question"],
        choice_0=line["A"],
        choice_1=line["B"],
        choice_2=line["C"],
        choice_3=line["D"],
    )
    choices = [line["A"], line["B"], line["C"], line["D"]]

    answer_index = ["A", "B", "C", "D"].index(line["Answer"])
    return Doc(
        task_name=task_name,
        query=query,
        original_query=line["Question"],
        choices=choices,
        gold_index=answer_index,
        instruction="",
        specific={"question": line["Question"], "choices": choices},
    )


def generate_per_language_task_config(
    language: str, data_subset: str, generation_size: int = 512
):
    """
    Generate the task config for the openaimmlu_math task for a specific language.
    """
    return LightevalTaskConfig(
        name=f"afrimathevals:openaimmlu_math_{language}",
        prompt_function=prompt_fn_openaimmlu_math,
        suite=["community"],
        hf_repo="taresco/OPENAI-MMLU-FILTERED-MATH",
        hf_subset=data_subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        metric=[llm_judge_math_gpt_judge],
        generation_size=generation_size,
    )


llm_judge_math_gpt_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_math"],
    higher_is_better={"judge_score_gpt-4o": True},
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=JudgeLLMMathEval(
        judge_model_name="gpt-4o",
        template=gpt_judge_for_closeended_freeform,
        process_judge_response=process_judge_response_gpt,
        judge_backend="openai",
        short_judge_name="gpt-4o",
    ).compute,
    corpus_level_fn={
        "judge_score_gpt-4o": np.mean,
    },
)

TASKS_TABLE = [
    generate_per_language_task_config("ara", "AR_XY"),
    generate_per_language_task_config("swa", "SW_KE"),
    generate_per_language_task_config("yor", "YO_NG"),
]
