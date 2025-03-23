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


def prompt_fn_afrimgsm(line, task_name: str = None):
    query_template = """Question: {question}

    Answer:
    """
    query = query_template.format(
        question=line["question"],
    )
    return Doc(
        task_name=task_name,
        query=query,
        original_query=line["question"],
        choices=[str(line["answer_number"])],
        gold_index=0,
        instruction="",
        specific={"question": line["question"]},
    )


def generate_per_language_task_config(language: str, generation_size: int = 512):
    """
    Generate the task config for the afrimgsm task for a specific language.
    """
    return LightevalTaskConfig(
        name=f"afrimathevals:afrimgsm_{language}",
        prompt_function=prompt_fn_afrimgsm,
        suite=["community"],
        hf_repo="masakhane/afrimgsm",
        hf_subset=language,
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
    generate_per_language_task_config("amh"),
    generate_per_language_task_config("eng"),
    generate_per_language_task_config("ewe"),
    generate_per_language_task_config("fra"),
    generate_per_language_task_config("hau"),
    generate_per_language_task_config("ibo"),
    generate_per_language_task_config("kin"),
    generate_per_language_task_config("lin"),
    generate_per_language_task_config("lug"),
    generate_per_language_task_config("orm"),
    generate_per_language_task_config("sna"),
    generate_per_language_task_config("sot"),
    generate_per_language_task_config("swa"),
    generate_per_language_task_config("twi"),
    generate_per_language_task_config("vai"),
    generate_per_language_task_config("wol"),
    generate_per_language_task_config("xho"),
    generate_per_language_task_config("yor"),
    generate_per_language_task_config("zul"),
]
