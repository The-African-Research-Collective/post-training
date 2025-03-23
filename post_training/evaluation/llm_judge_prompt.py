import re
import logging

from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics_sample import JudgeLLM

logger = logging.getLogger(__name__)


def gpt_judge_for_closeended_freeform(
    question: str, options: str, answer: str, gold: str
):
    """
    Returns a prompt for the GPT model to judge if the model's answer is equal to the golden answer(s) for a close-ended freeform question.
    """
    return [
        {"role": "system", "content": "In this task, I want you to act as a judge."},
        {
            "role": "user",
            "content": f"""You will be provided with a mathematics question, its golden answer, and the model's answer. The question could be in any language, and the model's answer could be in any language. Your task is to judge if the model's answer is equal to the golden answer,
Your task is to judge if the model's answer is correct or not based on the golden answer.
Your should first briefly give your reasoning process regarding how the model's answer is equal the golden answer(s), and then give the boolean score of 0 or 1 if they are equal or not.

The score must strictly follow this format: \"[[score]]\", e.g., \": [[1]]\". Below are some examples.

Example 1:
Question: Roger ní bọ́ọ́lù aláfajọ̀ 5. Ó ra agolo bọ́ọ́lù aláfajọ̀gbá 2 kún-un. Agolo kọ̀ọ̀kan ní bọ́ọ́lù aláfajọ̀gbá 3. Bọ́ọ́lù aláfajọ̀gbá mélòó ni ó ní báyìí?
Golden Answer: 11
Model's Answer: Ìdáhùn lẹ́sẹsẹ: Roger bẹ̀rẹ̀ pẹ̀lú bọ́ọ́lù 5. Agolo 2 bọ́ọ́lù aláfajọ̀gbá kọ̀ọ̀kan 3 jẹ́ bọ́ọ́lù aláfajọ̀gbá 6. 5 + 6 = 11. Ìdáhùn náà ni 11.
Your Judgment: The model and the groundtruth answer are the same. The score: [[1]]

Example 2:
Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Golden Answer: 8
Model's Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8.
Your Judgment: The model and the groundtruth answer are the same. The score: [[1]]

Example 3:
Question: URoger unamabhola ethenisi ama-5. Uthenga amanye amathini ama-2 amabhola ethenisi. Ikani ngalinye linamabhola ethenisi ama-3. Mangaki amabhola ethenisi anawo manje?
Golden Answer: 11
Model's Answer: Mangaki alikuwa na 5 x 2 = <<5*2=10>>amabhola 10 ethenisi.\nBaada ya kununua zingine, sasa ana 10 + 3 = <<10+3=13>>amabhola 13 ethenisi.\nJibu ni 13
Your Judgment: The model answer and the golden answer do not align. The correctness score: [[0]].

Note that there is only one gold answer that is correct. Thus if the model's answer matches the gold answer, it should be considered correct. 
Judge the below case, give the brief reasoning process and the correctness score.

Question: {question}
Golden Answer: {gold}
Model's Answer: {answer}
Your Judgment:
""",
        },
    ]


class JudgeLLMMathEval(JudgeLLM):
    def compute(
        self,
        sample_ids: list[str],
        responses: list,
        formatted_docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        """
        Compute the score of a generative task using a llm as a judge.
        The generative task can be multiturn with 2 turns max, in that case, we
        return scores for turn 1 and 2. Also returns user_prompt and judgement
        which are ignored later by the aggregator.
        """
        questions = [
            formatted_doc.specific["question"] for formatted_doc in formatted_docs
        ]
        options = [formatted_doc.choices for formatted_doc in formatted_docs]
        golds = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]
        predictions = [response[0].result[0] for response in responses]

        metrics = []
        for q, o, g, p in zip(questions, options, golds, predictions):
            score, message, judge = self.judge.evaluate_answer(q, p, o, g)

            metrics.append(
                {
                    f"judge_score_{self.short_judge_name}": score,
                    f"user_prompt_{self.short_judge_name}": message,
                    f"judgement_{self.short_judge_name}": judge,
                }
            )

        return metrics


def process_judge_response_gpt(x: str) -> int:
    """
    Process the judge response and extract the score for the GPT model.

    Args:
        x (str): The judge response.

    Returns:
        int: The score extracted from the judge response.
    """
    try:
        search = re.search(r"\[\[([01])\]\]", x)
        return int(search.group(1)) if search else 0
    except Exception as e:
        logger.warning(f"Error processing judge response for multichoice GPT: {e}")
        return 0
