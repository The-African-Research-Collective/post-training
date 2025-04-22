from pydantic import BaseModel


class Answer(BaseModel):
    is_valid_question: bool
    reasoning: str
    final_answer: str | None


class TranslatedProblem(BaseModel):
    problem_translation: str
    step_by_step_response: str
