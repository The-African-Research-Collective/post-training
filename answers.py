from pydantic import BaseModel


class Answer(BaseModel):
    is_valid_question: bool
    reasoning: str
    final_answer: str | None
