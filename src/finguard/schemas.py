"""HTTP request bodies. Kept tiny so route modules stay readable."""
from pydantic import BaseModel


class Features(BaseModel):
    features: dict[str, float | None]


class Question(BaseModel):
    question: str


class FeedbackBody(BaseModel):
    txn_id: int
    is_fraud: bool
