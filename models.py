# models.py
from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    """Defines the structure for the incoming API request."""
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    """Defines the structure for the API response."""
    answers: List[str]