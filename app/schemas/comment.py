from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional, List

class CommentCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The comment text to classify", json_schema_extra={"example": "phim dở vcl"})
    task: Literal["toxic-speech-detection", "hate-speech-detection", "hate-spans-detection", "all"] = Field(
        default="all",
        description="The detection task to execute, or 'all' to run all tasks"
    )

class CommentCheckResponse(BaseModel):
    text: str
    task: str
    result: Optional[str] = None
    results: Optional[Dict[str, str]] = None
    toxicSpans: Optional[List[Dict[str, int]]] = None
    time_taken_seconds: float
