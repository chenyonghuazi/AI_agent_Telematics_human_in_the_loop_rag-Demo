from pydantic import BaseModel
from typing import Literal,List

class event_detail(BaseModel):
    
    event_type: str | None
    event_value: int | float | str | None

class Query(BaseModel):
    evaluation: Literal["Good", "moderate", "bad"]  
    event: List[event_detail] | None


