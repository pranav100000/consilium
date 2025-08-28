from typing import List, Optional, Union
from pydantic import BaseModel, Field


class ConsiliumRequest(BaseModel):
    models: List[str]
    initial_contexts: List[str]
    problem: str
    max_iterations: int = Field(ge=1)


class Critique(BaseModel):
    no_critique_needed: bool
    critiques: List[str] = Field(default_factory=list)


class Solution(BaseModel):
    model_index: int
    content: str
    iteration: int


class ConsiliumResult(BaseModel):
    final_solution: str
    iterations_used: int
    consensus_reached: bool
    winning_model_index: int
    all_solutions: List[Solution] = Field(default_factory=list)