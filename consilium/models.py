from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class CritiqueLevel(str, Enum):
    PERFECT = "perfect"
    MINOR_ISSUES = "minor_issues"
    MAJOR_ISSUES = "major_issues"
    FUNDAMENTAL_FLAWS = "fundamental_flaws"


class ConsiliumRequest(BaseModel):
    models: List[str]
    initial_contexts: List[str]
    problem: str
    max_iterations: int = Field(ge=1)
    # New optional configuration
    consensus_threshold: float = Field(default=0.8, ge=0, le=1)
    min_improvement_threshold: float = Field(default=0.1)
    enable_critique_history: bool = Field(default=True)
    enable_solution_history: bool = Field(default=False)
    early_stop_on_plateau: bool = Field(default=True)
    weighted_voting: bool = Field(default=False)


class Critique(BaseModel):
    no_critique_needed: bool  # Kept for backwards compatibility
    critiques: List[str] = Field(default_factory=list)
    # New fields
    level: Optional[CritiqueLevel] = None
    suggestions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    
    @property
    def is_acceptable(self) -> bool:
        """Check if critique level is acceptable (perfect or minor issues)"""
        if self.level:
            return self.level in [CritiqueLevel.PERFECT, CritiqueLevel.MINOR_ISSUES]
        return self.no_critique_needed


class Solution(BaseModel):
    model_index: int
    content: str
    iteration: int
    # New fields
    critique_summary: Optional[Dict[str, Any]] = None
    improvement_delta: Optional[float] = None
    previous_content: Optional[str] = None


class ModelPerformance(BaseModel):
    model_index: int
    critiques_given: int = 0
    solutions_approved: int = 0
    average_confidence: float = 0
    improvement_rates: List[float] = Field(default_factory=list)
    vote_weight: float = 1.0


class ConsiliumResult(BaseModel):
    final_solution: str
    iterations_used: int
    consensus_reached: bool
    winning_model_index: int
    all_solutions: List[Solution] = Field(default_factory=list)
    # New fields
    consensus_level: float = Field(default=0.0)
    model_performance: List[ModelPerformance] = Field(default_factory=list)
    improvement_trajectory: List[float] = Field(default_factory=list)
    termination_reason: str = Field(default="max_iterations")