from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class CritiqueLevel(str, Enum):
    PERFECT = "perfect"
    MINOR_ISSUES = "minor_issues"
    MAJOR_ISSUES = "major_issues"
    FUNDAMENTAL_FLAWS = "fundamental_flaws"


class ImprovedCritique(BaseModel):
    """Enhanced critique with severity levels and suggestions"""
    level: CritiqueLevel
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    
    @property
    def no_critique_needed(self) -> bool:
        return self.level == CritiqueLevel.PERFECT


class ConsiliumConfig(BaseModel):
    """Enhanced configuration with more control"""
    models: List[str]
    initial_contexts: List[str]
    problem: str
    max_iterations: int = Field(ge=1)
    
    # New configuration options
    consensus_threshold: float = Field(default=0.8, ge=0, le=1)  # % of models that must approve
    min_improvement_threshold: float = Field(default=0.1)  # Minimum improvement to continue
    enable_critique_history: bool = Field(default=True)  # Show past critiques
    enable_solution_history: bool = Field(default=False)  # Show past solutions
    early_stop_on_plateau: bool = Field(default=True)  # Stop if no improvement
    weighted_voting: bool = Field(default=False)  # Weight votes by performance
    critique_severity_threshold: CritiqueLevel = Field(default=CritiqueLevel.MINOR_ISSUES)
    

class ModelPerformance(BaseModel):
    """Track model performance metrics"""
    model_index: int
    critiques_given: int = 0
    critiques_received: int = 0
    solutions_approved: int = 0
    average_confidence: float = 0
    improvement_rate: float = 0  # How much solutions improve between iterations


class EnhancedSolution(BaseModel):
    """Solution with metadata for better tracking"""
    model_index: int
    content: str
    iteration: int
    critique_summary: Optional[Dict[str, Any]] = None
    improvement_delta: Optional[float] = None  # How much it improved from last iteration
    
    
class ConsiliumResultV2(BaseModel):
    """Enhanced result with more insights"""
    final_solution: str
    iterations_used: int
    consensus_reached: bool
    consensus_level: float  # What % of models approved
    winning_model_index: int
    all_solutions: List[EnhancedSolution] = Field(default_factory=list)
    model_performance: List[ModelPerformance] = Field(default_factory=list)
    improvement_trajectory: List[float] = Field(default_factory=list)  # Quality over iterations
    termination_reason: str  # Why did it stop (consensus/max_iter/plateau)