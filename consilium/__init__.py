from .models import (
    ConsiliumRequest, ConsiliumResult, Critique, Solution,
    CritiqueLevel, ModelPerformance
)
from .core_v2 import run_consilium, EnhancedConsiliumOrchestrator
# Keep old imports available for backwards compatibility
from .core import ConsiliumOrchestrator

__all__ = [
    "ConsiliumRequest",
    "ConsiliumResult", 
    "Critique",
    "Solution",
    "CritiqueLevel",
    "ModelPerformance",
    "run_consilium",
    "EnhancedConsiliumOrchestrator",
    "ConsiliumOrchestrator"  # For backwards compatibility
]