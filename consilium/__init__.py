from .models import (
    ConsiliumRequest, ConsiliumResult, Critique, Solution,
    CritiqueLevel, ModelPerformance
)
from .core_v2 import run_consilium as run_consilium_enhanced, EnhancedConsiliumOrchestrator
from .core_v3 import run_consilium_strict, StrictConsiliumOrchestrator
# Keep old imports available for backwards compatibility
from .core import ConsiliumOrchestrator

async def run_consilium(request: ConsiliumRequest) -> ConsiliumResult:
    """Main entry point - uses appropriate orchestrator based on settings"""
    if request.strict_mode:
        return await run_consilium_strict(request)
    else:
        return await run_consilium_enhanced(request)

__all__ = [
    "ConsiliumRequest",
    "ConsiliumResult", 
    "Critique",
    "Solution",
    "CritiqueLevel",
    "ModelPerformance",
    "run_consilium",
    "run_consilium_v2",
    "run_consilium_strict",
    "EnhancedConsiliumOrchestrator",
    "StrictConsiliumOrchestrator",
    "ConsiliumOrchestrator"  # For backwards compatibility
]