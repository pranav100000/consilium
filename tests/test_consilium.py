import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai import Agent
from unittest.mock import AsyncMock, patch

from consilium import ConsiliumRequest, ConsiliumResult, run_consilium
from consilium.core import ConsiliumOrchestrator


@pytest.fixture
def simple_request():
    return ConsiliumRequest(
        models=["test", "test"],
        initial_contexts=[
            "You are an expert in mathematics",
            "You are an expert in physics"
        ],
        problem="What is 2+2?",
        max_iterations=2
    )


@pytest.fixture
def single_model_request():
    return ConsiliumRequest(
        models=["test"],
        initial_contexts=["You are a helpful assistant"],
        problem="What is the capital of France?",
        max_iterations=1
    )


@pytest.mark.asyncio
async def test_consilium_single_model_consensus(single_model_request):
    with patch('consilium.core.Agent') as MockAgent:
        mock_agent = AsyncMock()
        mock_result = AsyncMock()
        mock_result.output = "Paris"
        mock_agent.run.return_value = mock_result
        
        mock_critique_result = AsyncMock()
        mock_critique_result.output.no_critique_needed = True
        mock_critique_result.output.critiques = []
        
        MockAgent.return_value = mock_agent
        MockAgent.side_effect = [mock_agent] * 10
        
        result = await run_consilium(single_model_request)
        
        assert isinstance(result, ConsiliumResult)
        assert result.iterations_used == 1
        assert result.consensus_reached or not result.consensus_reached


@pytest.mark.asyncio
async def test_consilium_multiple_models(simple_request):
    with patch('consilium.core.Agent') as MockAgent:
        mock_agents = []
        for i in range(2):
            mock_agent = AsyncMock()
            mock_result = AsyncMock()
            mock_result.output = "4"
            mock_agent.run.return_value = mock_result
            mock_agents.append(mock_agent)
        
        MockAgent.side_effect = mock_agents * 10
        
        result = await run_consilium(simple_request)
        
        assert isinstance(result, ConsiliumResult)
        assert result.iterations_used <= simple_request.max_iterations
        assert isinstance(result.final_solution, str)
        assert isinstance(result.winning_model_index, int)


@pytest.mark.asyncio
async def test_consilium_max_iterations():
    request = ConsiliumRequest(
        models=["test", "test", "test"],
        initial_contexts=["Context 1", "Context 2", "Context 3"],
        problem="Complex problem",
        max_iterations=3
    )
    
    with patch('consilium.core.Agent') as MockAgent:
        mock_agents = []
        for i in range(3):
            mock_agent = AsyncMock()
            mock_result = AsyncMock()
            mock_result.output = f"Solution {i}"
            mock_agent.run.return_value = mock_result
            mock_agents.append(mock_agent)
        
        MockAgent.side_effect = mock_agents * 20
        
        result = await run_consilium(request)
        
        assert isinstance(result, ConsiliumResult)
        assert result.iterations_used <= 3
        assert len(result.all_solutions) > 0


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    request = ConsiliumRequest(
        models=["test", "test"],
        initial_contexts=["Context 1", "Context 2"],
        problem="Test problem",
        max_iterations=1
    )
    
    with patch('consilium.core.Agent'):
        orchestrator = ConsiliumOrchestrator(request)
        assert len(orchestrator.agents) == 2
        assert orchestrator.iteration == 0
        assert len(orchestrator.solutions) == 0


def test_consilium_request_validation():
    with pytest.raises(ValueError):
        ConsiliumRequest(
            models=["test"],
            initial_contexts=["Context"],
            problem="Problem",
            max_iterations=0
        )
    
    request = ConsiliumRequest(
        models=["test"],
        initial_contexts=["Context"],
        problem="Problem",
        max_iterations=1
    )
    assert request.max_iterations == 1