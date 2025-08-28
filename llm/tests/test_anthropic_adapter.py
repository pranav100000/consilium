import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Sequence

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse as PydAI_ModelResponse, # <--- ADD/ENSURE THIS
    UserPromptPart,
    PartDeltaEvent as PydAI_PartDeltaEvent,
    PartStartEvent as PydAI_PartStartEvent,
    FinalResultEvent as PydAI_FinalResultEvent,
    TextPart as PydAI_TextPart,
    TextPartDelta as PydAI_TextPartDelta,
)
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.usage import Usage

from llm.adapters.anthropic_adapter import AnthropicAdapter
from llm.streaming import (
    StreamStartEvent,
    TextDeltaEvent,
    FinalOutputEvent,
    StreamEndEvent,
    AnyStreamEvent
)

# Mock fixtures for Anthropic components
@pytest.fixture
def mock_anthropic_provider(mocker):
    return mocker.patch('llm.adapters.anthropic_adapter.AnthropicProvider', autospec=True)

@pytest.fixture
def mock_anthropic_model(mocker):
    return mocker.patch('llm.adapters.anthropic_adapter.AnthropicModel', autospec=True)

@pytest.fixture
def mock_pydantic_ai_agent(mocker):
    mock_agent_instance = mocker.AsyncMock(spec=Agent) # The agent instance is an AsyncMock

    # Make run_stream a MagicMock attribute on the AsyncMock agent instance
    # This MagicMock, when called, will return an AsyncMock (the async context manager)
    async_context_manager = mocker.AsyncMock(name="run_stream_async_context_manager")
    mock_agent_instance.run_stream = MagicMock(
        name="run_stream_method_mock", 
        return_value=async_context_manager
    )
    
    # __aenter__ of async_context_manager will be configured in the test
    # to return mock_streamed_run_result by the test itself:
    # mock_agent_instance.run_stream.return_value.__aenter__.return_value = mock_streamed_run_result

    mock_agent_instance.run = AsyncMock() # For non-streaming calls
    mock_agent_class = mocker.patch('llm.adapters.anthropic_adapter.Agent', 
                                  return_value=mock_agent_instance, 
                                  autospec=True)
    return mock_agent_class, mock_agent_instance

@pytest.fixture
def mock_get_provider_config(mocker):
    return mocker.patch('llm.adapters.anthropic_adapter.get_provider_config')


@pytest.mark.asyncio
async def test_anthropic_adapter_initialize_agent_with_api_key(
    mock_anthropic_provider,
    mock_anthropic_model,
    mock_pydantic_ai_agent,
    mock_get_provider_config
):
    mock_get_provider_config.return_value = {} 
    mock_agent_class, _ = mock_pydantic_ai_agent

    adapter = AnthropicAdapter(
        model_name="claude-3-opus-20240229",
        api_key="anthropic_test_key",
        system_prompt="You are a helpful Anthropic assistant."
    )

    mock_anthropic_provider.assert_called_once_with(api_key="anthropic_test_key")
    mock_anthropic_model.assert_called_once_with(model_name="claude-3-opus-20240229", provider=mock_anthropic_provider.return_value)
    mock_agent_class.assert_called_once_with(model=mock_anthropic_model.return_value, instructions="You are a helpful Anthropic assistant.")
    assert adapter.agent is not None

@pytest.mark.asyncio
async def test_anthropic_adapter_initialize_agent_from_config(
    mock_anthropic_provider,
    mock_anthropic_model,
    mock_pydantic_ai_agent,
    mock_get_provider_config
):
    mock_get_provider_config.return_value = {"api_key": "anthropic_config_key"}
    mock_agent_class, _ = mock_pydantic_ai_agent

    adapter = AnthropicAdapter(
        model_name="claude-3-sonnet-20240229",
        original_provider_name="anthropic"
    )
    
    mock_get_provider_config.assert_called_once_with("anthropic")
    mock_anthropic_provider.assert_called_once_with(api_key="anthropic_config_key")
    mock_anthropic_model.assert_called_once_with(model_name="claude-3-sonnet-20240229", provider=mock_anthropic_provider.return_value)
    mock_agent_class.assert_called_once_with(model=mock_anthropic_model.return_value)
    assert adapter.agent is not None

@pytest.mark.asyncio
async def test_anthropic_adapter_send_message_async_non_streaming(
    mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_get_provider_config.return_value = {"api_key": "fake_anthropic_key"}
    _, mock_agent_instance = mock_pydantic_ai_agent
    
    mock_run_result = MagicMock()
    mock_run_result.output = "Anthropic assistant response"
    mock_agent_instance.run.return_value = mock_run_result

    adapter = AnthropicAdapter(model_name="claude-3-haiku-20240307", api_key="fake_anthropic_key")

    history: Sequence[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Previous Anthropic message")])]
    response = await adapter.send_message_async(
        user_prompt="Hello Anthropic assistant",
        history=history,
        stream=False,
        model_settings={"temperature": 0.6},
        usage_limits={"max_tokens": 150}
    )

    mock_agent_instance.run.assert_awaited_once_with(
        user_prompt="Hello Anthropic assistant",
        message_history=history,
        model_settings={"temperature": 0.6},
        usage_limits={"max_tokens": 150}
    )
    assert response == "Anthropic assistant response"

@pytest.mark.asyncio
async def test_anthropic_adapter_send_message_async_streaming(
    mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_get_provider_config.return_value = {"api_key": "fake_anthropic_key"}
    _, mock_agent_instance = mock_pydantic_ai_agent

    # Pydantic AI's agent.run_stream().stream_structured() yields (ModelResponse, bool) tuples
    async def mock_pydai_stream_structured_generator():
        # Part 1 of the response
        yield PydAI_ModelResponse(parts=[PydAI_TextPart(content="Anthropic says: ")]), False
        # Part 2 of the response, and it's the last one
        yield PydAI_ModelResponse(parts=[PydAI_TextPart(content="Hello!")]), True

    mock_streamed_run_result = MagicMock(spec=StreamedRunResult)
    # Mock the stream_structured method to return our async generator
    mock_streamed_run_result.stream_structured = MagicMock(return_value=mock_pydai_stream_structured_generator())
    
    mock_usage = Usage(requests=1, total_tokens=20, response_tokens=10, request_tokens=10)
    mock_streamed_run_result.usage = MagicMock(return_value=mock_usage)

    # Configure the mock agent's run_stream method
    # run_stream() returns an async context manager.
    # The __aenter__ of that context manager returns mock_streamed_run_result.
    mock_agent_instance.run_stream.return_value.__aenter__.return_value = mock_streamed_run_result

    adapter = AnthropicAdapter(model_name="claude-3-opus-20240229", api_key="fake_anthropic_key")
    
    history: Sequence[ModelMessage] = []
    response_generator = await adapter.send_message_async(
        user_prompt="Stream Anthropic hello",
        history=history,
        stream=True
    )

    # mock_agent_instance.run_stream.assert_called_once() # MOVED: Assertion must be after generator consumption
    
    assert hasattr(response_generator, '__aiter__')
    events_received: list[AnyStreamEvent] = [event async for event in response_generator]

    # NOW assert that run_stream was called, after the generator has been consumed.
    mock_agent_instance.run_stream.assert_called_once()

    # Adapter should yield StreamStart, 2x TextDelta (from adapter's logic), FinalOutput, StreamEnd
    assert len(events_received) == 5 
    
    assert isinstance(events_received[0], StreamStartEvent)
    assert events_received[0].metadata == {"model": "claude-3-opus-20240229", "provider": "anthropic"}

    assert isinstance(events_received[1], TextDeltaEvent)
    assert events_received[1].delta == "Anthropic says: "

    assert isinstance(events_received[2], TextDeltaEvent)
    assert events_received[2].delta == "Hello!"

    assert isinstance(events_received[3], FinalOutputEvent)
    assert events_received[3].output == "Anthropic says: Hello!" # Adapter accumulates this

    assert isinstance(events_received[4], StreamEndEvent)
    assert events_received[4].final_usage == {
        'requests': 1, 'request_tokens': 10, 'response_tokens': 10, 'total_tokens': 20, 'details': None
    }
    
    # Check that stream_structured on the mock_streamed_run_result was called
    mock_streamed_run_result.stream_structured.assert_called_once()

    # Check arguments passed to agent.run_stream
    called_args, called_kwargs = mock_agent_instance.run_stream.call_args
    assert called_kwargs['user_prompt'] == "Stream Anthropic hello"
    assert called_kwargs['message_history'] == [] 
    assert 'model_settings' in called_kwargs
    assert 'usage_limits' in called_kwargs
    expected_run_stream_kwargs = {"user_prompt", "message_history", "model_settings", "usage_limits"}
    for kwarg_key in called_kwargs:
        assert kwarg_key in expected_run_stream_kwargs
   