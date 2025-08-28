import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Sequence # Ensure Sequence is imported

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import (
    UserPromptPart,
    ModelMessage, # Import ModelMessage
    ModelRequest, # Import ModelRequest for constructing history
    PartDeltaEvent as PydAI_PartDeltaEvent,
    PartStartEvent as PydAI_PartStartEvent,
    FinalResultEvent as PydAI_FinalResultEvent,
    TextPart as PydAI_TextPart,
    TextPartDelta as PydAI_TextPartDelta,
    ToolCallPart as PydAI_ToolCallPart,
    ModelResponse as PydAI_ModelResponse
)
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.usage import Usage

from llm.adapters.openai_adapter import OpenAIAdapter
from llm.streaming import (
    StreamStartEvent,
    TextDeltaEvent,
    FinalOutputEvent,
    StreamEndEvent,
    AnyStreamEvent
)

# Mock pydantic_ai classes at the module level where OpenAIAdapter imports them
@pytest.fixture
def mock_openai_provider(mocker):
    return mocker.patch('llm.adapters.openai_adapter.OpenAIProvider', autospec=True)

@pytest.fixture
def mock_openai_model(mocker):
    return mocker.patch('llm.adapters.openai_adapter.OpenAIModel', autospec=True)

@pytest.fixture
def mock_pydantic_ai_agent(mocker):
    mock_agent_instance = mocker.AsyncMock(spec=Agent) # The agent instance is an AsyncMock

    # Make run_stream a MagicMock attribute on the AsyncMock agent instance
    # This MagicMock, when called, will return an AsyncMock (the async context manager)
    async_context_manager = mocker.AsyncMock(name="run_stream_async_context_manager_openai")
    mock_agent_instance.run_stream = MagicMock(
        name="run_stream_method_mock_openai", 
        return_value=async_context_manager
    )
    
    mock_agent_instance.run = AsyncMock() # For non-streaming calls
    mock_agent_class = mocker.patch('llm.adapters.openai_adapter.Agent', 
                                  return_value=mock_agent_instance, 
                                  autospec=True)
    return mock_agent_class, mock_agent_instance

@pytest.fixture
def mock_get_provider_config(mocker):
    return mocker.patch('llm.adapters.openai_adapter.get_provider_config')


@pytest.mark.asyncio
async def test_openai_adapter_initialize_agent_with_api_key(
    mock_openai_provider,
    mock_openai_model,
    mock_pydantic_ai_agent,
    mock_get_provider_config
):
    mock_get_provider_config.return_value = {} # Ensure no env vars are picked up unless specified
    mock_agent_class, _ = mock_pydantic_ai_agent

    adapter = OpenAIAdapter(
        model_name="gpt-4o",
        api_key="test_api_key",
        system_prompt="You are helpful."
    )

    mock_openai_provider.assert_called_once_with(api_key="test_api_key")
    mock_openai_model.assert_called_once_with(model_name="gpt-4o", provider=mock_openai_provider.return_value)
    mock_agent_class.assert_called_once_with(model=mock_openai_model.return_value, instructions="You are helpful.")
    assert adapter.agent is not None

@pytest.mark.asyncio
async def test_openai_adapter_initialize_agent_from_config(
    mock_openai_provider,
    mock_openai_model,
    mock_pydantic_ai_agent,
    mock_get_provider_config
):
    mock_get_provider_config.return_value = {"api_key": "config_api_key", "base_url": "http://config.url"}
    mock_agent_class, _ = mock_pydantic_ai_agent

    adapter = OpenAIAdapter(
        model_name="gpt-3.5-turbo",
        original_provider_name="openai" # Passed by ChatSession normally
    )
    
    mock_get_provider_config.assert_any_call("openai") # Called to get api_key and base_url
    mock_openai_provider.assert_called_once_with(api_key="config_api_key", base_url="http://config.url")
    mock_openai_model.assert_called_once_with(model_name="gpt-3.5-turbo", provider=mock_openai_provider.return_value)
    mock_agent_class.assert_called_once_with(model=mock_openai_model.return_value) # No system prompt here
    assert adapter.agent is not None

@pytest.mark.asyncio
async def test_openai_adapter_initialize_agent_ollama_compatible(
    mock_openai_provider,
    mock_openai_model,
    mock_pydantic_ai_agent,
    mock_get_provider_config
):
    # Simulate that ChatSession passes original_provider_name as "ollama"
    mock_get_provider_config.return_value = {"base_url": "http://localhost:11434"}
    mock_agent_class, _ = mock_pydantic_ai_agent

    adapter = OpenAIAdapter(
        model_name="llama3",
        original_provider_name="ollama" # This kwarg is key for config lookup logic
    )

    # Check that get_provider_config was called with "ollama"
    mock_get_provider_config.assert_any_call("ollama")
    mock_openai_provider.assert_called_once_with(base_url="http://localhost:11434")
    mock_openai_model.assert_called_once_with(model_name="llama3", provider=mock_openai_provider.return_value)
    mock_agent_class.assert_called_once_with(model=mock_openai_model.return_value)
    assert adapter.agent is not None

@pytest.mark.asyncio
async def test_openai_adapter_send_message_async_non_streaming(
    mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_get_provider_config.return_value = {"api_key": "fake_key"}
    _, mock_agent_instance = mock_pydantic_ai_agent
    
    # Configure the mock for agent.run()
    mock_run_result = MagicMock()
    mock_run_result.output = "Assistant response"
    mock_agent_instance.run.return_value = mock_run_result # Configure here

    adapter = OpenAIAdapter(model_name="gpt-4", api_key="fake_key")
    # adapter.agent is already the mock_agent_instance due to the fixture patch

    history: Sequence[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Previous message")])]
    response = await adapter.send_message_async(
        user_prompt="Hello assistant",
        history=history,
        stream=False,
        model_settings={"temperature": 0.5},
        usage_limits={"max_tokens": 100}
    )
    mock_agent_instance.run.assert_awaited_once_with(
        user_prompt="Hello assistant",
        message_history=history,
        model_settings={"temperature": 0.5},
        usage_limits={"max_tokens": 100}
    )
    assert response == "Assistant response"

@pytest.mark.asyncio
async def test_openai_adapter_send_message_async_streaming(
    mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_get_provider_config.return_value = {"api_key": "fake_key"}
    _, mock_agent_instance = mock_pydantic_ai_agent

    # Pydantic AI's agent.run_stream().stream_structured() yields (ModelResponse, bool) tuples
    async def mock_pydai_stream_structured_generator(): # Renamed from mock_pydai_modelresponse_generator
        yield PydAI_ModelResponse(parts=[PydAI_TextPart(content="OpenAI says: ")]), False
        yield PydAI_ModelResponse(parts=[PydAI_TextPart(content="Hello!")]), True

    mock_streamed_run_result = MagicMock(spec=StreamedRunResult)
    mock_streamed_run_result.stream_structured = MagicMock(return_value=mock_pydai_stream_structured_generator())
    
    mock_usage = Usage(requests=1, total_tokens=10, response_tokens=5, request_tokens=5) # Example usage
    mock_streamed_run_result.usage = MagicMock(return_value=mock_usage)

    # Configure the __aenter__ of the AsyncMock context manager returned by mock_agent_instance.run_stream()
    mock_agent_instance.run_stream.return_value.__aenter__.return_value = mock_streamed_run_result

    adapter = OpenAIAdapter(model_name="gpt-4", api_key="fake_key")
    # If the fixture doesn't patch Agent for OpenAIAdapter, explicitly set the agent
    if not isinstance(adapter.agent, type(mock_agent_instance)):
         adapter.agent = mock_agent_instance

    history: Sequence[ModelMessage] = []
    response_generator = await adapter.send_message_async(
        user_prompt="Stream OpenAI hello", # Changed prompt for clarity
        history=history,
        stream=True
    )

    assert hasattr(response_generator, '__aiter__')
    events_received: list[AnyStreamEvent] = [event async for event in response_generator]

    # NOW assert that run_stream was called, after the generator has been consumed.
    mock_agent_instance.run_stream.assert_called_once()

    # Expected events: StreamStart, TextDelta, TextDelta, FinalOutput, StreamEnd
    assert len(events_received) == 5
    
    assert isinstance(events_received[0], StreamStartEvent)
    assert events_received[0].metadata == {"model": "gpt-4", "provider": "openai"}

    assert isinstance(events_received[1], TextDeltaEvent)
    assert events_received[1].delta == "OpenAI says: "

    assert isinstance(events_received[2], TextDeltaEvent)
    assert events_received[2].delta == "Hello!"

    assert isinstance(events_received[3], FinalOutputEvent)
    assert events_received[3].output == "OpenAI says: Hello!"

    assert isinstance(events_received[4], StreamEndEvent)
    assert events_received[4].final_usage == {
        'requests': 1, 
        'request_tokens': 5, 
        'response_tokens': 5, 
        'total_tokens': 10, 
        'details': None 
    }
    
    mock_streamed_run_result.stream_structured.assert_called_once()

    called_args, called_kwargs = mock_agent_instance.run_stream.call_args
    assert called_kwargs['user_prompt'] == "Stream OpenAI hello"
    assert called_kwargs['message_history'] == []
    assert 'model_settings' in called_kwargs
    assert 'usage_limits' in called_kwargs 