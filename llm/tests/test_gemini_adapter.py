import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Sequence

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse, # For mocking stream_structured output
    UserPromptPart,
    TextPart as PydAI_TextPart, # For mocking stream_structured output
    # ToolCallPart can be added if testing tool use
)
from pydantic_ai.result import StreamedRunResult # Corrected import for StreamedRunResult
from pydantic_ai.usage import Usage # For mocking usage

from llm.adapters.gemini_adapter import GeminiAdapter
from llm.streaming import (
    StreamStartEvent,
    TextDeltaEvent,
    FinalOutputEvent,
    StreamEndEvent,
    AnyStreamEvent,
    # ToolCallStartEvent, ToolCallEndEvent for future tool tests
)

# Mock fixtures for Gemini components
@pytest.fixture
def mock_gemini_model(mocker):
    return mocker.patch('llm.adapters.gemini_adapter.GeminiModel', autospec=True)

@pytest.fixture
def mock_gla_provider(mocker):
    return mocker.patch('llm.adapters.gemini_adapter.GoogleGLAProvider', autospec=True)

@pytest.fixture
def mock_vertex_provider(mocker):
    return mocker.patch('llm.adapters.gemini_adapter.GoogleVertexProvider', autospec=True)

@pytest.fixture
def mock_pydantic_ai_agent(mocker):
    mock_agent_instance = mocker.AsyncMock(spec=Agent)

    # Setup for run_stream (for streaming tests)
    async_context_manager = mocker.AsyncMock(name="run_stream_async_context_manager_gemini")
    mock_agent_instance.run_stream = MagicMock(
        name="run_stream_method_mock_gemini", 
        return_value=async_context_manager
    )
    
    # Setup for run (for non-streaming tests)
    mock_agent_instance.run = AsyncMock(name="run_method_mock_gemini")

    mock_agent_class = mocker.patch('llm.adapters.gemini_adapter.Agent', 
                                  return_value=mock_agent_instance, 
                                  autospec=True)
    return mock_agent_class, mock_agent_instance

@pytest.fixture
def mock_get_provider_config(mocker):
    # Configure to return an empty dict by default to avoid interference
    # Individual tests can override this with .return_value
    return mocker.patch('llm.adapters.gemini_adapter.get_provider_config', return_value={})

# --- Tests for _initialize_agent --- 
@pytest.mark.asyncio
async def test_gemini_adapter_initialize_agent_google_gla_from_config(
    mock_gemini_model, mock_gla_provider, mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_get_provider_config.return_value = {"api_key": "gla_config_key"}
    mock_agent_class, _ = mock_pydantic_ai_agent

    adapter = GeminiAdapter(
        model_name="gemini-pro",
        system_prompt="You are a Gemini assistant.",
        original_provider_name="google-gla" # Passed by ChatSession
    )

    mock_get_provider_config.assert_called_once_with("google-gla")
    mock_gla_provider.assert_called_once_with(api_key="gla_config_key")
    mock_gemini_model.assert_called_once_with(model_name="gemini-pro", provider=mock_gla_provider.return_value)
    mock_agent_class.assert_called_once_with(model=mock_gemini_model.return_value, instructions="You are a Gemini assistant.")
    assert adapter.agent is not None

@pytest.mark.asyncio
async def test_gemini_adapter_initialize_agent_google_gla_direct_key(
    mock_gemini_model, mock_gla_provider, mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_agent_class, _ = mock_pydantic_ai_agent
    # Ensure get_provider_config doesn't interfere when direct key is given
    mock_get_provider_config.return_value = {"api_key": "some_other_config_key"}

    adapter = GeminiAdapter(
        model_name="gemini-pro",
        api_key="direct_gla_key", # Direct key takes precedence
        original_provider_name="google-gla"
    )
    mock_gla_provider.assert_called_once_with(api_key="direct_gla_key")
    # Check get_provider_config was called (our logic calls it), but its value wasn't used for api_key
    mock_get_provider_config.assert_called_with("google-gla") 

@pytest.mark.asyncio
async def test_gemini_adapter_initialize_agent_google_vertex_from_adapter_kwargs(
    mock_gemini_model, mock_vertex_provider, mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_agent_class, _ = mock_pydantic_ai_agent
    # Ensure get_provider_config for vertex returns empty so direct kwargs are used
    mock_get_provider_config.return_value = {}

    adapter = GeminiAdapter(
        model_name="gemini-1.5-flash-latest",
        original_provider_name="google-vertex",
        # These are passed into provider_kwargs of GeminiAdapter constructor by ChatSession
        vertex_project_id="test-project-kwarg",
        vertex_region="us-central1-kwarg",
        vertex_service_account_file="/path/to/sa-kwarg.json"
    )
    mock_vertex_provider.assert_called_once_with(
        project_id="test-project-kwarg", 
        region="us-central1-kwarg", 
        service_account_file="/path/to/sa-kwarg.json"
    )
    mock_gemini_model.assert_called_once_with(model_name="gemini-1.5-flash-latest", provider=mock_vertex_provider.return_value)
    mock_agent_class.assert_called_once_with(model=mock_gemini_model.return_value)
    assert adapter.agent is not None

@pytest.mark.asyncio
async def test_gemini_adapter_initialize_agent_google_vertex_from_env_config(
    mock_gemini_model, mock_vertex_provider, mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_get_provider_config.return_value = {
        "project_id": "config-project-env", 
        "region": "config-region-env", 
        "service_account_file": "/config/sa-env.json"
    }
    mock_agent_class, _ = mock_pydantic_ai_agent
    adapter = GeminiAdapter(
        model_name="gemini-1.5-pro-latest",
        original_provider_name="google-vertex"
    )
    mock_get_provider_config.assert_called_once_with("google-vertex")
    mock_vertex_provider.assert_called_once_with(
        project_id="config-project-env", 
        region="config-region-env", 
        service_account_file="/config/sa-env.json"
    )

@pytest.mark.asyncio
async def test_gemini_adapter_initialize_agent_google_vertex_service_account_info(
    mock_gemini_model, mock_vertex_provider, mock_pydantic_ai_agent, mock_get_provider_config
):
    mock_agent_class, _ = mock_pydantic_ai_agent
    # Test with service_account_info dict instead of file path
    sa_info = {"type": "service_account", "project_id": "my-project"} # Simplified example
    adapter = GeminiAdapter(
        model_name="gemini-pro",
        original_provider_name="google-vertex",
        vertex_service_account_info=sa_info
    )
    mock_vertex_provider.assert_called_once_with(service_account_info=sa_info)

@pytest.mark.asyncio
async def test_gemini_adapter_initialize_agent_invalid_backend(
    mock_get_provider_config # Ensure config mock is active
):
    with pytest.raises(ValueError) as excinfo:
        GeminiAdapter(
            model_name="gemini-pro",
            original_provider_name="invalid-backend"
        )
    assert "Unsupported or missing Gemini backend specified" in str(excinfo.value)

# --- Tests for send_message_async --- 
@pytest.mark.asyncio
async def test_gemini_adapter_send_message_async_non_streaming(
    mock_pydantic_ai_agent, mock_get_provider_config, mock_gla_provider # Need a provider for init
):
    # Setup for GLA for this non-streaming test
    mock_get_provider_config.return_value = {"api_key": "fake_gemini_key"}
    _, mock_agent_instance = mock_pydantic_ai_agent
    
    mock_run_result = MagicMock()
    mock_run_result.output = "Gemini assistant response"
    mock_agent_instance.run = AsyncMock(return_value=mock_run_result)

    adapter = GeminiAdapter(model_name="gemini-pro", api_key="fake_gemini_key", original_provider_name="google-gla")
    adapter.agent = mock_agent_instance 

    history: Sequence[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Previous Gemini message")])]
    response = await adapter.send_message_async(
        user_prompt="Hello Gemini assistant",
        history=history,
        stream=False
    )
    mock_agent_instance.run.assert_awaited_once_with(
        user_prompt="Hello Gemini assistant",
        message_history=history,
        model_settings=None,
        usage_limits=None
    )
    assert response == "Gemini assistant response"

@pytest.mark.asyncio
async def test_gemini_adapter_send_message_async_streaming(
    mock_pydantic_ai_agent, mock_get_provider_config, mock_gla_provider # mock_gla_provider might be for agent init
):
    mock_get_provider_config.return_value = {"api_key": "fake_gemini_key"}
    # Assuming mock_pydantic_ai_agent fixture in this file is similar to the one in anthropic_test
    # It should set up mock_agent_instance.run_stream as a MagicMock returning an AsyncMock context manager
    _, mock_agent_instance = mock_pydantic_ai_agent

    # Pydantic AI's agent.run_stream().stream_structured() yields (ModelResponse, bool) tuples
    async def mock_pydai_stream_structured_generator():
        yield ModelResponse(parts=[PydAI_TextPart(content="Gemini says: ")]), False
        yield ModelResponse(parts=[PydAI_TextPart(content="Hello!")]), True

    mock_streamed_run_result = MagicMock(spec=StreamedRunResult)
    mock_streamed_run_result.stream_structured = MagicMock(return_value=mock_pydai_stream_structured_generator())
    
    mock_usage = Usage(requests=1, total_tokens=30, response_tokens=15, request_tokens=15) # Example usage
    mock_streamed_run_result.usage = MagicMock(return_value=mock_usage)

    # Configure the __aenter__ of the AsyncMock context manager returned by mock_agent_instance.run_stream()
    mock_agent_instance.run_stream.return_value.__aenter__.return_value = mock_streamed_run_result

    # Adapter initialization
    adapter = GeminiAdapter(model_name="gemini-pro", api_key="fake_gemini_key", original_provider_name="google-gla")
    # If the fixture doesn't patch Agent for GeminiAdapter, explicitly set the agent
    if not isinstance(adapter.agent, type(mock_agent_instance)): # Check if agent is already the mock
         adapter.agent = mock_agent_instance


    history: Sequence[ModelMessage] = []
    response_generator = await adapter.send_message_async(
        user_prompt="Stream Gemini hello",
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
    assert events_received[0].metadata == {"model": "gemini-pro", "provider": "google-gla"}

    assert isinstance(events_received[1], TextDeltaEvent)
    assert events_received[1].delta == "Gemini says: "

    assert isinstance(events_received[2], TextDeltaEvent)
    assert events_received[2].delta == "Hello!"

    assert isinstance(events_received[3], FinalOutputEvent)
    assert events_received[3].output == "Gemini says: Hello!"

    assert isinstance(events_received[4], StreamEndEvent)
    assert events_received[4].final_usage == {
        'requests': 1, 
        'request_tokens': 15, 
        'response_tokens': 15, 
        'total_tokens': 30, 
        'details': None 
    }
    
    mock_streamed_run_result.stream_structured.assert_called_once()

    called_args, called_kwargs = mock_agent_instance.run_stream.call_args
    assert called_kwargs['user_prompt'] == "Stream Gemini hello"
    assert called_kwargs['message_history'] == []
    assert 'model_settings' in called_kwargs
    assert 'usage_limits' in called_kwargs 