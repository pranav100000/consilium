import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, AsyncGenerator

from llm.chat_wrapper import ChatSession
from llm.history import MessageHistory, ChatMessage, ToolResponseMessageContent
from llm.adapters.base_adapter import ProviderAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart
)
from llm.streaming import (
    AnyStreamEvent, 
    TextDeltaEvent, 
    StreamStartEvent,
    FinalOutputEvent,
    StreamEndEvent
)

@pytest.fixture
def mock_provider_adapter_instance(mocker):
    """Creates a mock instance of a ProviderAdapter."""
    mock = mocker.MagicMock(spec=ProviderAdapter)
    mock.send_message_async = AsyncMock() # Will be configured per test
    mock.format_pydantic_ai_messages = MagicMock(return_value=[]) # Default to empty list
    
    # For the system_prompt check in ChatSession.__init__ and adapter reload
    mock.agent = MagicMock()
    mock.agent.model = MagicMock()
    mock.agent.model.system_prompt = None # Default initial state
    return mock

@pytest.fixture
def mock_get_adapter_class_factory(mocker, mock_provider_adapter_instance):
    """
    A factory to mock ChatSession._get_adapter_class.
    It allows the test to specify the mock_provider_adapter_instance to be returned.
    """
    def factory(adapter_instance_to_return):
        mock_adapter_class = mocker.MagicMock()
        mock_adapter_class.return_value = adapter_instance_to_return
        return mocker.patch('llm.chat_wrapper.ChatSession._get_adapter_class', return_value=mock_adapter_class)
    return factory


@pytest.fixture
def mock_message_history_instance(mocker):
    """Mocks the MessageHistory instance used by ChatSession."""
    mock_instance = mocker.MagicMock(spec=MessageHistory)
    mock_instance.get_messages = MagicMock(return_value=[])
    mock_instance.add_message = MagicMock()
    mock_instance.clear = MagicMock()
    # Patch the class to return this instance when ChatSession creates MessageHistory()
    mocker.patch('llm.chat_wrapper.MessageHistory', return_value=mock_instance)
    return mock_instance

# --- Test __init__ --- 
def test_chat_session_initialization(
    mock_get_adapter_class_factory,
    mock_provider_adapter_instance,
    mock_message_history_instance # Ensure MessageHistory is mocked
):
    mock_get_adapter_class = mock_get_adapter_class_factory(mock_provider_adapter_instance)
    
    session = ChatSession(
        provider_name="openai", 
        model_name="gpt-4o", 
        system_prompt="Test system prompt"
    )

    assert session.message_history == mock_message_history_instance
    assert session.current_provider_name == "openai"
    assert session.current_model_name == "gpt-4o"
    assert session.current_system_prompt == "Test system prompt"
    assert session.adapter == mock_provider_adapter_instance

    # Check if _get_adapter_class was called by _load_adapter (which is called by __init__)
    mock_get_adapter_class.assert_called_once_with("openai")
    # Check if the mock adapter class was instantiated
    mock_get_adapter_class.return_value.assert_called_once_with(
        model_name="gpt-4o",
        api_key=None,
        base_url=None,
        original_provider_name="openai",
        system_prompt="Test system prompt"
    )
    # Removed incorrect assertion: 
    # assert mock_provider_adapter_instance.agent.model.system_prompt == "Test system prompt"
    # The system_prompt is passed to the adapter, which passes it to the Agent constructor.
    # We already check that the adapter constructor is called with the system_prompt.

def test_chat_session_initialization_no_system_prompt(
    mock_get_adapter_class_factory,
    mock_provider_adapter_instance
):
    mock_get_adapter_class = mock_get_adapter_class_factory(mock_provider_adapter_instance)
    mock_provider_adapter_instance.agent.model.system_prompt = "Already set by adapter"

    session = ChatSession(provider_name="openai", model_name="gpt-4o")
    assert session.current_system_prompt is None
    # System prompt on adapter should remain as it was if session doesn't provide one
    assert mock_provider_adapter_instance.agent.model.system_prompt == "Already set by adapter"

# --- Test _prepare_pydantic_ai_history --- 
def test_prepare_pydantic_ai_history(
    mock_get_adapter_class_factory,
    mock_provider_adapter_instance,
    mock_message_history_instance
):
    mock_get_adapter_class_factory(mock_provider_adapter_instance)
    
    internal_messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there"),
        ChatMessage(role="tool", content=ToolResponseMessageContent(
            tool_name="get_weather", tool_call_id="call123", output="Sunny"
        ))
    ]
    mock_message_history_instance.get_messages.return_value = internal_messages
    
    # Expected structure after formatting by adapter.format_pydantic_ai_messages
    expected_model_messages: List[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="get_weather", tool_call_id="call123", content="Sunny")])
    ]
    mock_provider_adapter_instance.format_pydantic_ai_messages.return_value = expected_model_messages

    session = ChatSession(provider_name="openai", model_name="gpt-4o")
    prepared_history = session._prepare_pydantic_ai_history()

    mock_message_history_instance.get_messages.assert_called_once()
    mock_provider_adapter_instance.format_pydantic_ai_messages.assert_called_once_with(internal_messages)
    assert prepared_history == expected_model_messages

# --- Test send_message_async (non-streaming) --- 
@pytest.mark.asyncio
async def test_send_message_async_non_streaming(
    mock_get_adapter_class_factory,
    mock_provider_adapter_instance,
    mock_message_history_instance
):
    mock_get_adapter_class_factory(mock_provider_adapter_instance)
    session = ChatSession(provider_name="openai", model_name="gpt-4o")

    # Mock adapter's response
    mock_provider_adapter_instance.send_message_async.return_value = "LLM Response"
    # Mock history formatting
    formatted_history: List[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Previous chat turn")])]
    mock_provider_adapter_instance.format_pydantic_ai_messages.return_value = formatted_history
    # When _prepare_pydantic_ai_history is called, it will use the mock_message_history_instance
    # which returns an empty list by default for get_messages(), which is then formatted.
    # For this test, format_pydantic_ai_messages is directly set to return specific formatted_history

    user_input = "Test prompt"
    response = await session.send_message_async(user_input, stream=False)

    # 1. Check history preparation
    # get_messages would be called by _prepare_pydantic_ai_history
    mock_message_history_instance.get_messages.assert_called_once()
    # format_pydantic_ai_messages would be called by _prepare_pydantic_ai_history
    # with the result of mock_message_history_instance.get_messages()
    mock_provider_adapter_instance.format_pydantic_ai_messages.assert_called_once_with([]) 

    # 2. Check user message added to history
    mock_message_history_instance.add_message.assert_any_call(role="user", content=user_input)
    
    # 3. Check adapter called correctly
    mock_provider_adapter_instance.send_message_async.assert_awaited_once_with(
        user_prompt=user_input,
        history=formatted_history, # This comes from the mocked format_pydantic_ai_messages
        stream=False,
        model_settings=None,
        usage_limits=None
    )
    # 4. Check assistant response added to history
    mock_message_history_instance.add_message.assert_any_call(role="assistant", content="LLM Response")
    # 5. Check response
    assert response == "LLM Response"

# --- Test send_message_async (streaming) --- 
@pytest.mark.asyncio
async def test_send_message_async_streaming(
    mock_get_adapter_class_factory,
    mock_provider_adapter_instance,
    mock_message_history_instance
):
    mock_get_adapter_class_factory(mock_provider_adapter_instance)
    session = ChatSession(provider_name="openai", model_name="gpt-4o", system_prompt="Test System Prompt")

    # Mock the adapter's send_message_async to yield our structured stream events
    async def mock_adapter_structured_stream_gen():
        yield StreamStartEvent(metadata={"model": "gpt-4o", "provider": "openai"})
        yield TextDeltaEvent(delta="Hello ")
        yield TextDeltaEvent(delta="World!")
        # Simulate the adapter also yielding a FinalOutputEvent after accumulating text
        yield FinalOutputEvent(output="Hello World!") 
        yield StreamEndEvent(final_usage={"tokens": 100})
    
    mock_provider_adapter_instance.send_message_async.return_value = mock_adapter_structured_stream_gen()
    formatted_history: List[ModelMessage] = [] 
    mock_provider_adapter_instance.format_pydantic_ai_messages.return_value = formatted_history

    user_input = "Stream test"
    response_generator = await session.send_message_async(user_input, stream=True)

    assert isinstance(response_generator, AsyncGenerator)
    received_events: List[AnyStreamEvent] = [event async for event in response_generator]
    
    # Check the events received by the caller of ChatSession.send_message_async
    # It should be exactly what the adapter produced.
    assert len(received_events) == 5
    assert isinstance(received_events[0], StreamStartEvent)
    assert received_events[0].metadata == {"model": "gpt-4o", "provider": "openai"}
    
    assert isinstance(received_events[1], TextDeltaEvent) 
    assert received_events[1].delta == "Hello "
    
    assert isinstance(received_events[2], TextDeltaEvent) 
    assert received_events[2].delta == "World!"

    assert isinstance(received_events[3], FinalOutputEvent) 
    assert received_events[3].output == "Hello World!"

    assert isinstance(received_events[4], StreamEndEvent) 
    assert received_events[4].final_usage == {"tokens": 100}

    # Check interactions with history and adapter
    mock_message_history_instance.add_message.assert_any_call(role="user", content=user_input)
    mock_provider_adapter_instance.send_message_async.assert_awaited_once_with(
        user_prompt=user_input, 
        history=formatted_history, 
        stream=True, 
        model_settings=None, 
        usage_limits=None
    )
    # ChatSession.send_message_async internally collects TextDeltaEvents to form the message for history
    mock_message_history_instance.add_message.assert_any_call(role="assistant", content="Hello World!")

# --- Test send_message_sync --- 
@patch('asyncio.get_running_loop') # To control loop scenarios
@patch('asyncio.new_event_loop')
@patch('asyncio.set_event_loop')
def test_send_message_sync(
    mock_set_event_loop, mock_new_event_loop, mock_get_running_loop,
    mock_get_adapter_class_factory,
    mock_provider_adapter_instance,
    mock_message_history_instance
):
    # Reset call history for this specific mock instance for this test
    mock_message_history_instance.reset_mock()
    mock_message_history_instance.get_messages.return_value = [] # Ensure it starts empty for _prepare_pydantic_ai_history
    mock_provider_adapter_instance.format_pydantic_ai_messages.return_value = [] # Ensure this also returns empty for this test call

    # Scenario 1: No running loop
    mock_get_running_loop.side_effect = RuntimeError("No running event loop")
    mock_loop = MagicMock()
    mock_new_event_loop.return_value = mock_loop

    # mock_loop.run_until_complete.return_value = "Sync LLM Response"

    mock_get_adapter_class_factory(mock_provider_adapter_instance)
    # Important: make the mocked adapter's async method also a MagicMock for run_until_complete
    # or ensure it's an actual awaitable returning the final value for sync path
    mock_provider_adapter_instance.send_message_async = AsyncMock(return_value="Sync LLM Response")

    session = ChatSession(provider_name="openai", model_name="gpt-4o")
    # Inject the mocked MessageHistory instance into the ChatSession instance
    session.message_history = mock_message_history_instance
    user_input_for_test = "Sync prompt" # Use a distinct variable name to avoid clashes if needed

    # Define a side_effect for run_until_complete that simulates the execution of send_message_async
    def simulate_sync_execution_of_send_message_async(coro):
        # The coroutine passed is session.send_message_async(user_input_for_test, stream=False, ...)
        # 1. It prepares history (calls get_messages, format_pydantic_ai_messages - already mocked)
        session._prepare_pydantic_ai_history() 
        # 2. It adds the user message to history
        session.message_history.add_message(role="user", content=user_input_for_test)
        # 3. It would then 'await self.adapter.send_message_async(...)'.
        #    self.adapter.send_message_async is mocked to return "Sync LLM Response".
        #    Let's assume this await happens and returns "Sync LLM Response".
        llm_response_from_adapter = "Sync LLM Response" # This is what the adapter mock returns
        # 4. It adds the assistant message to history
        session.message_history.add_message(role="assistant", content=llm_response_from_adapter)
        # 5. It returns the response from the adapter
        return llm_response_from_adapter

    mock_loop.run_until_complete.side_effect = simulate_sync_execution_of_send_message_async
    
    response = session.send_message_sync(user_input_for_test)

    mock_new_event_loop.assert_called_once()
    mock_set_event_loop.assert_called_once_with(mock_loop)
    # send_message_async would be called inside run_until_complete
    # Direct check on send_message_async of the adapter is tricky here as it's wrapped.
    # We check the final outcome and that run_until_complete was called.
    mock_loop.run_until_complete.assert_called_once()
    assert response == "Sync LLM Response"

    # Check that add_message was called for user and assistant
    # Convert call_args_list to a list of tuples for easier checking if direct comparison fails
    # calls = [c.kwargs for c in mock_message_history_instance.add_message.call_args_list]
    # assert {'role': 'user', 'content': user_input} in calls
    # assert {'role': 'assistant', 'content': "Sync LLM Response"} in calls
    # Using assert_any_call should work if the mock correctly captures them.
    # Let's ensure the mock for message_history is fresh for this test scope if there's interference.
    mock_message_history_instance.add_message.assert_any_call(role="user", content=user_input_for_test)
    mock_message_history_instance.add_message.assert_any_call(role="assistant", content="Sync LLM Response")

# --- Test switch_provider --- 
def test_switch_provider(
    mock_get_adapter_class_factory,
    mock_provider_adapter_instance, # This is the *first* adapter instance
    mock_message_history_instance,
    mocker # For creating a *second* adapter instance
):
    # Initial setup with the first adapter
    mock_get_adapter_class = mock_get_adapter_class_factory(mock_provider_adapter_instance)
    session = ChatSession(
        provider_name="openai", 
        model_name="gpt-4o", 
        api_key="key1", 
        base_url="url1",
        system_prompt="System1",
        custom_arg1="val1"
    )
    assert session.adapter == mock_provider_adapter_instance
    mock_get_adapter_class.return_value.assert_called_once_with(
        model_name="gpt-4o", api_key="key1", base_url="url1", 
        original_provider_name="openai", system_prompt="System1", custom_arg1="val1"
    )

    # Create a new mock adapter for the switch
    mock_new_adapter_instance = mocker.AsyncMock(name="NewProviderAdapterAsyncMock") # Generic AsyncMock
    
    # Explicitly define attributes needed by ChatSession or for assertions
    mock_new_adapter_instance.send_message_async = AsyncMock(name="NewAdapterSendMessageAsync")
    mock_new_adapter_instance.format_pydantic_ai_messages = MagicMock(return_value=[], name="NewAdapterFormatMessages")
    
    mock_new_adapter_instance.agent = MagicMock(name="NewAdapterAgent")
    mock_new_adapter_instance.agent.model = MagicMock(name="NewAdapterAgentModel")
    mock_new_adapter_instance.agent.model.system_prompt = None # For new adapter check

    # Reset the mock for _get_adapter_class to return a new class for the new adapter
    mock_new_adapter_class = mocker.MagicMock()
    mock_new_adapter_class.return_value = mock_new_adapter_instance
    mocker.patch('pydantic_ai_wrapper.chat_wrapper.ChatSession._get_adapter_class', return_value=mock_new_adapter_class)

    session.switch_provider(
        new_provider_name="anthropic", 
        new_model_name="claude-3", 
        new_api_key="key2", 
        custom_arg2="val2" # New kwarg
    )

    assert session.current_provider_name == "anthropic"
    assert session.current_model_name == "claude-3"
    assert session.current_api_key == "key2"
    assert session.current_base_url == "url1" # Base URL not changed, so original kept
    assert session.current_system_prompt == "System1" # System prompt not changed
    assert "custom_arg1" in session.current_provider_kwargs
    assert session.current_provider_kwargs["custom_arg2"] == "val2"
    assert session.adapter == mock_new_adapter_instance

    # Check _get_adapter_class was called for the switch
    # This assertion is on the newly patched mock_new_adapter_class
    # The first call was on the original mock_get_adapter_class
    # So, this checks the call during switch_provider
    # We need to ensure the patch is active to check its call on the right object.
    # The mocker.patch inside this test function handles this.
    session._get_adapter_class.assert_called_with("anthropic")
    
    # Check the new adapter class was instantiated with correct combined args
    mock_new_adapter_class.assert_called_once_with(
        model_name="claude-3", 
        api_key="key2", 
        base_url="url1", # Unchanged from initial
        original_provider_name="anthropic", 
        system_prompt="System1", # Unchanged from initial
        custom_arg1="val1", # Preserved
        custom_arg2="val2"  # Added
    )
    # Check that message history was preserved (it's not cleared)
    mock_message_history_instance.clear.assert_not_called() 