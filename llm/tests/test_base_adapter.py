import pytest
from typing import List, Any, Sequence, AsyncGenerator, Dict, Optional

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart
)

from llm.adapters.base_adapter import ProviderAdapter
from llm.history import ChatMessage, ToolResponseMessageContent

from unittest.mock import MagicMock

# Create a minimal concrete implementation of ProviderAdapter for testing
class ConcreteTestAdapter(ProviderAdapter):
    def __init__(self, model_name: str = "test_model", **kwargs):
        # Call super().__init__ but handle agent initialization carefully as it calls _initialize_agent
        # We can defer actual agent creation or mock parts of it if super() needs it.
        # For testing format_pydantic_ai_messages, we don't need a real agent.
        # Let's provide a minimal super().__init__ call.
        self.model_name = model_name
        self.api_key = None
        self.base_url = None
        self.system_prompt = None
        self.provider_kwargs = kwargs
        # Normally self.agent = self._initialize_agent() is called here.
        # For this test class, we can skip it or make _initialize_agent trivial.
        self.agent = MagicMock(spec=Agent) # Mock the agent for this test adapter

    def _initialize_agent(self) -> Agent:
        # Dummy implementation, not used for format_pydantic_ai_messages test
        return MagicMock(spec=Agent) 

    async def send_message_async(
        self,
        user_prompt: str,
        history: Sequence[ModelMessage],
        stream: bool = False,
        model_settings: Optional[Dict[str, Any]] = None,
        usage_limits: Optional[Dict[str, Any]] = None
    ) -> Any:
        # Dummy implementation
        pass

@pytest.fixture
def test_adapter() -> ConcreteTestAdapter:
    return ConcreteTestAdapter()

def test_format_empty_history(test_adapter: ConcreteTestAdapter):
    chat_history: List[ChatMessage] = []
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert result == []

def test_format_user_message(test_adapter: ConcreteTestAdapter):
    chat_history = [ChatMessage(role="user", content="Hello AI")]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    assert result[0].parts[0].content == "Hello AI"

def test_format_assistant_text_message(test_adapter: ConcreteTestAdapter):
    chat_history = [ChatMessage(role="assistant", content="Hello User")]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == "Hello User"

def test_format_assistant_tool_call_message(test_adapter: ConcreteTestAdapter):
    tool_call_content = {
        "tool_name": "search_web",
        "tool_call_id": "tc123",
        "args": {"query": "Pydantic AI"}
    }
    chat_history = [ChatMessage(role="assistant", content=tool_call_content)]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    part = result[0].parts[0]
    assert isinstance(part, ToolCallPart)
    assert part.tool_name == "search_web"
    assert part.tool_call_id == "tc123"
    assert part.args == {"query": "Pydantic AI"}

def test_format_tool_response_message(test_adapter: ConcreteTestAdapter):
    tool_response_content = ToolResponseMessageContent(
        tool_name="search_web",
        tool_call_id="tc123",
        output="Pydantic AI is a great library!"
    )
    chat_history = [ChatMessage(role="tool", content=tool_response_content)]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    part = result[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    assert part.tool_name == "search_web"
    assert part.tool_call_id == "tc123"
    assert part.content == "Pydantic AI is a great library!"

def test_format_system_message_skipped(test_adapter: ConcreteTestAdapter, capsys):
    chat_history = [ChatMessage(role="system", content="You are a helpful assistant.")]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert result == []
    captured = capsys.readouterr()
    assert "Warning: System role in ChatMessage history" in captured.out

def test_format_unknown_role_skipped(test_adapter: ConcreteTestAdapter, capsys):
    # Create a mock object that quacks like a ChatMessage with an invalid role
    mock_chat_message = MagicMock(spec=ChatMessage)
    mock_chat_message.role = "unknown_role"
    mock_chat_message.content = "Some content"
    
    chat_history = [mock_chat_message]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert result == [] # Expect empty list because the message should be skipped
    captured = capsys.readouterr()
    assert "Warning: Unknown role 'unknown_role'" in captured.out

def test_format_invalid_tool_content_skipped(test_adapter: ConcreteTestAdapter, capsys):
    chat_history = [ChatMessage(role="tool", content={"some_key": "some_value"})]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert result == []
    captured = capsys.readouterr()
    assert "Warning: Tool role in ChatMessage history does not have ToolResponseMessageContent" in captured.out

def test_format_mixed_history(test_adapter: ConcreteTestAdapter):
    chat_history = [
        ChatMessage(role="user", content="Query weather in London"),
        ChatMessage(role="assistant", content={"tool_name": "get_weather", "tool_call_id": "weather_call_1", "args": {"location": "London"}}),
        ChatMessage(role="tool", content=ToolResponseMessageContent(tool_name="get_weather", tool_call_id="weather_call_1", output={"temp_c": 15, "condition": "Cloudy"})),
        ChatMessage(role="assistant", content="The weather in London is 15°C and Cloudy.")
    ]
    result = test_adapter.format_pydantic_ai_messages(chat_history)
    assert len(result) == 4
    
    assert isinstance(result[0], ModelRequest) and isinstance(result[0].parts[0], UserPromptPart)
    assert result[0].parts[0].content == "Query weather in London"
    
    assert isinstance(result[1], ModelResponse) and isinstance(result[1].parts[0], ToolCallPart)
    assert result[1].parts[0].tool_name == "get_weather"
    assert result[1].parts[0].args == {"location": "London"}

    assert isinstance(result[2], ModelRequest) and isinstance(result[2].parts[0], ToolReturnPart)
    assert result[2].parts[0].tool_name == "get_weather"
    assert result[2].parts[0].content == {"temp_c": 15, "condition": "Cloudy"}

    assert isinstance(result[3], ModelResponse) and isinstance(result[3].parts[0], TextPart)
    assert result[3].parts[0].content == "The weather in London is 15°C and Cloudy." 