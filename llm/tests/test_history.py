import pytest
from pydantic import ValidationError

from llm.history import ChatMessage, MessageHistory


def test_chat_message_creation():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

def test_chat_message_invalid_role():
    with pytest.raises(ValidationError) as excinfo:
        ChatMessage(role="invalid_role", content="Test")
    # Check the type of error and the input value for more robustness
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]['type'] == 'literal_error'
    assert errors[0]['input'] == 'invalid_role'
    # We can also check that the expected values are mentioned in the context or message
    assert "'user'" in str(errors[0]['ctx']['expected']) # Example for checking context if needed
    assert "Input should be 'user', 'assistant', 'system' or 'tool'" in errors[0]["msg"]

def test_message_history_initialization():
    history = MessageHistory()
    assert len(history) == 0
    assert history.get_messages() == []
    assert history.get_messages_as_dicts() == []

def test_message_history_add_message():
    history = MessageHistory()
    history.add_message(role="user", content="First message")
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "First message"

    history.add_message(role="assistant", content={"type": "tool_call", "name": "get_weather"})
    assert len(history) == 2
    assert history[1].role == "assistant"
    assert history[1].content == {"type": "tool_call", "name": "get_weather"}

def test_message_history_add_chat_message():
    history = MessageHistory()
    msg1 = ChatMessage(role="system", content="System init")
    history.add_chat_message(msg1)
    assert len(history) == 1
    assert history[0] == msg1

    with pytest.raises(TypeError):
        history.add_chat_message({"role": "user", "content": "fail"}) # type: ignore

def test_message_history_get_messages():
    history = MessageHistory()
    history.add_message(role="user", content="Hello")
    messages = history.get_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessage)
    # Test that it's a copy
    messages.append(ChatMessage(role="user", content="Another")) # type: ignore
    assert len(history.get_messages()) == 1 

def test_message_history_get_messages_as_dicts():
    history = MessageHistory()
    history.add_message(role="user", content="Hello")
    history.add_message(role="assistant", content={"data": 123})
    dicts = history.get_messages_as_dicts()
    assert len(dicts) == 2
    assert dicts[0] == {"role": "user", "content": "Hello"}
    assert dicts[1] == {"role": "assistant", "content": {"data": 123}}

def test_message_history_clear():
    history = MessageHistory()
    history.add_message(role="user", content="Temporary message")
    assert len(history) == 1
    history.clear()
    assert len(history) == 0
    assert history.get_messages() == []

def test_message_history_iteration_and_indexing():
    history = MessageHistory()
    history.add_message(role="user", content="Msg1")
    history.add_message(role="assistant", content="Msg2")

    # Test iteration
    roles = []
    for msg in history:
        roles.append(msg.role)
    assert roles == ["user", "assistant"]

    # Test indexing
    assert history[0].content == "Msg1"
    assert history[1].content == "Msg2"
    with pytest.raises(IndexError):
        _ = history[2] 