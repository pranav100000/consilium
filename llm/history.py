from typing import List, Literal, Any, Union
from pydantic import BaseModel, field_validator

class ToolResponseMessageContent(BaseModel):
    """Content model for a ChatMessage with role='tool'."""
    tool_name: str
    tool_call_id: str
    output: Any # The actual result from the tool execution

class ChatMessage(BaseModel):
    """
    Represents a single message in the chat history.
    - role: The originator of the message (user, assistant, system, or tool).
    - content: The content of the message. 
               For role='tool', this should be ToolResponseMessageContent.
               For other roles, can be string or other structured data.
    """
    role: Literal["user", "assistant", "system", "tool"]
    content: Any # Kept as Any for flexibility, but adapter will check type for role='tool'
    # Alternatively, for stricter typing, content could be a Union based on role,
    # but that adds more complexity to ChatMessage construction.
    # Example for stricter typing (more complex):
    # content: Union[
    #     str, # For user, assistant (simple text), system
    #     ToolResponseMessageContent, # For tool
    #     Dict[str, Any] # For assistant (tool call dict before execution)
    # ]

    @field_validator('role')
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        """Validates the role field."""
        allowed_roles = ["user", "assistant", "system", "tool"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}, got '{v}'")
        return v
    
    # We could add a root_validator to ensure content matches role if we wanted to enforce it here
    # e.g., if role == "tool", content must be ToolResponseMessageContent
    # from pydantic import root_validator
    # @root_validator(pre=False, skip_on_failure=True) # 'pre=False' to run after individual field validation
    # def check_content_type_for_role(cls, values):
    #     role, content = values.get('role'), values.get('content')
    #     if role == "tool" and not isinstance(content, ToolResponseMessageContent):
    #         raise ValueError("For role='tool', content must be an instance of ToolResponseMessageContent")
    #     # Add other role/content checks if necessary
    #     return values

class MessageHistory:
    """
    Manages a sequence of chat messages.
    """
    _messages: List[ChatMessage]

    def __init__(self) -> None:
        """Initializes an empty message history."""
        self._messages = []

    def add_message(self, role: Literal["user", "assistant", "system", "tool"], content: Any) -> None:
        """
        Creates a ChatMessage from the provided role and content, and adds it to the history.
        """
        message = ChatMessage(role=role, content=content)
        self._messages.append(message)

    def add_chat_message(self, message: ChatMessage) -> None:
        """
        Adds a pre-constructed ChatMessage object to the history.
        """
        if not isinstance(message, ChatMessage):
            raise TypeError("message must be an instance of ChatMessage")
        self._messages.append(message)

    def get_messages(self) -> List[ChatMessage]:
        """
        Returns a shallow copy of the list of ChatMessage objects.
        Modifying the returned list will not affect the internal history,
        but modifying the ChatMessage objects themselves will.
        """
        return self._messages.copy()

    def get_messages_as_dicts(self) -> List[dict]:
        """
        Returns a list of messages, where each message is represented as a dictionary.
        """
        return [message.model_dump() for message in self._messages]

    def clear(self) -> None:
        """Clears all messages from the history."""
        self._messages = []

    def __len__(self) -> int:
        """Returns the number of messages in the history."""
        return len(self._messages)

    def __iter__(self):
        """Allows iteration over the messages in the history."""
        return iter(self._messages)

    def __getitem__(self, index: int) -> ChatMessage:
        """Allows accessing messages by index."""
        return self._messages[index] 