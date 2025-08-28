import logging
from abc import ABC, abstractmethod
from typing import List, Any, AsyncGenerator, Dict, Optional, Sequence, Union, Tuple

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage, 
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    SystemPromptPart,
    ToolCallPart, 
    ToolReturnPart
)

from app2.llm.history import ChatMessage, ToolResponseMessageContent

logger = logging.getLogger(__name__)

class ProviderAdapter(ABC):
    """
    Abstract Base Class for provider adapters.
    Each adapter will encapsulate the logic for interacting with a specific
    LLM provider via Pydantic AI.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initializes the adapter with common and provider-specific settings.

        Args:
            model_name: The specific model name to use for this provider.
            api_key: The API key for the provider, if required.
            base_url: The base URL for the provider's API, if not the default.
            system_prompt: An optional system prompt / initial instructions for the agent.
            **kwargs: Additional provider-specific keyword arguments.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.tools = tools
        self.provider_kwargs = kwargs
        self.agent = self._initialize_agent()

    @abstractmethod
    def _initialize_agent(self) -> Agent:
        """
        Initializes and returns the Pydantic AI Agent configured for this provider.
        This method will be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    async def send_message_async(
        self,
        user_prompt: str,
        history: Sequence[ModelMessage],
        stream: bool = False,
        model_settings: Optional[Dict[str, Any]] = None,
        usage_limits: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Union[Any, Tuple[Any, Optional[Dict[str, Any]]]]:
        """
        Sends messages to the LLM and gets a response.

        Args:
            user_prompt: The current prompt or message from the user.
            history: A sequence of Pydantic AI ModelMessage objects representing the prior conversation.
            stream: Whether to stream the response.
            model_settings: Provider-specific model settings.
            usage_limits: Usage limits for the request.
            tools: Optional list of Pydantic AI tool functions for the LLM to use.

        Returns:
            If stream is False, returns tuple of (response_content, usage_dict)
                where usage_dict contains: request_tokens, response_tokens, total_tokens
            If stream is True, returns an AsyncGenerator yielding response events
        """
        pass

    def format_pydantic_ai_messages(self, chat_history: List[ChatMessage]) -> List[ModelMessage]:
        """
        Converts a list of our internal ChatMessage objects to a list of
        Pydantic AI ModelMessage objects suitable for message_history.
        """
        pydantic_ai_history: List[ModelMessage] = []
        for msg in chat_history:
            if msg.role == "user":
                pydantic_ai_history.append(ModelRequest(parts=[UserPromptPart(content=str(msg.content))]))
            elif msg.role == "assistant":
                # Assistant content could be simple text or tool calls.
                # This simplified version assumes text or a dict that can represent a ToolCallPart if needed.
                # For now, let's assume assistant message is primarily text for history.
                # If the assistant response included tool calls, those would be handled by the agent
                # and the subsequent tool results would be in history as ToolReturnPart.
                if isinstance(msg.content, dict) and "tool_name" in msg.content and "args" in msg.content and "tool_call_id" in msg.content:
                    pydantic_ai_history.append(
                        ModelResponse(parts=[
                            ToolCallPart(
                                tool_name=msg.content["tool_name"],
                                args=msg.content["args"],
                                tool_call_id=msg.content["tool_call_id"]
                            )
                        ])
                    )
                else:
                     pydantic_ai_history.append(ModelResponse(parts=[TextPart(content=str(msg.content))]))
            elif msg.role == "system":
                # System messages are typically handled by Agent's instructions/system_prompt,
                # not as part of the iterative message_history like user/assistant messages.
                # However, if one were to be included, it might look like this, but it's unusual for history.
                # pydantic_ai_history.append(ModelRequest(parts=[SystemPromptPart(content=str(msg.content))]))
                # For now, we'll skip adding system messages directly to the formatted history 
                # as they are usually set on the agent itself.
                logger.warning(f"System role in ChatMessage history is usually handled by agent's system_prompt/instructions, not direct history. Skipping: {msg.content}")
                pass
            elif msg.role == "tool":
                if isinstance(msg.content, ToolResponseMessageContent):
                    pydantic_ai_history.append(
                        ModelRequest(parts=[
                            ToolReturnPart(
                                tool_name=msg.content.tool_name,
                                tool_call_id=msg.content.tool_call_id,
                                content=msg.content.output
                            )
                        ])
                    )
                else:
                    logger.warning(f"Tool role in ChatMessage history does not have ToolResponseMessageContent. Skipping: {msg.content}")
            else:
                logger.warning(f"Unknown role '{msg.role}' in ChatMessage history. Skipping: {msg.content}")
        
        return pydantic_ai_history

    # Placeholder for token counting - to be implemented by concrete adapters if possible
    # @abstractmethod
    # async def get_token_count_async(self, messages: List[ChatMessage]) -> int:
    #     pass 