import asyncio
from typing import Optional, Any, Dict, Type, List, Union, AsyncGenerator, Sequence, Tuple

from pydantic_ai.messages import ModelMessage

from app2.sse.sse_queue_manager import SSEQueueManager
from app2.core.logging import get_service_logger

from .history import MessageHistory, ChatMessage
from .adapters.base_adapter import ProviderAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.gemini_adapter import GeminiAdapter
# Import other adapters as they are created, e.g.:
# from .adapters.gemini_adapter import GeminiAdapter

from .config import get_provider_config
from .streaming import AnyStreamEvent, TextDeltaEvent, StreamEndEvent

import json
from json_repair import repair_json

logger = get_service_logger("chat_wrapper")


class ChatSession:
    """
    Manages a chat conversation, including history and provider interaction.
    """

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        queue: SSEQueueManager,
        system_prompt: Optional[str] = None, # Will be passed to Pydantic AI Agent
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tools: Optional[List[Any]] = None,  # Pydantic AI tool functions
        project_id: Optional[str] = None,  # For saving messages to database
        db_session: Optional[Any] = None,  # Database session for saving messages
        request_id: Optional[str] = None,  # Request ID for message metadata
        mode: Optional[str] = None,  # Mode for message saving
        step: Optional[str] = None,  # Generation step for message saving
        **provider_kwargs: Any
    ) -> None:
        """
        Initializes a new chat session.

        Args:
            provider_name: The name of the LLM provider (e.g., "openai", "anthropic").
            model_name: The specific model name to use (e.g., "gpt-4o", "claude-3-opus").
            system_prompt: An optional system prompt for the Pydantic AI agent.
            api_key: Optional API key for the provider.
            base_url: Optional base URL for the provider (e.g., for OpenAI-compatible or Ollama).
            project_id: Optional project ID for saving messages to database.
            db_session: Optional database session for saving messages.
            request_id: Optional request ID for message metadata.
            mode: Optional mode for message saving.
            step: Optional generation step for message saving.
            **provider_kwargs: Additional keyword arguments to pass to the provider adapter's constructor
                                 and subsequently to the Pydantic AI provider setup.
        """
        self.message_history = MessageHistory()
        self.current_provider_name = provider_name
        self.current_model_name = model_name
        self.queue = queue
        self.current_system_prompt = system_prompt
        self.current_api_key = api_key # Store for potential re-use or if adapter needs it explicitly
        self.current_base_url = base_url
        self.current_tools = tools
        self.current_provider_kwargs = provider_kwargs
        self.is_cancelled = False # Initialize cancellation flag
        
        # Database saving context
        self.project_id = project_id
        self.db_session = db_session
        self.request_id = request_id
        self.mode = mode
        self.step = step
        
        # Token tracking
        self.total_request_tokens: int = 0
        self.total_response_tokens: int = 0
        self.total_tokens: int = 0
        self.last_request_tokens: int = 0
        self.last_response_tokens: int = 0

        self.adapter: Optional[ProviderAdapter] = self._load_adapter(
            provider_name,
            model_name,
            api_key,
            base_url,
            **provider_kwargs
        )

    def _get_adapter_class(self, provider_name: str) -> Type[ProviderAdapter]:
        """
        Returns the adapter class based on the provider name.
        This acts as a simple factory.
        """
        # Normalize provider name for matching (e.g., case-insensitive)
        normalized_provider = provider_name.lower()
        
        if normalized_provider == "openai" or \
           normalized_provider.endswith("-openai-compatible") or \
           normalized_provider in ["openrouter", "grok", "perplexity", "fireworks", "together", "ollama"]:
            # Includes common OpenAI-compatible providers by name for clarity
            # and a generic suffix for others.
            return OpenAIAdapter
        elif normalized_provider == "anthropic":
            return AnthropicAdapter
        elif normalized_provider.startswith("gemini") or \
             normalized_provider == "google-gla" or \
             normalized_provider == "google-vertex":
            # ChatSession user can specify "gemini", "google-gla", or "google-vertex"
            # The GeminiAdapter itself will use original_provider_name to pick the backend.
            return GeminiAdapter
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    def _load_adapter(
        self,
        provider_name: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **provider_kwargs: Any
    ) -> ProviderAdapter:
        """
        Loads and initializes the appropriate provider adapter.
        """
        adapter_class = self._get_adapter_class(provider_name)
        
        # Pass original_provider_name so adapter can use it for config lookup if needed
        # e.g. Ollama using OpenAIAdapter might look for OLLAMA_BASE_URL
        kwargs_for_adapter = provider_kwargs.copy()
        kwargs_for_adapter['original_provider_name'] = provider_name 
        
        # If system_prompt is part of current_provider_kwargs, ensure it's passed
        if self.current_system_prompt and 'system_prompt' not in kwargs_for_adapter:
            kwargs_for_adapter['system_prompt'] = self.current_system_prompt
            
        # Pass tools if they exist
        if self.current_tools and 'tools' not in kwargs_for_adapter:
            kwargs_for_adapter['tools'] = self.current_tools

        return adapter_class(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs_for_adapter
        )

    def _prepare_pydantic_ai_history(self) -> List[ModelMessage]:
        """
        Converts the internal chat history to the format expected by Pydantic AI agents,
        using the current adapter's formatting logic.
        This prepares the history *before* adding the current user prompt.
        """
        if not self.adapter:
            raise RuntimeError("Adapter not loaded.")
        return self.adapter.format_pydantic_ai_messages(self.message_history.get_messages())
    
    def cancel(self) -> None:
        """Sets the cancellation flag to stop ongoing streaming."""
        logger.info(f"ChatSession received cancel signal for provider {self.current_provider_name}, model {self.current_model_name}")
        self.is_cancelled = True
    
    async def send_message_async(
        self,
        user_prompt_content: str,
        stream: bool = False,
        model_settings: Optional[Dict[str, Any]] = None,
        usage_limits: Optional[Dict[str, Any]] = None,
        expect_json: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[Any, AsyncGenerator[AnyStreamEvent, None]]:
        """
        Sends a message from the user to the LLM and gets a response, potentially streaming.

        Args:
            user_prompt_content: The text of the user's message.
            stream: Whether to stream the response. Defaults to False.
            model_settings: Optional dictionary of model-specific settings.
            usage_limits: Optional dictionary of usage limits for the request.
            expect_json: If True, will attempt to parse and repair JSON output from the LLM.
            tools: Optional list of Pydantic AI tool functions for the LLM to use.

        Returns:
            If stream is False, returns the complete assistant response content.
            If stream is True, returns an async generator yielding AnyStreamEvent objects.
        """
        if not self.adapter:
            raise RuntimeError("ChatSession adapter not initialized.")

        # Proactive cancellation check
        if self.is_cancelled:
            logger.info(f"ChatSession ({self.current_model_name}): send_message_async called but session is already cancelled.")
            # For non-streaming, we should indicate failure or raise CancelledError.
            # For streaming, the generator wrapper would normally handle this by yielding nothing or an error.
            # Raising CancelledError is consistent with how ongoing operations are stopped.
            raise asyncio.CancelledError("ChatSession operation cancelled because session was already flagged.")

        try:
            # Prepare history for Pydantic AI *before* adding the current user message to our internal history
            # This history will be passed to the Pydantic AI agent.
            pydantic_ai_formatted_history = self._prepare_pydantic_ai_history()

            # Add the current user's message to our internal history
            self.message_history.add_message(role="user", content=user_prompt_content)

            # Get model_settings and usage_limits from session if not provided, allowing override
            final_model_settings = model_settings # or self.current_model_settings if we store them
            final_usage_limits = usage_limits # or self.current_usage_limits

            response_content_for_history: Any
            if stream:
                async def stream_wrapper() -> AsyncGenerator[AnyStreamEvent, None]:
                    nonlocal response_content_for_history
                    full_response_text_chunks = []
                    last_yielded_text = "" # For calculating true deltas
                    try:
                        # Check cancellation again right before adapter call for streaming
                        if self.is_cancelled:
                            logger.info(f"ChatSession ({self.current_model_name}): stream_wrapper initiated but session is cancelled.")
                            raise asyncio.CancelledError("ChatSession streaming cancelled before adapter call.")

                        async_gen = await self.adapter.send_message_async(
                            user_prompt=user_prompt_content,
                            history=pydantic_ai_formatted_history,
                            stream=True,
                            model_settings=final_model_settings,
                            usage_limits=final_usage_limits,
                            tools=tools
                        )
                        if not hasattr(async_gen, '__aiter__'):
                            raise TypeError("Adapter did not return an async generator for streaming.")
                            
                        async for event in async_gen:
                            if self.is_cancelled: # Check before processing/yielding each event
                                logger.info("ChatSession streaming cancelled mid-stream (event loop).")
                                break 

                            if isinstance(event, TextDeltaEvent):
                                current_chunk_from_adapter = event.delta
                                # Calculate the actual new delta (new part only)
                                actual_delta_to_yield = current_chunk_from_adapter[len(last_yielded_text):]
                                
                                # Only append the NEW part, not the whole cumulative text
                                if actual_delta_to_yield:
                                    full_response_text_chunks.append(actual_delta_to_yield)
                                    yield actual_delta_to_yield 
                                
                                last_yielded_text = current_chunk_from_adapter 
                            elif isinstance(event, StreamEndEvent):
                                # Capture token usage from StreamEndEvent
                                if event.final_usage:
                                    usage_dict = self._safe_extract_tokens(event.final_usage)
                                    stream_request_tokens = usage_dict.get('request_tokens', 0)
                                    stream_response_tokens = usage_dict.get('response_tokens', 0)
                                    
                                    # Update instance totals
                                    self.last_request_tokens = stream_request_tokens
                                    self.last_response_tokens = stream_response_tokens
                                    self.total_request_tokens += stream_request_tokens
                                    self.total_response_tokens += stream_response_tokens
                                    self.total_tokens = self.total_request_tokens + self.total_response_tokens
                                    
                                    logger.info(f"ChatSession token usage - Request: {stream_request_tokens}, Response: {stream_response_tokens}, Total: {self.total_tokens}") 
                            
                        response_content_for_history = "".join(full_response_text_chunks)
                    except Exception as e_stream_inner: # Catch other errors within the stream_wrapper
                        logger.error(f"ChatSession ({self.current_model_name}): Error during streaming adapter call: {e_stream_inner}", exc_info=True)
                        response_content_for_history = f"Error during streaming: {e_stream_inner}"
                        self.message_history.add_message(role="assistant", content=response_content_for_history)
                        # Save error message to database
                        await self._save_assistant_message_to_db(response_content_for_history, is_tool_response=False)
                        raise # Re-raise to be caught by outer handler
                    else:
                        if full_response_text_chunks:
                             logger.info(f"🔍 Adding streaming assistant message to history (length: {len(response_content_for_history)}): {response_content_for_history[:100]}...")
                             self.message_history.add_message(role="assistant", content=response_content_for_history)
                             # Save assistant message to database
                             await self._save_assistant_message_to_db(response_content_for_history, is_tool_response=expect_json)
                
                return stream_wrapper()
            else: # Not streaming (stream=False)
                adapter_result = await self.adapter.send_message_async(
                    user_prompt=user_prompt_content,
                    history=pydantic_ai_formatted_history,
                    stream=False,
                    model_settings=final_model_settings,
                    usage_limits=final_usage_limits,
                    tools=tools
                )
                
                # Handle the new return format from adapters
                if isinstance(adapter_result, tuple) and len(adapter_result) == 2:
                    response_content_for_history, usage_info = adapter_result
                    if usage_info:
                        usage_dict = self._safe_extract_tokens(usage_info)
                        request_tokens = usage_dict.get('request_tokens', 0)
                        response_tokens = usage_dict.get('response_tokens', 0)
                        
                        self.last_request_tokens = request_tokens
                        self.last_response_tokens = response_tokens
                        self.total_request_tokens += request_tokens
                        self.total_response_tokens += response_tokens
                        self.total_tokens = self.total_request_tokens + self.total_response_tokens
                        
                        logger.info(f"ChatSession non-streaming token usage - Request: {request_tokens}, Response: {response_tokens}, Total: {self.total_tokens}")
                else:
                    # Fallback for adapters that haven't been updated yet
                    response_content_for_history = adapter_result
                
                # Add the assistant's response to our internal history only if not cancelled during await
                if not self.is_cancelled: # Check after the await
                    logger.info(f"🔍 Adding assistant message to history (length: {len(response_content_for_history)}): {response_content_for_history[:100]}...")
                    self.message_history.add_message(role="assistant", content=response_content_for_history)
                    # Save assistant message to database
                    await self._save_assistant_message_to_db(response_content_for_history, is_tool_response=expect_json)
                else:
                    logger.info(f"ChatSession ({self.current_model_name}): Non-streaming call cancelled during/after adapter await; not adding to history.")
                    # Potentially set response_content_for_history to a specific marker if needed by caller
                    # but _focused_llm_call will likely fail parsing if it's not JSON as expected.

                # JSON processing should only happen if not cancelled and response is valid
                if not self.is_cancelled and expect_json:
                    try:
                        json_object = json.loads(response_content_for_history)
                        return response_content_for_history
                    except json.JSONDecodeError:
                        try:
                            repaired = repair_json(response_content_for_history)
                            json.loads(repaired) 
                            return repaired
                        except Exception as repair_exc:
                            logger.error(f"ChatSession ({self.current_model_name}): JSON repair failed: {repair_exc}")
                            raise
                return response_content_for_history

        except asyncio.CancelledError:
            logger.info(f"ChatSession ({self.current_model_name}): send_message_async task cancelled.")
            self.is_cancelled = True # Ensure flag is set
            # Add a generic cancelled message to history if it makes sense for your flow
            # self.message_history.add_message(role="assistant", content="[Operation Cancelled]")
            raise
        except Exception as e_outer:
            logger.error(f"ChatSession ({self.current_model_name}): Unhandled error in send_message_async: {e_outer}", exc_info=True)
            # Add error to history if not already handled by more specific blocks
            if not isinstance(e_outer, asyncio.CancelledError): # Avoid double-logging if already handled as CancelledError
                 self.message_history.add_message(role="assistant", content=f"Error in ChatSession: {e_outer}")
            raise

    def send_message_sync(
        self,
        user_prompt_content: str,
        model_settings: Optional[Dict[str, Any]] = None,
        usage_limits: Optional[Dict[str, Any]] = None,
        expect_json: bool = False,
    ) -> Any:
        """
        Sends a message synchronously and gets a complete response.
        Streaming is not supported in the sync version.

        Args:
            user_prompt_content: The text of the user's message.
            model_settings: Optional dictionary of model-specific settings.
            usage_limits: Optional dictionary of usage limits for the request.
            expect_json: If True, will attempt to parse and repair JSON output from the LLM.

        Returns:
            The complete assistant response content.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No running event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.send_message_async(
                    user_prompt_content=user_prompt_content,
                    stream=False,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                    expect_json=expect_json
                )
            )
        else: # Existing loop
            # If called from within an async context that has a running loop,
            # but this sync function is blocking it.
            # This is a common issue. For true sync behavior in an async world,
            # one might run the async call in a separate thread.
            # However, Pydantic AI's `run_sync` for agents does something similar to this.
            return loop.run_until_complete(
                 self.send_message_async(
                    user_prompt_content=user_prompt_content,
                    stream=False,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                    expect_json=expect_json
                )
            )

    def switch_provider(
        self,
        new_provider_name: str,
        new_model_name: str,
        new_api_key: Optional[str] = None,
        new_base_url: Optional[str] = None,
        # We could also allow changing system_prompt on switch, but for now, it uses the session's current_system_prompt
        # new_system_prompt: Optional[str] = None, 
        **new_provider_kwargs: Any
    ) -> None:
        """
        Switches the LLM provider and/or model being used for the chat session.
        The existing message history is preserved.

        Args:
            new_provider_name: The name of the new LLM provider.
            new_model_name: The specific model name for the new provider.
            new_api_key: Optional API key for the new provider. If None, will try to use existing or env vars.
            new_base_url: Optional base URL for the new provider.
            **new_provider_kwargs: Additional keyword arguments for the new provider adapter.
        """
        logger.info(f"Switching provider to {new_provider_name} with model {new_model_name}")

        # Update current provider details
        self.current_provider_name = new_provider_name
        self.current_model_name = new_model_name
        
        # If new_api_key or new_base_url are explicitly provided, they should override previous ones.
        # If they are None, the _load_adapter method (and subsequently the adapter itself)
        # will try to use any existing self.current_api_key/base_url or fetch from config.
        # So, we should update self.current_api_key and self.current_base_url if new values are given.
        if new_api_key is not None:
            self.current_api_key = new_api_key
        if new_base_url is not None:
            self.current_base_url = new_base_url
        
        # Update provider_kwargs, new ones take precedence
        self.current_provider_kwargs.update(new_provider_kwargs)

        # Load the new adapter
        # _load_adapter will use the updated self.current_ values including system_prompt
        self.adapter = self._load_adapter(
            provider_name=self.current_provider_name,
            model_name=self.current_model_name,
            api_key=self.current_api_key, # Pass the potentially updated current key
            base_url=self.current_base_url, # Pass the potentially updated current base_url
            **self.current_provider_kwargs
        )

        # The existing self.message_history is preserved and will be used by the new adapter.
        logger.info(f"Provider switched successfully to {self.current_provider_name}.")

    def get_token_usage(self) -> Dict[str, int]:
        """Get the accumulated token usage for this session."""
        return {
            "request_tokens": self.total_request_tokens,
            "response_tokens": self.total_response_tokens,
            "total_tokens": self.total_tokens
        }

    def get_last_call_usage(self) -> Dict[str, int]:
        """Get token usage for the last LLM call only."""
        return {
            "request_tokens": self.last_request_tokens,
            "response_tokens": self.last_response_tokens,
            "total_tokens": self.last_request_tokens + self.last_response_tokens
        }

    def reset_token_usage(self) -> None:
        """Reset all token counters to zero."""
        self.total_request_tokens = 0
        self.total_response_tokens = 0
        self.total_tokens = 0
        self.last_request_tokens = 0
        self.last_response_tokens = 0

    def _safe_extract_tokens(self, usage_data: Any) -> Dict[str, int]:
        """Safely extract token counts from various usage data formats."""
        default = {"request_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        
        if not usage_data:
            return default
            
        try:
            if hasattr(usage_data, 'model_dump'):
                return usage_data.model_dump()
            elif hasattr(usage_data, 'dict'):
                return usage_data.dict()
            elif isinstance(usage_data, dict):
                return usage_data
            else:
                # Try to extract known attributes
                return {
                    "request_tokens": getattr(usage_data, 'request_tokens', 0),
                    "response_tokens": getattr(usage_data, 'response_tokens', 0),
                    "total_tokens": getattr(usage_data, 'total_tokens', 0)
                }
        except Exception as e:
            logger.warning(f"Failed to extract token usage: {e}")
            return default

    async def _save_assistant_message_to_db(self, content: str, is_tool_response: bool = False) -> None:
        """Save assistant message to database if context is available"""
        logger.info(f"🔍 CHAT SESSION: About to save message")
        logger.info(f"🔍 CHAT SESSION: project_id: {self.project_id}")
        logger.info(f"🔍 CHAT SESSION: request_id: {self.request_id}")
        logger.info(f"🔍 CHAT SESSION: mode: {self.mode}")
        logger.info(f"🔍 CHAT SESSION: step: {self.step}")
        logger.info(f"🔍 CHAT SESSION: is_tool_response: {is_tool_response}")
        logger.info(f"🔍 CHAT SESSION: content preview: {content[:100]}...")
        
        if not all([self.project_id, self.db_session]):
            logger.warning(f"🔍 CHAT SESSION: Skipping DB save - missing project_id: {self.project_id}, db_session: {self.db_session is not None}")
            return
        
        # Determine the appropriate step name
        step_to_save = self.step
        if is_tool_response and self.step and not self.step.endswith('_TOOL'):
            step_to_save = f"{self.step}_TOOL"
            logger.info(f"🔍 CHAT SESSION: Tool response detected, using step: {step_to_save}")
            
        try:
            import uuid
            from app2.services.assistant_message_service import AssistantMessageService
            
            message_service = AssistantMessageService(self.db_session)
            await message_service.save_assistant_message(
                project_id=uuid.UUID(self.project_id),
                request_id=self.request_id or "unknown",
                content=content,
                mode=self.mode or 'chat',
                step=step_to_save,
                metadata={
                    'model': self.current_model_name,
                    'provider': self.current_provider_name,
                    'tokens': self.get_last_call_usage()
                }
            )
            logger.info(f"💾 CHAT SESSION: Successfully saved assistant message to DB")
            logger.info(f"💾 CHAT SESSION: Saved to project_id: {self.project_id}")
            logger.info(f"💾 CHAT SESSION: With mode: {self.mode}, step: {self.step}")
            
        except Exception as e:
            # Don't break chat flow if DB save fails
            logger.error(f"Failed to save assistant message to DB: {e}")

    # --- End of class --- 