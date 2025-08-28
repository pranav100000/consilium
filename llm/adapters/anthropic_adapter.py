from typing import List, Any, AsyncGenerator, Dict, Optional, Sequence, Union, Tuple

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage, # For history type hint
    ModelRequest,
    ModelResponse, # Ensure this is imported
    UserPromptPart,
    # Pydantic AI specific stream event types (assuming common ones for Anthropic)
    PartDeltaEvent as PydAI_PartDeltaEvent,
    PartStartEvent as PydAI_PartStartEvent,
    FinalResultEvent as PydAI_FinalResultEvent,
    FunctionToolCallEvent as PydAI_FunctionToolCallEvent,
    FunctionToolResultEvent as PydAI_FunctionToolResultEvent,
    TextPart as PydAI_TextPart,
    TextPartDelta as PydAI_TextPartDelta,
    ToolCallPart as PydAI_ToolCallPart,
    ToolCallPartDelta as PydAI_ToolCallPartDelta
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
# from pydantic_ai.stream import StreamedRunResult # Not strictly needed if using stream_text

from .base_adapter import ProviderAdapter
from ..config import get_provider_config
# Import our structured stream event types
from ..streaming import (
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolResultEvent,
    FinalOutputEvent,
    StreamErrorEvent,
    StreamEndEvent,
    AnyStreamEvent,
)

class AnthropicAdapter(ProviderAdapter):
    """
    Adapter for Anthropic models.
    """

    def _initialize_agent(self) -> Agent:
        """
        Initializes and returns the Pydantic AI Agent configured for Anthropic.
        """
        effective_api_key = self.api_key
        if not effective_api_key:
            # original_provider_name might be just "anthropic"
            cfg = get_provider_config(self.provider_kwargs.get("original_provider_name", "anthropic"))
            effective_api_key = cfg.get('api_key')

        if not effective_api_key:
            # Pydantic AI might also pick up ANTHROPIC_API_KEY from env by default if provider is not given key
            # but being explicit is better.
            pass

        anthropic_provider_args = {}
        if effective_api_key:
            anthropic_provider_args['api_key'] = effective_api_key
        
        # Add any other AnthropicProvider specific kwargs from self.provider_kwargs
        for key, value in self.provider_kwargs.items():
            if key not in ['api_key', 'original_provider_name', 'system_prompt']:
                anthropic_provider_args[key] = value

        provider = AnthropicProvider(**anthropic_provider_args)
        model = AnthropicModel(model_name=self.model_name, provider=provider)
        
        agent_kwargs = {}
        if self.system_prompt:
            agent_kwargs['instructions'] = self.system_prompt
        
        if self.tools:
            agent_kwargs['tools'] = self.tools
        
        return Agent(model=model, **agent_kwargs)

    async def send_message_async(
        self,
        user_prompt: str,
        history: Sequence[ModelMessage],
        stream: bool = False,
        model_settings: Optional[Dict[str, Any]] = None,
        usage_limits: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any: # Return type will be Union[RunResult.output, AsyncGenerator[AnyStreamEvent, None]]
        """
        Sends messages to the Anthropic LLM and gets a response.
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call _initialize_agent first.")

        common_params = {
            "user_prompt": user_prompt,
            "message_history": history or [],
            "model_settings": model_settings,
            "usage_limits": usage_limits
        }
        
        # Note: Tools should be passed to Agent constructor, not run method
        # This parameter is kept for future compatibility but not used
        if tools:
            logger.warning("Tools parameter passed to adapter, but tools should be set on Agent constructor")

        if stream:
            async def stream_generator() -> AsyncGenerator[AnyStreamEvent, None]:
                final_usage_stats: Optional[Dict[str, Any]] = None
                yield StreamStartEvent(metadata={"model": self.model_name, "provider": "anthropic"})
                
                turn_accumulated_text = ""
                last_event_was_tool_call = False

                try:
                    async with self.agent.run_stream(**common_params) as response_stream: # type: ignore[pydantic-ai]
                        async for pydai_model_response, is_last_model_response in response_stream.stream_structured():
                            current_model_response_text_parts = [] 
                            
                            for part_idx, part in enumerate(pydai_model_response.parts):
                                if isinstance(part, PydAI_TextPart):
                                    yield TextDeltaEvent(delta=part.content, part_index=part_idx) 
                                    current_model_response_text_parts.append(part.content)
                                    last_event_was_tool_call = False
                                elif isinstance(part, PydAI_ToolCallPart):
                                    if current_model_response_text_parts: 
                                        intermediate_text = "".join(current_model_response_text_parts)
                                        yield FinalOutputEvent(output=intermediate_text)
                                        current_model_response_text_parts = [] 
                                    
                                    yield ToolCallStartEvent(
                                        tool_call_id=part.tool_call_id,
                                        tool_name=part.tool_name,
                                    )
                                    yield ToolCallEndEvent(
                                        tool_call_id=part.tool_call_id,
                                        tool_name=part.tool_name,
                                        args=part.args_as_dict()
                                    )
                                    last_event_was_tool_call = True
                                    turn_accumulated_text = ""
                            
                            if current_model_response_text_parts:
                                if last_event_was_tool_call:
                                    turn_accumulated_text = "".join(current_model_response_text_parts)
                                else:
                                    turn_accumulated_text += "".join(current_model_response_text_parts)
                                last_event_was_tool_call = False
                        
                        if turn_accumulated_text and not last_event_was_tool_call:
                            yield FinalOutputEvent(output=turn_accumulated_text)

                        if hasattr(response_stream, 'usage') and response_stream.usage():
                           usage_data = response_stream.usage()
                           if hasattr(usage_data, 'model_dump'):
                               final_usage_stats = usage_data.model_dump()
                           elif hasattr(usage_data, 'dict'):
                               final_usage_stats = usage_data.dict()
                           elif isinstance(usage_data, dict):
                               final_usage_stats = usage_data
                           else: 
                               # Fallback: Manually construct from known Usage attributes
                               final_usage_stats = {
                                   "requests": getattr(usage_data, 'requests', 0),
                                   "request_tokens": getattr(usage_data, 'request_tokens', 0),
                                   "response_tokens": getattr(usage_data, 'response_tokens', 0),
                                   "total_tokens": getattr(usage_data, 'total_tokens', 0),
                                   "details": getattr(usage_data, 'details', None) # Assuming details might exist
                               }
                except Exception as e:
                    yield StreamErrorEvent(error_message=str(e), error_type=type(e).__name__)
                finally:
                    yield StreamEndEvent(final_usage=final_usage_stats)
            return stream_generator()
        else:
            result = await self.agent.run(**common_params) # type: ignore[pydantic-ai]
            
            # Extract usage information
            usage_dict = None
            if hasattr(result, 'usage') and result.usage():
                usage_data = result.usage()
                if hasattr(usage_data, 'model_dump'):
                    usage_dict = usage_data.model_dump()
                elif hasattr(usage_data, 'dict'):
                    usage_dict = usage_data.dict()
                elif isinstance(usage_data, dict):
                    usage_dict = usage_data
                else:
                    # Fallback: Manually construct from known Usage attributes
                    usage_dict = {
                        "request_tokens": getattr(usage_data, 'request_tokens', 0),
                        "response_tokens": getattr(usage_data, 'response_tokens', 0),
                        "total_tokens": getattr(usage_data, 'total_tokens', 0)
                    }
            
            return (result.output, usage_dict) 