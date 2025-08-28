from typing import List, Any, AsyncGenerator, Dict, Optional, Sequence, Union, Tuple

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import Pydantic AI specific stream event types
from pydantic_ai.messages import (
    ModelMessage, 
    ModelRequest,
    UserPromptPart,
    PartDeltaEvent as PydAI_PartDeltaEvent,
    PartStartEvent as PydAI_PartStartEvent,
    FinalResultEvent as PydAI_FinalResultEvent,
    TextPart as PydAI_TextPart, # Used by stream_parts()
    TextPartDelta as PydAI_TextPartDelta,
    ToolCallPart as PydAI_ToolCallPart, # Used by stream_parts()
    ToolCallPartDelta as PydAI_ToolCallPartDelta,
    FunctionToolCallEvent as PydAI_FunctionToolCallEvent, # May not be directly yielded by stream_parts
    FunctionToolResultEvent as PydAI_FunctionToolResultEvent, # May not be directly yielded by stream_parts
)

from .base_adapter import ProviderAdapter
from ..config import get_provider_config # To fetch API key if not directly provided
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

class OpenAIAdapter(ProviderAdapter):
    """
    Adapter for OpenAI and OpenAI-compatible providers.
    """

    def _initialize_agent(self) -> Agent:
        """
        Initializes and returns the Pydantic AI Agent configured for OpenAI.
        """
        # Fetch config from environment if api_key is not directly provided
        # provider_kwargs might also contain api_key or base_url from ChatSession init
        provider_name_for_config = self.provider_kwargs.get("original_provider_name", "openai")
        
        effective_api_key = self.api_key
        if not effective_api_key:
            cfg = get_provider_config(provider_name_for_config)
            effective_api_key = cfg.get('api_key')

        effective_base_url = self.base_url # From constructor
        if not effective_base_url:
            cfg = get_provider_config(provider_name_for_config) # Fetch again if base_url might be there
            effective_base_url = cfg.get('base_url')

        # Prepare provider arguments, prioritizing direct args, then env config, then defaults
        openai_provider_args = {}
        if effective_api_key:
            openai_provider_args['api_key'] = effective_api_key
        if effective_base_url:
            openai_provider_args['base_url'] = effective_base_url
        
        # Add any other OpenAIProvider specific kwargs from self.provider_kwargs
        # For example, one might pass a custom `openai_client` via provider_kwargs
        for key, value in self.provider_kwargs.items():
            if key not in ['api_key', 'base_url', 'original_provider_name', 'system_prompt']: # Avoid overriding already set ones
                 # This needs to be more specific to OpenAIProvider valid args
                 # For now, we assume they are passed correctly if intended for OpenAIProvider
                openai_provider_args[key] = value

        provider = OpenAIProvider(**openai_provider_args)
        
        # self.model_name here should be just the model identifier, e.g., "gpt-4o"
        # For OpenAI-compatible, it could be something like "mistralai/Mixtral-8x7B-Instruct-v0.1"
        model = OpenAIModel(model_name=self.model_name, provider=provider)
        
        # Use self.system_prompt if provided
        agent_kwargs = {}
        if self.system_prompt:
            agent_kwargs['instructions'] = self.system_prompt # Pydantic AI recommends 'instructions'
        
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
        Sends messages to the OpenAI LLM and gets a response.
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call _initialize_agent first.")

        common_params = {
            "user_prompt": user_prompt,
            "message_history": history or [], # Pydantic AI expects a list
            "model_settings": model_settings,
            "usage_limits": usage_limits
        }
        
        # Add tools if provided (Pydantic AI uses 'toolsets' not 'tools')
        if tools:
            common_params["toolsets"] = [tools] if not isinstance(tools[0], list) else tools

        if stream:
            async def stream_generator() -> AsyncGenerator[AnyStreamEvent, None]:
                final_usage_stats: Optional[Dict[str, Any]] = None
                yield StreamStartEvent(metadata={"model": self.model_name, "provider": "openai"})
                
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
                           if hasattr(usage_data, 'model_dump'): # Pydantic v2
                               final_usage_stats = usage_data.model_dump()
                           elif hasattr(usage_data, 'dict'): # Pydantic v1 compatibility
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
                                   "details": getattr(usage_data, 'details', None) 
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