from typing import List, Any, AsyncGenerator, Dict, Optional, Sequence, Union, Tuple

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest, # For constructing history if needed by Pydantic AI Gemini model
    ModelResponse, 
    UserPromptPart, # For constructing history if needed
    TextPart as PydAI_TextPart, 
    ToolCallPart as PydAI_ToolCallPart,
    # Import other Pydantic AI event/part types as needed for streaming if Gemini differs
)
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
# from pydantic_ai.stream import StreamedRunResult # If needed for typing

from .base_adapter import ProviderAdapter
from ..config import get_provider_config
from ..streaming import (
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    FinalOutputEvent,
    StreamErrorEvent,
    StreamEndEvent,
    AnyStreamEvent,
)

class GeminiAdapter(ProviderAdapter):
    """
    Adapter for Google Gemini models (Generative Language API and Vertex AI).
    """

    def _initialize_agent(self) -> Agent:
        """
        Initializes and returns the Pydantic AI Agent configured for Gemini.
        Determines whether to use Generative Language API or Vertex AI based on
        `self.provider_kwargs.get("original_provider_name")` which should be
        either "google-gla" or "google-vertex".
        """
        # original_provider_name is passed by ChatSession via provider_kwargs
        # It dictates which Gemini backend to use.
        gemini_backend = self.provider_kwargs.get("original_provider_name")

        provider_instance: Any

        if gemini_backend == "google-gla":
            gla_config = get_provider_config("google-gla") # Fetches GOOGLE-GLA_API_KEY etc.
            api_key = self.api_key or gla_config.get('api_key')
            
            provider_args = {}
            if api_key:
                provider_args['api_key'] = api_key
            # Add other GoogleGLAProvider specific kwargs from self.provider_kwargs if any
            # (e.g., http_client)
            for key, value in self.provider_kwargs.items():
                if key not in ['api_key', 'original_provider_name', 'system_prompt', 'use_vertex', 'vertex_project_id', 'vertex_region', 'vertex_service_account_file']:
                    provider_args[key] = value
            provider_instance = GoogleGLAProvider(**provider_args)
            
        elif gemini_backend == "google-vertex":
            vertex_config = get_provider_config("google-vertex") # Fetches GCP_PROJECT_ID etc.
            
            provider_args = {}
            # Explicitly passed args to GeminiAdapter constructor via provider_kwargs take precedence
            project_id = self.provider_kwargs.get('vertex_project_id', vertex_config.get('project_id'))
            region = self.provider_kwargs.get('vertex_region', vertex_config.get('region'))
            service_account_file = self.provider_kwargs.get('vertex_service_account_file', vertex_config.get('service_account_file'))
            # Pydantic AI also supports service_account_info (dict)
            service_account_info = self.provider_kwargs.get('vertex_service_account_info')

            if project_id:
                provider_args['project_id'] = project_id
            if region:
                provider_args['region'] = region
            if service_account_file:
                provider_args['service_account_file'] = service_account_file
            elif service_account_info: # Pydantic AI takes one or the other
                provider_args['service_account_info'] = service_account_info
            
            # Add other GoogleVertexProvider specific kwargs
            for key, value in self.provider_kwargs.items():
                if key not in ['api_key', 'original_provider_name', 'system_prompt', 'use_vertex', 
                                'vertex_project_id', 'vertex_region', 'vertex_service_account_file', 'vertex_service_account_info']:
                    provider_args[key] = value
            provider_instance = GoogleVertexProvider(**provider_args)
        else:
            raise ValueError(
                f"Unsupported or missing Gemini backend specified. Expected 'google-gla' or 'google-vertex' "
                f"in provider_kwargs['original_provider_name']. Got: {gemini_backend}"
            )

        # self.model_name is like "gemini-1.5-flash-latest" or "gemini-pro"
        # GeminiModel also needs the provider type string if not inferable (but we pass provider instance)
        model = GeminiModel(model_name=self.model_name, provider=provider_instance) # type: ignore[pydantic-ai]
        
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
    ) -> Any:
        """
        Sends messages to the Gemini LLM and gets a response.
        Uses stream_structured() for streaming, similar to OpenAI and Anthropic adapters.
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized.")
        
        common_params = {
            "user_prompt": user_prompt,
            "message_history": history or [],
            "model_settings": model_settings, # GeminiModelSettings can be passed here
            "usage_limits": usage_limits
        }
        
        # Add tools if provided (Pydantic AI uses 'toolsets' not 'tools')
        if tools:
            common_params["toolsets"] = [tools] if not isinstance(tools[0], list) else tools

        if stream:
            async def stream_generator() -> AsyncGenerator[AnyStreamEvent, None]:
                final_usage_stats: Optional[Dict[str, Any]] = None
                # Pass original_provider_name for metadata consistency
                provider_display_name = self.provider_kwargs.get("original_provider_name", "gemini")
                yield StreamStartEvent(metadata={"model": self.model_name, "provider": provider_display_name})
                
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
                           # Add similar debug prints if needed, then remove
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