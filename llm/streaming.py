from typing import Literal, Any, Dict, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone

StreamEventType = Literal[
    "text_delta",
    "tool_call_start", # Model decides to call a tool
    "tool_call_chunk", # For streaming arguments of a tool call if applicable
    "tool_call_end",   # Marks the end of a tool call decision from the model
    "tool_result",     # Result from executing a tool
    "final_output",    # The final complete message from the assistant
    "stream_error",
    "stream_start",
    "stream_end"
]

class BaseStreamEvent(BaseModel):
    """Base model for all streaming events."""
    event_type: StreamEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StreamStartEvent(BaseStreamEvent):
    """Signals the beginning of a stream of events."""
    event_type: Literal["stream_start"] = "stream_start"
    metadata: Optional[Dict[str, Any]] = None # E.g., request_id, model_name

class StreamEndEvent(BaseStreamEvent):
    """Signals the end of a stream of events."""
    event_type: Literal["stream_end"] = "stream_end"
    final_usage: Optional[Dict[str, Any]] = None # E.g., token counts

class TextDeltaEvent(BaseStreamEvent):
    """Event for a text delta (chunk of text) in the stream."""
    event_type: Literal["text_delta"] = "text_delta"
    delta: str
    part_index: Optional[int] = None # If the response has multiple parts

class ToolCallStartEvent(BaseStreamEvent):
    """Event when the model decides to call a tool."""
    event_type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str
    tool_name: str
    part_index: Optional[int] = None

class ToolCallChunkEvent(BaseStreamEvent):
    """Event for a chunk of arguments for a tool call (if streamed by provider)."""
    event_type: Literal["tool_call_chunk"] = "tool_call_chunk"
    tool_call_id: str
    args_delta: str # JSON string delta for arguments
    part_index: Optional[int] = None

class ToolCallEndEvent(BaseStreamEvent):
    """Event marking the end of the model's tool call definition."""
    event_type: Literal["tool_call_end"] = "tool_call_end"
    tool_call_id: str
    tool_name: str # Repeated for context at the end of the tool call block
    args: Dict[str, Any] # Complete arguments for the tool
    part_index: Optional[int] = None

class ToolResultEvent(BaseStreamEvent):
    """Event for the result obtained after executing a tool."""
    event_type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool = False

class FinalOutputEvent(BaseStreamEvent):
    """Event for the final, complete output from the assistant (not a tool call)."""
    event_type: Literal["final_output"] = "final_output"
    output: Any # Could be a string or a structured Pydantic model if Agent has output_type

class StreamErrorEvent(BaseStreamEvent):
    """Event to signal an error encountered during streaming."""
    event_type: Literal["stream_error"] = "stream_error"
    error_message: str
    error_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# A type alias for any of the defined stream events
AnyStreamEvent = Union[
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolResultEvent,
    FinalOutputEvent,
    StreamErrorEvent,
    StreamEndEvent,
] 