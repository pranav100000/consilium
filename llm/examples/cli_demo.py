import asyncio
import os
import sys

# Adjust the import path if running from the root of the project vs. inside examples folder
# This assumes you run `python pydantic_ai_wrapper/examples/cli_demo.py` from the project root,
# or that pydantic_ai_wrapper is in your PYTHONPATH.

# For simple execution from project root: python -m pydantic_ai_wrapper.examples.cli_demo
# or add project root to PYTHONPATH.
# If pydantic_ai_wrapper is installed as a package (e.g. pip install .), then direct imports work.

# Let's try to make it runnable from project root: `python pydantic_ai_wrapper/examples/cli_demo.py`

# Go two levels up from this file (cli_demo.py) to reach the project root (pydantic-ai-test3)
# where the pydantic_ai_wrapper package resides.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.chat_wrapper import ChatSession
from llm.streaming import (
    AnyStreamEvent, StreamStartEvent, TextDeltaEvent, 
    FinalOutputEvent, StreamEndEvent, ToolCallStartEvent,
    ToolCallChunkEvent, ToolCallEndEvent, ToolResultEvent,
    StreamErrorEvent
)

async def main():
    print("Initializing ChatSession with OpenAI...")
    # Make sure OPENAI_API_KEY is in your .env file at the project root
    try:
        chat_session = ChatSession(
            provider_name="openai",
            model_name="gpt-3.5-turbo", # Cheaper and faster for demo
            system_prompt="You are a concise and helpful assistant."
        )
    except Exception as e:
        print(f"Error initializing OpenAI session: {e}")
        print("Please ensure OPENAI_API_KEY is set in your .env file at the project root.")
        return

    print("--- 1. OpenAI: Non-streaming message ---")
    prompt1 = "What is the capital of France?"
    print(f"> User: {prompt1}")
    try:
        response1 = await chat_session.send_message_async(prompt1, stream=False)
        print(f"< Assistant: {response1}")
    except Exception as e:
        print(f"Error sending non-streaming message to OpenAI: {e}")

    print("\n--- 2. OpenAI: Streaming message ---")
    prompt2 = "Tell me a very short story about a brave robot."
    print(f"> User: {prompt2}")
    print("< Assistant (streaming):")
    full_streamed_response = ""
    try:
        async for event in await chat_session.send_message_async(prompt2, stream=True):
            if isinstance(event, StreamStartEvent):
                print(f"  [STREAM_START: Model={event.metadata.get('model') if event.metadata else 'N/A'}]")
            elif isinstance(event, TextDeltaEvent):
                print(event.delta, end="", flush=True)
                full_streamed_response += event.delta
            elif isinstance(event, FinalOutputEvent):
                # This might be redundant if we are already printing TextDeltaEvents
                # and ChatSession already assembles the full text for history.
                # Adapters yield this with the full accumulated text from LLM.
                print(f"\n  [FINAL_OUTPUT: {event.output[:50]}...]" if event.output else "")
            elif isinstance(event, StreamEndEvent):
                print(f"\n  [STREAM_END: Usage={event.final_usage}]")
            elif isinstance(event, ToolCallStartEvent):
                print(f"\n  [TOOL_CALL_START: ID={event.tool_call_id}, Name={event.tool_name}]")
            elif isinstance(event, ToolCallChunkEvent):
                print(f"  [TOOL_CALL_CHUNK: ID={event.tool_call_id}, ArgsDelta={event.args_delta}]")
            elif isinstance(event, ToolCallEndEvent):
                print(f"  [TOOL_CALL_END: ID={event.tool_call_id}, Name={event.tool_name}, Args={event.args}]")
            elif isinstance(event, ToolResultEvent):
                 print(f"  [TOOL_RESULT: ID={event.tool_call_id}, Name={event.tool_name}, Result={event.result}]")   
            elif isinstance(event, StreamErrorEvent):
                print(f"\n  [STREAM_ERROR: Type={event.error_type}, Message={event.error_message}]")
            else:
                print(f"\n  [UNKNOWN_EVENT: {type(event)}]")
        print("\n------------------------")
    except Exception as e:
        print(f"\nError during OpenAI streaming: {e}")

    # Switch to Anthropic
    print("\n--- 3. Switching to Anthropic --- ")
    # Make sure ANTHROPIC_API_KEY is in your .env file
    try:
        chat_session.switch_provider(
            new_provider_name="anthropic",
            new_model_name="claude-3-haiku-20240307" # Fastest and cheapest Claude 3 model
        )
        print("Switched to Anthropic successfully.")
    except Exception as e:
        print(f"Error switching to Anthropic: {e}")
        print("Please ensure ANTHROPIC_API_KEY is set in your .env file.")
        return
    
    print("\n--- 4. Anthropic: Streaming message ---")
    prompt3 = "Why is the sky blue? Explain concisely."
    print(f"> User: {prompt3}")
    print("< Assistant (streaming):")
    try:
        async for event in await chat_session.send_message_async(prompt3, stream=True):
            if isinstance(event, TextDeltaEvent):
                print(event.delta, end="", flush=True)
            elif isinstance(event, StreamStartEvent):
                print(f"  [STREAM_START: Model={event.metadata.get('model') if event.metadata else 'N/A'}]")
            elif isinstance(event, StreamEndEvent):
                print(f"\n  [STREAM_END: Usage={event.final_usage}]")
            # Add other event type handling if needed for Anthropic specific details
        print("\n------------------------")
    except Exception as e:
        print(f"\nError during Anthropic streaming: {e}")

    # Switch to Gemini (Generative Language API)
    print("\n--- 5. Switching to Gemini (google-gla) --- ")
    # Make sure GOOGLE-GLA_API_KEY is set in your .env file (or GEMINI_API_KEY as per Pydantic AI docs for google-gla)
    # Our config.py looks for <PROVIDER_NAME>_API_KEY, so GOOGLE-GLA_API_KEY
    try:
        chat_session.switch_provider(
            new_provider_name="google-gla", 
            new_model_name="gemini-1.5-flash-latest" # Or "gemini-pro"
            # api_key="YOUR_GEMINI_API_KEY_HERE" # Optionally pass directly
        )
        print("Switched to Gemini (google-gla) successfully.")
    except Exception as e:
        print(f"Error switching to Gemini (google-gla): {e}")
        print("Please ensure your Gemini API key (e.g., GOOGLE-GLA_API_KEY) is set for the google-gla provider.")
        # return # Optionally stop if switch fails

    # Only proceed if switch was successful or not skipped
    if chat_session.current_provider_name == "google-gla":
        print("\n--- 6. Gemini (google-gla): Streaming message ---")
        prompt4 = "What are the main benefits of using Pydantic for data validation?"
        print(f"> User: {prompt4}")
        print("< Assistant (streaming):")
        try:
            async for event in await chat_session.send_message_async(prompt4, stream=True):
                if isinstance(event, StreamStartEvent):
                    print(f"  [STREAM_START: Provider={event.metadata.get('provider')}, Model={event.metadata.get('model')}]")
                elif isinstance(event, TextDeltaEvent):
                    print(event.delta, end="", flush=True)
                elif isinstance(event, FinalOutputEvent):
                    print(f"\n  [FINAL_OUTPUT: {event.output[:70]}...]" if event.output else "")
                elif isinstance(event, StreamEndEvent):
                    print(f"\n  [STREAM_END: Usage={event.final_usage}]")
                elif isinstance(event, ToolCallStartEvent):
                    print(f"\n  [TOOL_CALL_START: ID={event.tool_call_id}, Name={event.tool_name}]")
                elif isinstance(event, ToolCallEndEvent):
                    print(f"  [TOOL_CALL_END: ID={event.tool_call_id}, Name={event.tool_name}, Args={event.args}]")
                elif isinstance(event, StreamErrorEvent):
                    print(f"\n  [STREAM_ERROR: Type={event.error_type}, Message={event.error_message}]")
                # Add other event type handling if needed for Gemini specific details
            print("\n------------------------")
        except Exception as e:
            print(f"\nError during Gemini (google-gla) streaming: {e}")

    print("\n--- Demo Complete --- ")
    print("Final Chat History:")
    for i, msg in enumerate(chat_session.message_history.get_messages()):
        print(f"  {i+1}. Role: {msg.role}, Content: {str(msg.content)[:200]}...")

if __name__ == "__main__":
    asyncio.run(main()) 