import asyncio
import sys
sys.path.insert(0, '/Users/pranavsharan/Developer/consilium')

from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field

load_dotenv('.env.local')


class QuickJudgment(BaseModel):
    winner: str = Field(description="'A' or 'B' or 'Tie'")
    score_a: int = Field(ge=0, le=100)
    score_b: int = Field(ge=0, le=100)
    reasoning: str


async def judge():
    # These are actual outputs from our test
    solution_a = """def reverse_string(input_string: str | None) -> str | None:
    \"\"\"
    Reverses the given string.

    Parameters:
    input_string (str or None): The string to be reversed. If the input is None, 
                                 the function will return None. If the input is 
                                 not a string or None, a ValueError will be raised.

    Returns:
    str: The reversed string, or None if the input is None.

    Raises:
    ValueError: If input_string is not a string or None.

    Examples:
    >>> reverse_string("hello")
    'olleh'
    >>> reverse_string("Python")
    'nohtyP'
    >>> reverse_string(None)
    None
    \"\"\"
    
    if input_string is None:
        return None
    
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string or None")
    
    return input_string[::-1]"""
    
    solution_b = """def reverse_string(input_string):
    \"\"\"
    Reverse the given string.

    Args:
        input_string (str or None): The string to be reversed. 
            If the input is None, the function will return None.

    Returns:
        str or None: The reversed string if input_string is a valid string,
            otherwise None.
    
    Example:
        >>> reverse_string("hello")
        'olleh'
        >>> reverse_string(None)
        None
    \"\"\"
    if input_string is None:
        return None
    return input_string[::-1]"""
    
    judge = Agent(
        model='openai:gpt-4o',
        output_type=QuickJudgment,
        system_prompt="You are an expert Python code reviewer. Judge solutions based on correctness, clarity, completeness, and best practices."
    )
    
    result = await judge.run(
        f"Problem: Write a Python function to reverse a string. Include a docstring and handle None input.\n\n"
        f"Solution A:\n{solution_a}\n\n"
        f"Solution B:\n{solution_b}\n\n"
        f"Which is better? Be objective."
    )
    
    print("=" * 60)
    print("LLM JUDGE EVALUATION")
    print("=" * 60)
    print(f"\nWinner: Solution {result.output.winner}")
    print(f"Score A (Original - 57 lines): {result.output.score_a}")
    print(f"Score B (Enhanced - 29 lines): {result.output.score_b}")
    print(f"\nReasoning: {result.output.reasoning}")
    
    print("\n" + "=" * 60)
    print("CONTEXT:")
    print("- Solution A is from ORIGINAL Consilium (3 iterations, 43.8s)")
    print("- Solution B is from ENHANCED Consilium (1 iteration, 12.6s)")
    print("=" * 60)
    
    if result.output.winner == 'A':
        print("\n⚖️ Original had better quality despite being slower")
    elif result.output.winner == 'B':
        print("\n✅ Enhanced is better in BOTH quality AND speed!")
    else:
        print("\n✅ Same quality, but Enhanced is 71% faster!")

asyncio.run(judge())