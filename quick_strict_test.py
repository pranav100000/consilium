import asyncio
from pathlib import Path
from dotenv import load_dotenv
from consilium import ConsiliumRequest, run_consilium

load_dotenv('.env.local')

async def test():
    problem = "Write a function to check if a number is prime"
    contexts = ["Focus on efficiency", "Focus on correctness"]
    
    print("Testing Regular Mode (easy consensus):")
    req1 = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * 2,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3,
        strict_mode=False
    )
    result1 = await run_consilium(req1)
    print(f"  Iterations: {result1.iterations_used}")
    print(f"  Consensus: {result1.consensus_level:.0%}")
    print(f"  Termination: {result1.termination_reason}")
    
    print("\nTesting Strict Mode (harder consensus):")
    req2 = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * 2,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3,
        strict_mode=True,
        force_iterations=2
    )
    result2 = await run_consilium(req2)
    print(f"  Iterations: {result2.iterations_used}")
    print(f"  Consensus: {result2.consensus_level:.0%}")
    print(f"  Termination: {result2.termination_reason}")
    
    print(f"\nDifference: Strict mode used {result2.iterations_used - result1.iterations_used} more iterations")

asyncio.run(test())