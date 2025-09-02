import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from consilium import ConsiliumRequest, run_consilium

# Load environment variables
env_path = Path(__file__).parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(env_path)


async def test_graduated_consensus():
    """Test graduated consensus with 80% threshold"""
    print("=" * 60)
    print("TEST: Graduated Consensus (80% threshold)")
    print("=" * 60)
    
    request = ConsiliumRequest(
        models=["openai:gpt-4o-mini", "openai:gpt-4o-mini", "openai:gpt-4o-mini"],
        initial_contexts=[
            "You are focused on simplicity and clarity",
            "You are focused on performance",
            "You are focused on correctness"
        ],
        problem="Write a one-line Python function to check if a number is even",
        max_iterations=3,
        consensus_threshold=0.8,  # 80% approval is enough
        early_stop_on_plateau=True
    )
    
    result = await run_consilium(request)
    
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Consensus level: {result.consensus_level:.2%}")
    print(f"Iterations used: {result.iterations_used}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Final solution:\n{result.final_solution}")
    print()


async def test_early_stopping():
    """Test early stopping when only minor issues remain"""
    print("=" * 60)
    print("TEST: Early Stopping on Minor Issues")
    print("=" * 60)
    
    request = ConsiliumRequest(
        models=["openai:gpt-4o-mini", "openai:gpt-4o-mini"],
        initial_contexts=[
            "You are a perfectionist about code style",
            "You are practical and focus on functionality"
        ],
        problem="Write a function that returns the sum of two numbers",
        max_iterations=5,
        early_stop_on_plateau=True,
        min_improvement_threshold=0.1
    )
    
    result = await run_consilium(request)
    
    print(f"Iterations used: {result.iterations_used} (max was {request.max_iterations})")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Improvement trajectory: {result.improvement_trajectory}")
    print(f"Final solution:\n{result.final_solution}")
    print()


async def test_solution_history():
    """Test with solution history enabled"""
    print("=" * 60)
    print("TEST: Solution History Context")
    print("=" * 60)
    
    request = ConsiliumRequest(
        models=["openai:gpt-4o-mini", "openai:gpt-4o-mini"],
        initial_contexts=[
            "You focus on algorithmic efficiency",
            "You focus on code readability"
        ],
        problem="Write a function to find the maximum element in a list",
        max_iterations=2,
        enable_solution_history=True,  # Models can see their previous attempts
        enable_critique_history=True
    )
    
    result = await run_consilium(request)
    
    print(f"Consensus level: {result.consensus_level:.2%}")
    
    # Show how solutions evolved
    for solution in result.all_solutions:
        if solution.iteration > 0 and solution.improvement_delta:
            print(f"Model {solution.model_index} improvement: {solution.improvement_delta:.2%}")
    
    print(f"Final solution:\n{result.final_solution}")
    print()


async def test_weighted_voting():
    """Test weighted voting based on performance"""
    print("=" * 60)
    print("TEST: Weighted Voting")
    print("=" * 60)
    
    request = ConsiliumRequest(
        models=["openai:gpt-4o-mini", "openai:gpt-4o-mini", "openai:gpt-4o-mini"],
        initial_contexts=[
            "You are an expert in algorithms",
            "You are an expert in testing",
            "You are an expert in documentation"
        ],
        problem="Write a binary search function with proper error handling",
        max_iterations=2,
        weighted_voting=True,  # Votes weighted by performance
        consensus_threshold=0.9  # High threshold to force voting
    )
    
    result = await run_consilium(request)
    
    print(f"Termination: {result.termination_reason}")
    
    # Show model performance
    for perf in result.model_performance:
        print(f"Model {perf.model_index}:")
        print(f"  - Solutions approved: {perf.solutions_approved}")
        print(f"  - Vote weight: {perf.vote_weight:.2f}")
    
    print(f"Winner: Model {result.winning_model_index}")
    print(f"Final solution:\n{result.final_solution}")
    print()


async def test_complex_problem():
    """Test with the sample request file"""
    print("=" * 60)
    print("TEST: Complex Problem with All Features")
    print("=" * 60)
    
    request = ConsiliumRequest(
        models=["openai:gpt-4o-mini", "openai:gpt-4o-mini", "openai:gpt-4o-mini"],
        initial_contexts=[
            "You are an expert in algorithm design and computational efficiency.",
            "You are an expert in code readability and best practices.",
            "You are an expert in error handling and edge cases."
        ],
        problem="Write a Python function that finds the k-th largest element in an unsorted array. The function should be efficient and handle edge cases properly.",
        max_iterations=3,
        consensus_threshold=0.75,
        early_stop_on_plateau=True,
        enable_solution_history=True,
        weighted_voting=True
    )
    
    result = await run_consilium(request)
    
    print(f"Result Summary:")
    print(f"  - Consensus: {result.consensus_reached} ({result.consensus_level:.2%})")
    print(f"  - Iterations: {result.iterations_used}/{request.max_iterations}")
    print(f"  - Termination: {result.termination_reason}")
    print(f"  - Improvement over iterations: {[f'{x:.2%}' for x in result.improvement_trajectory]}")
    
    print(f"\nFinal solution:\n{result.final_solution}")


async def main():
    print("\n" + "=" * 60)
    print("ENHANCED CONSILIUM FEATURE TESTS")
    print("=" * 60 + "\n")
    
    # Run tests
    await test_graduated_consensus()
    await test_early_stopping()
    await test_solution_history()
    await test_weighted_voting()
    await test_complex_problem()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())