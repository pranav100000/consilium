import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from consilium import ConsiliumRequest, run_consilium

# Load environment variables from .env.local
env_path = Path(__file__).parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")


async def main():
    print("=" * 60)
    print("Consilium Demo - Multi-Model Consensus System")
    print("=" * 60)
    
    request = ConsiliumRequest(
        models=[
            "openai:gpt-4o-mini",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-haiku-20240307"
        ],
        initial_contexts=[
            "You are an expert in creative writing and storytelling.",
            "You are an expert in technical accuracy and logical consistency.",
            "You are an expert in audience engagement and clarity."
        ],
        problem="Write a one-paragraph explanation of how photosynthesis works for a 10-year-old child.",
        max_iterations=2
    )
    
    print(f"\nProblem: {request.problem}")
    print(f"Models: {request.models}")
    print(f"Max iterations: {request.max_iterations}")
    print("\nRunning consensus process...")
    print("-" * 60)
    
    try:
        result = await run_consilium(request)
        
        print(f"\n{'=' * 60}")
        print("RESULT")
        print(f"{'=' * 60}")
        print(f"Consensus reached: {result.consensus_reached}")
        print(f"Iterations used: {result.iterations_used}")
        print(f"Winning model index: {result.winning_model_index}")
        print(f"\nFinal solution:")
        print("-" * 60)
        print(result.final_solution)
        print("-" * 60)
        
        if result.all_solutions:
            print(f"\nTotal solutions generated: {len(result.all_solutions)}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set the required API keys:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")


async def simple_demo():
    print("\n" + "=" * 60)
    print("Simple Demo - Single Model")
    print("=" * 60)
    
    request = ConsiliumRequest(
        models=["openai:gpt-4o-mini"],
        initial_contexts=["You are a helpful assistant focused on clear, concise answers."],
        problem="What are the three primary colors?",
        max_iterations=1
    )
    
    print(f"\nProblem: {request.problem}")
    print(f"Model: {request.models[0]}")
    print("\nRunning...")
    
    try:
        result = await run_consilium(request)
        print(f"\nAnswer: {result.final_solution}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Starting Consilium Demo...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set (optional)")
    
    asyncio.run(simple_demo())
    
    print("\n" + "=" * 60)
    print("Press Enter to run the multi-model consensus demo...")
    input()
    
    asyncio.run(main())