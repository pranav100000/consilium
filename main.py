import asyncio
import json
import sys
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from consilium import ConsiliumRequest, run_consilium

# Load environment variables from .env.local
env_path = Path('.env.local')
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    load_dotenv()  # Try to load from default .env file


async def run_from_json(json_file: str):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        request = ConsiliumRequest(**data)
        print(f"Running Consilium with {len(request.models)} models...")
        print(f"Problem: {request.problem}")
        print(f"Max iterations: {request.max_iterations}")
        
        result = await run_consilium(request)
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Consensus reached: {result.consensus_reached}")
        print(f"Iterations used: {result.iterations_used}")
        print(f"Final solution:\n{result.final_solution}")
        
        return result
        
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


async def interactive_mode():
    print("Consilium Interactive Mode")
    print("=" * 60)
    
    models = []
    contexts = []
    
    print("\nEnter models (one per line, empty line to finish):")
    while True:
        model = input(f"Model {len(models) + 1}: ").strip()
        if not model:
            break
        models.append(model)
        
        context = input(f"Context for {model}: ").strip()
        contexts.append(context)
    
    if not models:
        print("Error: At least one model is required")
        return
    
    problem = input("\nEnter the problem to solve: ").strip()
    if not problem:
        print("Error: Problem is required")
        return
    
    max_iter = input("Max iterations (default 2): ").strip()
    max_iterations = int(max_iter) if max_iter else 2
    
    request = ConsiliumRequest(
        models=models,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=max_iterations
    )
    
    print("\n" + "=" * 60)
    print("Running Consilium...")
    print("=" * 60)
    
    result = await run_consilium(request)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Iterations used: {result.iterations_used}")
    print(f"Final solution:\n{result.final_solution}")
    
    save = input("\nSave result to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Filename (default: result.json): ").strip() or "result.json"
        with open(filename, 'w') as f:
            json.dump(result.model_dump(), f, indent=2)
        print(f"Result saved to {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Consilium - Multi-model consensus system")
    parser.add_argument('-f', '--file', help='JSON file with request configuration')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.file:
        asyncio.run(run_from_json(args.file))
    elif args.interactive:
        asyncio.run(interactive_mode())
    else:
        print("Consilium - Multi-model consensus system")
        print("\nUsage:")
        print("  python main.py -f request.json    # Run from JSON file")
        print("  python main.py -i                 # Interactive mode")
        print("\nExample JSON format:")
        print(json.dumps({
            "models": ["openai:gpt-4o-mini", "anthropic:claude-3-haiku"],
            "initial_contexts": ["You are an expert in X", "You are an expert in Y"],
            "problem": "Solve this problem",
            "max_iterations": 2
        }, indent=2))


if __name__ == "__main__":
    main()
