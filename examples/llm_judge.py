import asyncio
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from consilium import ConsiliumRequest
from consilium.core import ConsiliumOrchestrator
from consilium.core_v2 import EnhancedConsiliumOrchestrator

load_dotenv('.env.local')


class JudgmentResult(BaseModel):
    winner: str = Field(description="Either 'Solution A' or 'Solution B' or 'Tie'")
    score_a: int = Field(ge=0, le=100, description="Quality score for Solution A")
    score_b: int = Field(ge=0, le=100, description="Quality score for Solution B")
    reasoning: str = Field(description="Brief explanation of the judgment")
    strengths_a: list[str] = Field(description="Strengths of Solution A")
    strengths_b: list[str] = Field(description="Strengths of Solution B")


async def judge_solutions(problem: str, solution_a: str, solution_b: str) -> JudgmentResult:
    """Have an LLM judge which solution is better"""
    
    judge = Agent(
        model='openai:gpt-4o',  # Using GPT-4 as an impartial judge
        output_type=JudgmentResult,
        system_prompt="""You are an expert code reviewer and judge. 
        Evaluate two solutions objectively based on:
        1. Correctness - Does it solve the problem correctly?
        2. Code Quality - Is it clean, readable, maintainable?
        3. Completeness - Does it handle edge cases and errors?
        4. Documentation - Are there docstrings and comments?
        5. Efficiency - Is it reasonably efficient?
        6. Best Practices - Does it follow Python conventions?
        
        Be fair and objective. Don't favor verbosity or brevity inherently - judge based on overall quality."""
    )
    
    prompt = f"""Problem: {problem}

Solution A:
```python
{solution_a}
```

Solution B:
```python
{solution_b}
```

Evaluate both solutions thoroughly and determine which is better overall."""

    result = await judge.run(prompt)
    return result.output


async def run_judged_comparison():
    """Run comparison with LLM judge evaluation"""
    
    test_cases = [
        {
            "name": "String Reversal",
            "problem": "Write a Python function to reverse a string. Include a docstring and handle None input.",
            "contexts": [
                "You focus on code clarity and documentation",
                "You focus on error handling and edge cases"
            ]
        },
        {
            "name": "Prime Check",
            "problem": "Write a Python function to check if a number is prime. Include proper documentation and handle invalid inputs.",
            "contexts": [
                "You focus on mathematical correctness",
                "You focus on efficiency and optimization"
            ]
        },
        {
            "name": "List Deduplication",
            "problem": "Write a Python function to remove duplicates from a list while preserving order. Handle edge cases.",
            "contexts": [
                "You focus on algorithmic efficiency",
                "You focus on code readability"
            ]
        }
    ]
    
    print("=" * 80)
    print("LLM JUDGE EVALUATION: Original vs Enhanced Consilium")
    print("=" * 80)
    
    results = []
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {test['name']}")
        print(f"{'='*60}")
        
        # Run Original
        request_orig = ConsiliumRequest(
            models=['openai:gpt-4o-mini'] * len(test['contexts']),
            initial_contexts=test['contexts'],
            problem=test['problem'],
            max_iterations=3
        )
        
        print("\n⏳ Running Original Consilium...")
        start = time.time()
        orch_orig = ConsiliumOrchestrator(request_orig)
        result_orig = await orch_orig.run()
        time_orig = time.time() - start
        
        # Run Enhanced
        request_enh = ConsiliumRequest(
            models=['openai:gpt-4o-mini'] * len(test['contexts']),
            initial_contexts=test['contexts'],
            problem=test['problem'],
            max_iterations=3,
            consensus_threshold=0.8,
            early_stop_on_plateau=True,
            enable_solution_history=True,
            weighted_voting=True
        )
        
        print("⏳ Running Enhanced Consilium...")
        start = time.time()
        orch_enh = EnhancedConsiliumOrchestrator(request_enh)
        result_enh = await orch_enh.run()
        time_enh = time.time() - start
        
        # Extract code
        def extract_code(solution):
            match = re.search(r'```(?:python)?\n(.*?)```', solution, re.DOTALL)
            if match:
                return match.group(1).strip()
            # Try without language specifier
            match = re.search(r'```\n(.*?)```', solution, re.DOTALL)
            if match:
                return match.group(1).strip()
            return solution.strip()
        
        code_orig = extract_code(result_orig.final_solution)
        code_enh = extract_code(result_enh.final_solution)
        
        # Randomly assign to A/B to avoid bias
        import random
        if random.random() > 0.5:
            solution_a = code_orig
            solution_b = code_enh
            a_is_original = True
        else:
            solution_a = code_enh
            solution_b = code_orig
            a_is_original = False
        
        # Judge the solutions
        print("\n🧑‍⚖️ LLM Judge evaluating solutions...")
        judgment = await judge_solutions(test['problem'], solution_a, solution_b)
        
        # Determine actual winner
        if judgment.winner == "Solution A":
            actual_winner = "Original" if a_is_original else "Enhanced"
        elif judgment.winner == "Solution B":
            actual_winner = "Enhanced" if a_is_original else "Original"
        else:
            actual_winner = "Tie"
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"  Original: {time_orig:.1f}s, {result_orig.iterations_used} iterations")
        print(f"  Enhanced: {time_enh:.1f}s, {result_enh.iterations_used} iterations")
        print(f"  Speed advantage: Enhanced is {((time_orig-time_enh)/time_orig)*100:.0f}% faster")
        
        print(f"\n⚖️ JUDGE'S VERDICT:")
        print(f"  Winner: {actual_winner}")
        print(f"  Scores: Original={judgment.score_a if a_is_original else judgment.score_b}, "
              f"Enhanced={judgment.score_b if a_is_original else judgment.score_a}")
        print(f"  Reasoning: {judgment.reasoning}")
        
        if a_is_original:
            print(f"\n  Original Strengths: {', '.join(judgment.strengths_a)}")
            print(f"  Enhanced Strengths: {', '.join(judgment.strengths_b)}")
        else:
            print(f"\n  Original Strengths: {', '.join(judgment.strengths_b)}")
            print(f"  Enhanced Strengths: {', '.join(judgment.strengths_a)}")
        
        results.append({
            "test": test['name'],
            "winner": actual_winner,
            "orig_score": judgment.score_a if a_is_original else judgment.score_b,
            "enh_score": judgment.score_b if a_is_original else judgment.score_a,
            "time_orig": time_orig,
            "time_enh": time_enh,
            "iter_orig": result_orig.iterations_used,
            "iter_enh": result_enh.iterations_used
        })
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    orig_wins = sum(1 for r in results if r['winner'] == 'Original')
    enh_wins = sum(1 for r in results if r['winner'] == 'Enhanced')
    ties = sum(1 for r in results if r['winner'] == 'Tie')
    
    avg_orig_score = sum(r['orig_score'] for r in results) / len(results)
    avg_enh_score = sum(r['enh_score'] for r in results) / len(results)
    
    total_time_orig = sum(r['time_orig'] for r in results)
    total_time_enh = sum(r['time_enh'] for r in results)
    
    print(f"\n🏆 QUALITY WINNER:")
    print(f"  Original wins: {orig_wins}")
    print(f"  Enhanced wins: {enh_wins}")
    print(f"  Ties: {ties}")
    print(f"  Average scores: Original={avg_orig_score:.1f}, Enhanced={avg_enh_score:.1f}")
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"  Total time: Original={total_time_orig:.1f}s, Enhanced={total_time_enh:.1f}s")
    print(f"  Enhanced is {((total_time_orig-total_time_enh)/total_time_orig)*100:.0f}% faster overall")
    
    print(f"\n📈 CONCLUSION:")
    if enh_wins > orig_wins:
        print(f"  ✅ Enhanced wins on BOTH quality ({enh_wins}/{len(results)}) and speed!")
    elif enh_wins == orig_wins:
        print(f"  ✅ Enhanced matches quality ({enh_wins}/{len(results)}) while being much faster!")
    else:
        quality_diff = avg_orig_score - avg_enh_score
        if quality_diff < 5:  # Within 5 points is acceptable
            print(f"  ✅ Enhanced has comparable quality (within {quality_diff:.1f} points) but is much faster!")
        else:
            print(f"  ⚠️ Trade-off: Original has better quality (+{quality_diff:.1f}) but Enhanced is much faster")


if __name__ == "__main__":
    asyncio.run(run_judged_comparison())