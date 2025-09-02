import asyncio
import time
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, '/Users/pranavsharan/Developer/consilium')

from consilium import ConsiliumRequest
from consilium.core import ConsiliumOrchestrator
from consilium.core_v2 import EnhancedConsiliumOrchestrator
from consilium.core_v3 import StrictConsiliumOrchestrator

load_dotenv('.env.local')


class SolutionJudgment(BaseModel):
    """LLM Judge evaluation result"""
    scores: Dict[str, int] = Field(description="Quality scores for each solution (0-100)")
    winner: str = Field(description="Which solution is best: 'original', 'enhanced', 'strict', or 'tie'")
    reasoning: str = Field(description="Explanation of the judgment")
    strengths: Dict[str, List[str]] = Field(description="Key strengths of each solution")
    weaknesses: Dict[str, List[str]] = Field(description="Key weaknesses of each solution")


async def run_system(system_type: str, request: ConsiliumRequest) -> Tuple[str, Dict]:
    """Run a specific system and return solution + metrics"""
    start_time = time.time()
    
    if system_type == "original":
        orchestrator = ConsiliumOrchestrator(request)
    elif system_type == "enhanced":
        orchestrator = EnhancedConsiliumOrchestrator(request)
    else:  # strict
        orchestrator = StrictConsiliumOrchestrator(request)
    
    result = await orchestrator.run()
    
    elapsed_time = time.time() - start_time
    
    # Extract code from solution
    def extract_code(solution):
        match = re.search(r'```(?:python)?\n(.*?)```', solution, re.DOTALL)
        if match:
            return match.group(1).strip()
        return solution.strip()
    
    metrics = {
        "time": elapsed_time,
        "iterations": result.iterations_used,
        "consensus": result.consensus_reached,
        "consensus_level": getattr(result, 'consensus_level', 0),
        "termination": getattr(result, 'termination_reason', 'unknown'),
        "solution_length": len(extract_code(result.final_solution))
    }
    
    return extract_code(result.final_solution), metrics


async def judge_solutions(problem: str, solutions: Dict[str, str]) -> SolutionJudgment:
    """Have an LLM judge evaluate all three solutions"""
    
    judge = Agent(
        model='openai:gpt-4o',
        output_type=SolutionJudgment,
        system_prompt="""You are an expert code reviewer judging solution quality.
        
Evaluate based on:
1. Correctness - Does it solve the problem correctly?
2. Completeness - Are edge cases and errors handled?
3. Code Quality - Is it clean, efficient, readable?
4. Documentation - Are there docstrings and comments where needed?
5. Best Practices - Does it follow Python conventions?
6. Robustness - Will it work reliably in production?

Score each solution 0-100. Be objective and thorough."""
    )
    
    prompt = f"""Problem: {problem}

Original System Solution:
```python
{solutions['original']}
```

Enhanced System Solution:
```python
{solutions['enhanced']}
```

Strict System Solution:
```python
{solutions['strict']}
```

Evaluate all three solutions thoroughly. Which produces the best code overall?"""
    
    result = await judge.run(prompt)
    return result.output


async def benchmark_problem(problem_name: str, problem: str, contexts: List[str]) -> Dict:
    """Benchmark all three systems on a single problem"""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {problem_name}")
    print(f"{'='*80}")
    print(f"Problem: {problem}\n")
    
    # Setup requests for each system
    base_config = {
        "models": ["openai:gpt-4o-mini"] * len(contexts),
        "initial_contexts": contexts,
        "problem": problem,
        "max_iterations": 4
    }
    
    # Original system
    request_original = ConsiliumRequest(**base_config)
    
    # Enhanced system
    request_enhanced = ConsiliumRequest(
        **base_config,
        consensus_threshold=0.8,
        early_stop_on_plateau=True,
        enable_solution_history=True,
        weighted_voting=True
    )
    
    # Strict system
    request_strict = ConsiliumRequest(
        **base_config,
        consensus_threshold=0.85,
        early_stop_on_plateau=True,
        enable_solution_history=True,
        weighted_voting=True,
        strict_mode=True,
        force_iterations=2,
        diversity_threshold=0.35
    )
    
    # Run all systems
    print("Running systems...")
    print("  🔵 Original system...", end="", flush=True)
    solution_orig, metrics_orig = await run_system("original", request_original)
    print(f" Done ({metrics_orig['time']:.1f}s, {metrics_orig['iterations']} iter)")
    
    print("  🟢 Enhanced system...", end="", flush=True)
    solution_enh, metrics_enh = await run_system("enhanced", request_enhanced)
    print(f" Done ({metrics_enh['time']:.1f}s, {metrics_enh['iterations']} iter)")
    
    print("  🔴 Strict system...", end="", flush=True)
    solution_strict, metrics_strict = await run_system("strict", request_strict)
    print(f" Done ({metrics_strict['time']:.1f}s, {metrics_strict['iterations']} iter)")
    
    # Judge solutions
    print("\n🧑‍⚖️ LLM Judge evaluating solutions...")
    solutions = {
        "original": solution_orig,
        "enhanced": solution_enh,
        "strict": solution_strict
    }
    
    judgment = await judge_solutions(problem, solutions)
    
    # Display results
    print("\n📊 PERFORMANCE METRICS:")
    print(f"  Original: {metrics_orig['time']:.1f}s, {metrics_orig['iterations']} iterations")
    print(f"  Enhanced: {metrics_enh['time']:.1f}s, {metrics_enh['iterations']} iterations")
    print(f"  Strict:   {metrics_strict['time']:.1f}s, {metrics_strict['iterations']} iterations")
    
    print("\n🏆 QUALITY SCORES:")
    for system, score in judgment.scores.items():
        print(f"  {system.capitalize()}: {score}/100")
    
    print(f"\n⚖️ WINNER: {judgment.winner.upper()}")
    print(f"Reasoning: {judgment.reasoning}")
    
    return {
        "problem": problem_name,
        "metrics": {
            "original": metrics_orig,
            "enhanced": metrics_enh,
            "strict": metrics_strict
        },
        "judgment": judgment,
        "solutions": solutions
    }


async def main():
    """Run comprehensive benchmark"""
    
    print("="*80)
    print("COMPREHENSIVE BENCHMARK: Original vs Enhanced vs Strict")
    print("="*80)
    
    test_problems = [
        {
            "name": "Binary Search",
            "problem": "Write a Python function to perform binary search on a sorted array. Include proper error handling, docstring, and handle edge cases.",
            "contexts": [
                "You are an expert in algorithms and efficiency",
                "You are an expert in error handling and robustness",
                "You are an expert in code documentation"
            ]
        },
        {
            "name": "Email Validator",
            "problem": "Write a Python function to validate email addresses using regex. Should handle common formats and return detailed validation results.",
            "contexts": [
                "You are an expert in regex patterns",
                "You are an expert in input validation"
            ]
        },
        {
            "name": "LRU Cache",
            "problem": "Implement an LRU (Least Recently Used) cache in Python with get and put methods. Use efficient data structures.",
            "contexts": [
                "You are an expert in data structures",
                "You are an expert in performance optimization",
                "You are an expert in clean code principles"
            ]
        }
    ]
    
    results = []
    
    for test in test_problems:
        try:
            result = await benchmark_problem(test["name"], test["problem"], test["contexts"])
            results.append(result)
        except Exception as e:
            print(f"Error in {test['name']}: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Aggregate metrics
    total_time = {"original": 0, "enhanced": 0, "strict": 0}
    total_iterations = {"original": 0, "enhanced": 0, "strict": 0}
    quality_scores = {"original": [], "enhanced": [], "strict": []}
    wins = {"original": 0, "enhanced": 0, "strict": 0, "tie": 0}
    
    for result in results:
        for system in ["original", "enhanced", "strict"]:
            total_time[system] += result["metrics"][system]["time"]
            total_iterations[system] += result["metrics"][system]["iterations"]
            quality_scores[system].append(result["judgment"].scores[system])
        
        wins[result["judgment"].winner] += 1
    
    print("\n⏱️ TOTAL TIME:")
    for system in ["original", "enhanced", "strict"]:
        print(f"  {system.capitalize()}: {total_time[system]:.1f}s")
    
    print("\n🔄 TOTAL ITERATIONS:")
    for system in ["original", "enhanced", "strict"]:
        print(f"  {system.capitalize()}: {total_iterations[system]}")
    
    print("\n✨ AVERAGE QUALITY SCORES:")
    for system in ["original", "enhanced", "strict"]:
        avg_score = sum(quality_scores[system]) / len(quality_scores[system]) if quality_scores[system] else 0
        print(f"  {system.capitalize()}: {avg_score:.1f}/100")
    
    print("\n🏆 WINS BY SYSTEM:")
    for system, count in wins.items():
        if count > 0:
            print(f"  {system.capitalize()}: {count}/{len(results)}")
    
    # Efficiency vs Quality Analysis
    print("\n📈 EFFICIENCY vs QUALITY:")
    for system in ["original", "enhanced", "strict"]:
        avg_score = sum(quality_scores[system]) / len(quality_scores[system]) if quality_scores[system] else 0
        avg_time = total_time[system] / len(results) if results else 0
        efficiency = avg_score / avg_time if avg_time > 0 else 0
        print(f"  {system.capitalize()}: {efficiency:.2f} quality points per second")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    # Find best for different scenarios
    avg_scores = {sys: sum(scores)/len(scores) if scores else 0 for sys, scores in quality_scores.items()}
    avg_times = {sys: time/len(results) if results else 0 for sys, time in total_time.items()}
    
    best_quality = max(avg_scores.items(), key=lambda x: x[1])
    best_speed = min(avg_times.items(), key=lambda x: x[1])
    best_efficiency = max([(s, avg_scores[s]/avg_times[s] if avg_times[s] > 0 else 0) for s in ["original", "enhanced", "strict"]], key=lambda x: x[1])
    
    print(f"Best Quality: {best_quality[0].upper()} ({best_quality[1]:.1f}/100)")
    print(f"Best Speed: {best_speed[0].upper()} ({best_speed[1]:.1f}s avg)")
    print(f"Best Efficiency: {best_efficiency[0].upper()} ({best_efficiency[1]:.1f} points/sec)")
    
    print("\nUSE CASES:")
    print("• Original: When you need thorough exploration (research)")
    print("• Enhanced: Best for most production use (speed + quality)")
    print("• Strict: When quality is critical (safety-critical systems)")
    
    # Save detailed results
    with open('benchmark_results.json', 'w') as f:
        json.dump({
            "summary": {
                "total_time": total_time,
                "total_iterations": total_iterations,
                "average_scores": {s: sum(scores)/len(scores) if scores else 0 for s, scores in quality_scores.items()},
                "wins": wins
            },
            "problems": [
                {
                    "name": r["problem"],
                    "winner": r["judgment"].winner,
                    "scores": r["judgment"].scores,
                    "reasoning": r["judgment"].reasoning
                }
                for r in results
            ]
        }, indent=2)
    
    print("\n📄 Detailed results saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())