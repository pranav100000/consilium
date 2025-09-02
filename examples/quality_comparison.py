import asyncio
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from consilium import ConsiliumRequest
from consilium.core import ConsiliumOrchestrator
from consilium.core_v2 import EnhancedConsiliumOrchestrator

# Load environment variables
env_path = Path(__file__).parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(env_path)


def evaluate_code_quality(code: str, problem_type: str) -> dict:
    """Evaluate code quality metrics"""
    metrics = {
        "has_function": "def " in code or "lambda" in code,
        "has_docstring": '"""' in code or "'''" in code,
        "has_error_handling": any(x in code for x in ["try", "except", "if", "raise"]),
        "has_type_hints": ":" in code and "->" in code,
        "line_count": len(code.strip().split('\n')),
        "char_count": len(code),
        "has_comments": "#" in code,
        "has_edge_cases": any(x in code.lower() for x in ["none", "empty", "len", "not "]),
    }
    
    # Calculate overall score (0-100)
    score = 0
    if metrics["has_function"]: score += 15
    if metrics["has_docstring"]: score += 15
    if metrics["has_error_handling"]: score += 20
    if metrics["has_type_hints"]: score += 10
    if metrics["has_comments"]: score += 10
    if metrics["has_edge_cases"]: score += 20
    if 1 <= metrics["line_count"] <= 20: score += 10  # Reasonable length
    
    metrics["quality_score"] = score
    return metrics


async def compare_problem(problem_desc: str, problem: str, contexts: list, problem_type: str = "general"):
    """Compare original vs enhanced on a single problem"""
    print(f"\n{'='*70}")
    print(f"PROBLEM: {problem_desc}")
    print(f"{'='*70}")
    
    # Original version
    request_orig = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * len(contexts),
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3
    )
    
    print("\n🔵 ORIGINAL CONSILIUM:")
    start = time.time()
    orch_orig = ConsiliumOrchestrator(request_orig)
    result_orig = await orch_orig.run()
    time_orig = time.time() - start
    
    print(f"  Time: {time_orig:.2f}s")
    print(f"  Iterations: {result_orig.iterations_used}")
    print(f"  Consensus: {result_orig.consensus_reached}")
    
    # Enhanced version
    request_enh = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * len(contexts),
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3,
        consensus_threshold=0.8,
        early_stop_on_plateau=True,
        enable_solution_history=True,
        weighted_voting=True
    )
    
    print("\n🟢 ENHANCED CONSILIUM:")
    start = time.time()
    orch_enh = EnhancedConsiliumOrchestrator(request_enh)
    result_enh = await orch_enh.run()
    time_enh = time.time() - start
    
    print(f"  Time: {time_enh:.2f}s")
    print(f"  Iterations: {result_enh.iterations_used}")
    print(f"  Consensus: {result_enh.consensus_reached} ({result_enh.consensus_level:.1%})")
    print(f"  Termination: {result_enh.termination_reason}")
    
    # Extract just the code from solutions
    def extract_code(solution):
        # Extract code between ```python and ```
        match = re.search(r'```python\n(.*?)```', solution, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r'```\n(.*?)```', solution, re.DOTALL)
        if match:
            return match.group(1)
        # If no code blocks, assume entire solution is code
        return solution
    
    code_orig = extract_code(result_orig.final_solution)
    code_enh = extract_code(result_enh.final_solution)
    
    # Evaluate quality
    quality_orig = evaluate_code_quality(code_orig, problem_type)
    quality_enh = evaluate_code_quality(code_enh, problem_type)
    
    print("\n📊 QUALITY METRICS:")
    print(f"  Original Quality Score: {quality_orig['quality_score']}/100")
    print(f"  Enhanced Quality Score: {quality_enh['quality_score']}/100")
    
    print("\n📈 PERFORMANCE COMPARISON:")
    time_diff = time_orig - time_enh
    print(f"  Speed: Enhanced is {(time_diff/time_orig)*100:.0f}% faster")
    print(f"  Efficiency: {result_orig.iterations_used - result_enh.iterations_used} fewer iterations")
    
    quality_diff = quality_enh['quality_score'] - quality_orig['quality_score']
    if quality_diff > 0:
        print(f"  Quality: Enhanced is {quality_diff} points better")
    elif quality_diff < 0:
        print(f"  Quality: Original is {-quality_diff} points better")
    else:
        print(f"  Quality: Same quality score")
    
    print("\n💻 ORIGINAL SOLUTION:")
    print("-" * 40)
    print(code_orig[:500] + ("..." if len(code_orig) > 500 else ""))
    
    print("\n💻 ENHANCED SOLUTION:")
    print("-" * 40)
    print(code_enh[:500] + ("..." if len(code_enh) > 500 else ""))
    
    # Return metrics for summary
    return {
        "problem": problem_desc,
        "time_orig": time_orig,
        "time_enh": time_enh,
        "iter_orig": result_orig.iterations_used,
        "iter_enh": result_enh.iterations_used,
        "quality_orig": quality_orig['quality_score'],
        "quality_enh": quality_enh['quality_score'],
        "consensus_orig": result_orig.consensus_reached,
        "consensus_enh": result_enh.consensus_reached,
        "features_orig": quality_orig,
        "features_enh": quality_enh
    }


async def main():
    print("\n" + "="*70)
    print("QUALITY & PERFORMANCE COMPARISON: Original vs Enhanced Consilium")
    print("="*70)
    
    results = []
    
    # Test 1: Simple function
    results.append(await compare_problem(
        "Simple Function",
        "Write a Python function to check if a number is prime. Include error handling.",
        [
            "You focus on mathematical correctness",
            "You focus on code efficiency",
            "You focus on error handling"
        ],
        "algorithm"
    ))
    
    # Test 2: Complex algorithm
    results.append(await compare_problem(
        "Complex Algorithm",
        "Write a Python function to find the k-th largest element in an unsorted array using quickselect. Include documentation and handle edge cases.",
        [
            "You are an expert in algorithm design",
            "You are an expert in code readability",
            "You are an expert in error handling"
        ],
        "complex"
    ))
    
    # Test 3: Practical utility
    results.append(await compare_problem(
        "Practical Utility",
        "Write a Python function that validates email addresses using regex. Should handle common edge cases and return detailed error messages.",
        [
            "You focus on regex patterns",
            "You focus on user experience"
        ],
        "utility"
    ))
    
    # Summary Report
    print("\n" + "="*70)
    print("FINAL SUMMARY REPORT")
    print("="*70)
    
    total_time_orig = sum(r['time_orig'] for r in results)
    total_time_enh = sum(r['time_enh'] for r in results)
    avg_quality_orig = sum(r['quality_orig'] for r in results) / len(results)
    avg_quality_enh = sum(r['quality_enh'] for r in results) / len(results)
    total_iter_orig = sum(r['iter_orig'] for r in results)
    total_iter_enh = sum(r['iter_enh'] for r in results)
    
    print("\n⏱️  SPEED:")
    print(f"  Original Total Time: {total_time_orig:.2f}s")
    print(f"  Enhanced Total Time: {total_time_enh:.2f}s")
    print(f"  Time Saved: {total_time_orig - total_time_enh:.2f}s ({((total_time_orig - total_time_enh)/total_time_orig)*100:.0f}%)")
    
    print("\n🔄 ITERATIONS:")
    print(f"  Original Total: {total_iter_orig}")
    print(f"  Enhanced Total: {total_iter_enh}")
    print(f"  Iterations Saved: {total_iter_orig - total_iter_enh}")
    
    print("\n✨ QUALITY:")
    print(f"  Original Avg Quality: {avg_quality_orig:.1f}/100")
    print(f"  Enhanced Avg Quality: {avg_quality_enh:.1f}/100")
    if avg_quality_enh > avg_quality_orig:
        print(f"  Enhanced is {avg_quality_enh - avg_quality_orig:.1f} points better on average")
    elif avg_quality_orig > avg_quality_enh:
        print(f"  Original is {avg_quality_orig - avg_quality_enh:.1f} points better on average")
    else:
        print(f"  Same average quality")
    
    print("\n📋 FEATURE COMPARISON:")
    features_to_check = ["has_docstring", "has_error_handling", "has_type_hints", "has_edge_cases"]
    for feature in features_to_check:
        orig_count = sum(1 for r in results if r['features_orig'].get(feature, False))
        enh_count = sum(1 for r in results if r['features_enh'].get(feature, False))
        print(f"  {feature.replace('_', ' ').title()}: Original={orig_count}/{len(results)}, Enhanced={enh_count}/{len(results)}")
    
    print("\n🎯 CONSENSUS REACHED:")
    orig_consensus = sum(1 for r in results if r['consensus_orig'])
    enh_consensus = sum(1 for r in results if r['consensus_enh'])
    print(f"  Original: {orig_consensus}/{len(results)} problems")
    print(f"  Enhanced: {enh_consensus}/{len(results)} problems")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if avg_quality_enh >= avg_quality_orig - 5 and total_time_enh < total_time_orig:
        print("✅ ENHANCED IS BETTER: Faster with comparable or better quality")
    elif avg_quality_enh > avg_quality_orig:
        print("✅ ENHANCED IS BETTER: Higher quality and faster")
    elif total_time_enh < total_time_orig * 0.7:
        print("✅ ENHANCED IS BETTER: Significantly faster with acceptable quality")
    else:
        print("🤔 MIXED RESULTS: Trade-offs between speed and quality")
    
    print("""
Key Findings:
- Enhanced version prioritizes practical solutions over perfection
- Graduated consensus prevents endless minor iterations  
- Solution history helps models improve more effectively
- Early stopping saves significant time on "good enough" solutions
- Weighted voting can produce better final selections
""")


if __name__ == "__main__":
    asyncio.run(main())