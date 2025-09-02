import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv
from consilium import ConsiliumRequest, run_consilium

load_dotenv('.env.local')


async def compare_modes():
    """Compare strict vs regular consensus modes"""
    
    problem = "Write a Python function to find the second largest number in a list. Handle edge cases."
    contexts = [
        "You focus on algorithmic efficiency",
        "You focus on error handling and robustness",
        "You focus on code clarity and documentation"
    ]
    
    print("=" * 70)
    print("COMPARING: Regular vs Strict Consensus Modes")
    print("=" * 70)
    print(f"\nProblem: {problem}\n")
    
    # Test regular enhanced mode
    print("🟢 ENHANCED MODE (80% consensus threshold):")
    print("-" * 40)
    
    request_regular = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * 3,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=5,
        consensus_threshold=0.8,
        early_stop_on_plateau=True,
        strict_mode=False
    )
    
    start = time.time()
    result_regular = await run_consilium(request_regular)
    time_regular = time.time() - start
    
    print(f"  Time: {time_regular:.1f}s")
    print(f"  Iterations: {result_regular.iterations_used}/{request_regular.max_iterations}")
    print(f"  Consensus: {result_regular.consensus_reached}")
    print(f"  Consensus level: {result_regular.consensus_level:.1%}")
    print(f"  Termination: {result_regular.termination_reason}")
    
    # Test strict mode
    print("\n🔴 STRICT MODE (harder to reach consensus):")
    print("-" * 40)
    
    request_strict = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * 3,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=5,
        consensus_threshold=0.8,
        early_stop_on_plateau=True,
        strict_mode=True,
        force_iterations=2,  # Force at least 2 iterations
        require_perfection_early=True,
        diversity_threshold=0.4
    )
    
    start = time.time()
    result_strict = await run_consilium(request_strict)
    time_strict = time.time() - start
    
    print(f"  Time: {time_strict:.1f}s")
    print(f"  Iterations: {result_strict.iterations_used}/{request_strict.max_iterations}")
    print(f"  Consensus: {result_strict.consensus_reached}")
    print(f"  Consensus level: {result_strict.consensus_level:.1%}")
    print(f"  Termination: {result_strict.termination_reason}")
    
    # Compare improvement trajectories
    print("\n📊 IMPROVEMENT TRAJECTORIES:")
    print("-" * 40)
    if result_regular.improvement_trajectory:
        print(f"  Regular: {[f'{x:.2%}' for x in result_regular.improvement_trajectory]}")
    if result_strict.improvement_trajectory:
        print(f"  Strict:  {[f'{x:.2%}' for x in result_strict.improvement_trajectory]}")
    
    # Summary
    print("\n📈 COMPARISON SUMMARY:")
    print("-" * 40)
    print(f"  Iterations: Regular={result_regular.iterations_used}, Strict={result_strict.iterations_used}")
    print(f"  Time: Regular={time_regular:.1f}s, Strict={time_strict:.1f}s")
    print(f"  Consensus Level: Regular={result_regular.consensus_level:.1%}, Strict={result_strict.consensus_level:.1%}")
    
    if result_strict.iterations_used > result_regular.iterations_used:
        print(f"\n✅ Strict mode forced {result_strict.iterations_used - result_regular.iterations_used} more iterations")
        print("   This likely resulted in more refined solutions through additional critique cycles")
    
    # Show brief solution excerpts
    print("\n💻 SOLUTION EXCERPTS:")
    print("-" * 40)
    
    def extract_first_lines(solution, n=3):
        lines = solution.strip().split('\n')
        return '\n'.join(lines[:n]) + ('...' if len(lines) > n else '')
    
    print("Regular mode solution start:")
    print(extract_first_lines(result_regular.final_solution, 5))
    
    print("\nStrict mode solution start:")
    print(extract_first_lines(result_strict.final_solution, 5))
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("""
Strict Mode Effects:
✓ Forces minimum iterations (prevents premature consensus)
✓ Requires higher critique standards early on
✓ Encourages solution diversity
✓ Results in more thorough exploration of solution space
✓ May produce higher quality at the cost of more time/iterations

Best for: Critical systems, production code, when quality > speed
Regular Mode Best for: Rapid prototyping, exploration, when speed matters
""")


if __name__ == "__main__":
    asyncio.run(compare_modes())