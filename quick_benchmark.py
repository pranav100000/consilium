import asyncio
import time
import re
from pathlib import Path
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


class QuickJudgment(BaseModel):
    winner: str = Field(description="'A', 'B', or 'C'")
    scores: dict = Field(description="Scores for A, B, C")
    reasoning: str


async def quick_test():
    """Quick benchmark of all three systems"""
    
    problem = "Write a Python function to check if a string is a palindrome. Handle edge cases and add a docstring."
    contexts = [
        "You focus on code clarity",
        "You focus on efficiency"
    ]
    
    print("="*60)
    print("QUICK BENCHMARK: Original vs Enhanced vs Strict")
    print("="*60)
    print(f"Problem: {problem}\n")
    
    # Original
    print("🔵 Running Original...", end="", flush=True)
    start = time.time()
    req1 = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * 2,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3
    )
    orch1 = ConsiliumOrchestrator(req1)
    res1 = await orch1.run()
    time1 = time.time() - start
    print(f" {time1:.1f}s, {res1.iterations_used} iter")
    
    # Enhanced
    print("🟢 Running Enhanced...", end="", flush=True)
    start = time.time()
    req2 = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * 2,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3,
        consensus_threshold=0.8,
        early_stop_on_plateau=True
    )
    orch2 = EnhancedConsiliumOrchestrator(req2)
    res2 = await orch2.run()
    time2 = time.time() - start
    print(f" {time2:.1f}s, {res2.iterations_used} iter")
    
    # Strict
    print("🔴 Running Strict...", end="", flush=True)
    start = time.time()
    req3 = ConsiliumRequest(
        models=["openai:gpt-4o-mini"] * 2,
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3,
        strict_mode=True,
        force_iterations=2
    )
    orch3 = StrictConsiliumOrchestrator(req3)
    res3 = await orch3.run()
    time3 = time.time() - start
    print(f" {time3:.1f}s, {res3.iterations_used} iter")
    
    # Extract code
    def extract(s):
        m = re.search(r'```(?:python)?\n(.*?)```', s, re.DOTALL)
        return m.group(1) if m else s
    
    code1 = extract(res1.final_solution)[:500]
    code2 = extract(res2.final_solution)[:500]
    code3 = extract(res3.final_solution)[:500]
    
    # Judge
    print("\n🧑‍⚖️ Judging solutions...")
    judge = Agent(
        model='openai:gpt-4o-mini',
        output_type=QuickJudgment,
        system_prompt="Judge code quality objectively. A=Original, B=Enhanced, C=Strict"
    )
    
    judgment = await judge.run(
        f"Problem: {problem}\n\n"
        f"Solution A:\n{code1}\n\n"
        f"Solution B:\n{code2}\n\n"
        f"Solution C:\n{code3}\n\n"
        f"Which is best?"
    )
    
    # Results
    print("\n📊 RESULTS:")
    print(f"Performance: Orig={time1:.1f}s, Enh={time2:.1f}s, Strict={time3:.1f}s")
    print(f"Iterations: Orig={res1.iterations_used}, Enh={res2.iterations_used}, Strict={res3.iterations_used}")
    print(f"Quality Scores: {judgment.output.scores}")
    print(f"Winner: {judgment.output.winner} ({'Original' if judgment.output.winner=='A' else 'Enhanced' if judgment.output.winner=='B' else 'Strict'})")
    print(f"Reasoning: {judgment.output.reasoning}")
    
    # Analysis
    print("\n📈 ANALYSIS:")
    efficiency = {
        "Original": judgment.output.scores.get('A', 0) / time1,
        "Enhanced": judgment.output.scores.get('B', 0) / time2,
        "Strict": judgment.output.scores.get('C', 0) / time3
    }
    best_efficiency = max(efficiency.items(), key=lambda x: x[1])
    print(f"Best Efficiency: {best_efficiency[0]} ({best_efficiency[1]:.1f} points/sec)")
    
    if res2.iterations_used < res1.iterations_used:
        print(f"Enhanced saved {((res1.iterations_used - res2.iterations_used)/res1.iterations_used)*100:.0f}% iterations")
    if res3.iterations_used > res1.iterations_used:
        print(f"Strict used {((res3.iterations_used - res1.iterations_used)/res1.iterations_used)*100:.0f}% more iterations")

asyncio.run(quick_test())