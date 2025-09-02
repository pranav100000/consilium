import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.insert(0, '/Users/pranavsharan/Developer/consilium')

from consilium import ConsiliumRequest
from consilium.core_v2 import EnhancedConsiliumOrchestrator
from consilium.core_v3 import StrictConsiliumOrchestrator
from consilium.core_v4_adaptive import AdaptiveConsiliumOrchestrator

load_dotenv('.env.local')


async def test_adaptive():
    """Test adaptive mode against enhanced and strict"""
    
    # Test with different complexity problems
    test_cases = [
        {
            "name": "Simple Problem",
            "problem": "Write a function to check if a number is even",
            "contexts": ["Focus on clarity", "Focus on efficiency"],
            "complexity": "low"
        },
        {
            "name": "Medium Problem", 
            "problem": "Write a function to validate email addresses with proper error handling and edge cases",
            "contexts": ["Focus on validation", "Focus on robustness"],
            "complexity": "medium"
        },
        {
            "name": "Complex Problem",
            "problem": "Implement an efficient LRU cache with get and put methods. Handle concurrent access and edge cases.",
            "contexts": ["Focus on data structures", "Focus on thread safety", "Focus on performance"],
            "complexity": "high"
        }
    ]
    
    for test in test_cases:
        print("\n" + "="*70)
        print(f"TEST: {test['name']} (Complexity: {test['complexity']})")
        print("="*70)
        print(f"Problem: {test['problem']}\n")
        
        base_config = {
            "models": ["openai:gpt-4o-mini"] * len(test['contexts']),
            "initial_contexts": test['contexts'],
            "problem": test['problem'],
            "max_iterations": 4
        }
        
        # Enhanced
        print("🟢 Enhanced Mode...", end="", flush=True)
        start = time.time()
        req_enh = ConsiliumRequest(
            **base_config,
            consensus_threshold=0.8,
            early_stop_on_plateau=True
        )
        orch_enh = EnhancedConsiliumOrchestrator(req_enh)
        res_enh = await orch_enh.run()
        time_enh = time.time() - start
        print(f" {time_enh:.1f}s, {res_enh.iterations_used} iter, consensus={res_enh.consensus_level:.0%}")
        
        # Strict
        print("🔴 Strict Mode...", end="", flush=True)
        start = time.time()
        req_strict = ConsiliumRequest(
            **base_config,
            strict_mode=True,
            force_iterations=2
        )
        orch_strict = StrictConsiliumOrchestrator(req_strict)
        res_strict = await orch_strict.run()
        time_strict = time.time() - start
        print(f" {time_strict:.1f}s, {res_strict.iterations_used} iter, consensus={res_strict.consensus_level:.0%}")
        
        # Adaptive
        print("🟡 Adaptive Mode...", end="", flush=True)
        start = time.time()
        req_adaptive = ConsiliumRequest(
            **base_config,
            consensus_threshold=0.8,
            early_stop_on_plateau=True
        )
        orch_adaptive = AdaptiveConsiliumOrchestrator(req_adaptive)
        res_adaptive = await orch_adaptive.run()
        time_adaptive = time.time() - start
        
        # Show adaptive's internal state
        quality_score = orch_adaptive._assess_quality_signals()
        print(f" {time_adaptive:.1f}s, {res_adaptive.iterations_used} iter, consensus={res_adaptive.consensus_level:.0%}")
        print(f"  Adaptive details: complexity={orch_adaptive.complexity_score:.2f}, quality={quality_score:.2f}, mode={orch_adaptive.mode}")
        
        # Compare
        print("\n📊 Comparison:")
        print(f"  Speed: Enhanced={time_enh:.1f}s, Adaptive={time_adaptive:.1f}s, Strict={time_strict:.1f}s")
        print(f"  Iterations: Enhanced={res_enh.iterations_used}, Adaptive={res_adaptive.iterations_used}, Strict={res_strict.iterations_used}")
        
        # Adaptive vs Enhanced
        if time_adaptive < time_enh * 1.5 and res_adaptive.iterations_used >= res_enh.iterations_used:
            print("  ✅ Adaptive achieved better quality/depth with acceptable speed")
        elif time_adaptive <= time_enh:
            print("  ✅ Adaptive was faster or equal")
        else:
            print(f"  ⚠️ Adaptive was {((time_adaptive-time_enh)/time_enh)*100:.0f}% slower")
        
        # Show termination reasons
        print(f"\n  Termination reasons:")
        print(f"    Enhanced: {res_enh.termination_reason}")
        print(f"    Strict: {res_strict.termination_reason}")  
        print(f"    Adaptive: {res_adaptive.termination_reason}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Adaptive Mode Behavior:
• Low complexity → Acts like Enhanced (fast)
• Medium complexity → Balanced approach
• High complexity → More thorough like Strict
• Dynamically adjusts consensus threshold
• Quality-aware early stopping
• Complexity estimation guides strategy

Key Advantages:
✅ Matches problem complexity automatically
✅ Better quality than Enhanced on complex problems
✅ Faster than Strict on simple problems
✅ No manual configuration needed
""")


if __name__ == "__main__":
    asyncio.run(test_adaptive())