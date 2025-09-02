import asyncio
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from consilium import ConsiliumRequest
from consilium.core import ConsiliumOrchestrator
from consilium.core_v2 import EnhancedConsiliumOrchestrator

load_dotenv('.env.local')

async def compare_single_problem():
    problem = 'Write a Python function to reverse a string. Include a docstring and handle None input.'
    contexts = [
        'You focus on code clarity and documentation',
        'You focus on error handling and edge cases'
    ]
    
    print('='*60)
    print('COMPARING: String Reversal Function')
    print('='*60)
    
    # Original
    request_orig = ConsiliumRequest(
        models=['openai:gpt-4o-mini', 'openai:gpt-4o-mini'],
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3
    )
    
    print('\n🔵 ORIGINAL VERSION:')
    start = time.time()
    orch_orig = ConsiliumOrchestrator(request_orig)
    result_orig = await orch_orig.run()
    time_orig = time.time() - start
    
    print(f'Time: {time_orig:.1f}s, Iterations: {result_orig.iterations_used}')
    
    # Enhanced
    request_enh = ConsiliumRequest(
        models=['openai:gpt-4o-mini', 'openai:gpt-4o-mini'],
        initial_contexts=contexts,
        problem=problem,
        max_iterations=3,
        consensus_threshold=0.8,
        early_stop_on_plateau=True,
        enable_solution_history=True
    )
    
    print('\n🟢 ENHANCED VERSION:')
    start = time.time()
    orch_enh = EnhancedConsiliumOrchestrator(request_enh)
    result_enh = await orch_enh.run()
    time_enh = time.time() - start
    
    print(f'Time: {time_enh:.1f}s, Iterations: {result_enh.iterations_used}')
    print(f'Consensus: {result_enh.consensus_level:.0%}, Reason: {result_enh.termination_reason}')
    
    # Extract code
    def extract_code(solution):
        match = re.search(r'```(?:python)?\n(.*?)```', solution, re.DOTALL)
        return match.group(1) if match else solution
    
    code_orig = extract_code(result_orig.final_solution)
    code_enh = extract_code(result_enh.final_solution)
    
    # Quick quality check
    def check_quality(code):
        return {
            'has_docstring': '"""' in code or "'''" in code,
            'handles_none': 'None' in code or 'none' in code.lower(),
            'has_function': 'def ' in code,
            'lines': len(code.strip().split('\n'))
        }
    
    q_orig = check_quality(code_orig)
    q_enh = check_quality(code_enh)
    
    print('\n📊 QUALITY CHECK:')
    print(f'Original: docstring={q_orig["has_docstring"]}, None handling={q_orig["handles_none"]}, lines={q_orig["lines"]}')
    print(f'Enhanced: docstring={q_enh["has_docstring"]}, None handling={q_enh["handles_none"]}, lines={q_enh["lines"]}')
    
    print('\n⚡ PERFORMANCE:')
    print(f'Enhanced is {((time_orig-time_enh)/time_orig)*100:.0f}% faster')
    print(f'Saved {result_orig.iterations_used - result_enh.iterations_used} iterations')
    
    print('\n--- ORIGINAL SOLUTION ---')
    print(code_orig[:400] + ('...' if len(code_orig) > 400 else ''))
    
    print('\n--- ENHANCED SOLUTION ---')
    print(code_enh[:400] + ('...' if len(code_enh) > 400 else ''))

asyncio.run(compare_single_problem())