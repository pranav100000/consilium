# Consilium

Multi-model consensus system where LLM models collaborate to solve problems through iteration, critique, and voting.

## Installation

```bash
pip install pydantic-ai
```

Set up API keys:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # Optional
```

## Quick Start

### Interactive Mode

```bash
python main.py -i
```

### From JSON File

Create a `request.json`:
```json
{
  "models": ["openai:gpt-4o-mini", "openai:gpt-4o-mini"],
  "initial_contexts": [
    "You are an expert in creative solutions",
    "You are an expert in practical solutions"
  ],
  "problem": "How can we reduce plastic waste?",
  "max_iterations": 2
}
```

Run:
```bash
python main.py -f request.json
```

### Python API

```python
import asyncio
from consilium import ConsiliumRequest, run_consilium

async def main():
    request = ConsiliumRequest(
        models=["openai:gpt-4o-mini", "anthropic:claude-3-haiku"],
        initial_contexts=[
            "You are an expert in Python",
            "You are an expert in code review"
        ],
        problem="Write a function to calculate fibonacci numbers",
        max_iterations=2
    )
    
    result = await run_consilium(request)
    print(f"Final solution: {result.final_solution}")
    print(f"Consensus reached: {result.consensus_reached}")

asyncio.run(main())
```

## How It Works

1. **Phase 1: Initial Generation** - Each model generates an initial solution
2. **Phase 2: Critique** - Every model critiques every solution (including their own)
3. **Phase 3: Decision** - Check if consensus is reached (all models approve a solution)
4. **Phase 4: Improvement** - Models improve their solutions based on critiques
5. **Phase 5: Voting** - If no consensus after max iterations, models vote for best solution

## Supported Models

- OpenAI: `openai:gpt-4o`, `openai:gpt-4o-mini`, etc.
- Anthropic: `anthropic:claude-3-opus`, `anthropic:claude-3-sonnet`, `anthropic:claude-3-haiku`
- Any model supported by pydantic-ai

## Examples

See the `examples/` directory for more usage examples.

## Testing

```bash
pytest tests/
```
