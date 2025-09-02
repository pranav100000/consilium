# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Consilium - Multi-Model Consensus System

A system where multiple LLM models collaborate through iteration, critique, and voting to solve problems.

## Commands

### Run Tests
```bash
python -m pytest tests/test_consilium.py -v
```

### Run Application
- Interactive mode: `python main.py -i`
- From JSON file: `python main.py -f request.json`

### Install Dependencies
```bash
pip install pydantic-ai python-dotenv
```

## Architecture Overview

The Consilium system operates in 5 phases:

1. **Initial Generation** - Each model generates an initial solution
2. **Critique Phase** - Every model critiques every solution (including their own) with severity levels
3. **Consensus Check** - Check if consensus threshold is reached (graduated consensus)
4. **Improvement Phase** - Models improve solutions based on aggregated critiques
5. **Voting Phase** - If no consensus, models vote for best solution (with optional weighted voting)

### Key Components

- **`consilium/core_v2.py`**: Enhanced orchestrator with performance tracking, graduated consensus, and critique aggregation
- **`consilium/models.py`**: Data models for requests, results, critiques with severity levels
- **`main.py`**: CLI entry point supporting interactive mode and JSON file input
- **Environment**: Uses `.env.local` for API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)

### Enhanced Features

- **Graduated Consensus**: Configurable consensus threshold (default 0.7)
- **Critique Severity Levels**: PERFECT, MINOR_ISSUES, MAJOR_ISSUES, FUNDAMENTAL_FLAWS
- **Performance Tracking**: Model improvement rates, approval metrics, weighted voting
- **Early Stopping**: Can stop on plateau or when only minor issues remain
- **Solution History**: Optional tracking of solution evolution across iterations