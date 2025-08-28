Consilium - Product Requirements Document
System Overview
Multi-model consensus system where LLM models collaborate to solve problems through iteration, critique, and voting.
Core Requirements
1. User Configuration
Users provide the following inputs:

Models: List of models to use (any number, can include duplicates)

Example: ["gpt-4", "gpt-4", "claude-3", "gemini"]


Initial contexts: Individual context string for each model instance

Example: ["You are an expert in Python", "You are an expert in JavaScript", "You focus on security", "You focus on performance"]


Problem: Single problem statement shared by all models
Max iterations: Integer from 1 to unlimited

2. Process Flow
The system executes the following phases:
Phase 1: Initial Generation

Each model receives their individual initial context + the problem
Each model generates an initial solution
Solutions are collected for critique phase

Phase 2: Critique

Every model critiques every solution (including their own)
Critique output is structured:

json{
  "no_critique_needed": boolean,
  "critiques": ["issue 1", "issue 2", ...]
}

If no_critique_needed is true, the critiques array should be empty
Models evaluate all solutions in parallel

Phase 3: Decision Point

If max_iterations = 1: Skip to Voting Phase
Otherwise: Continue to Improvement Phase

Phase 4: Improvement Loop
For each iteration:

Context for improvement: Each model receives:

The original problem
Their last proposed solution
All critiques from other models about their solution


Generate improved solution: Each model produces a new solution
Return to Critique Phase: All models critique all new solutions
Check stop conditions:

Consensus reached: If any solution receives no_critique_needed=true from ALL other models → Return that solution
Max iterations reached: If iteration count = max_iterations → Go to Voting Phase
Otherwise: Continue loop



Phase 5: Voting

Each model votes for the single best solution
Models cannot vote for their own solution
Solution with most votes is returned
In case of tie, select randomly from tied solutions

3. Context Management
What persists between iterations:

The original problem statement
Each model's last solution
All critiques from the last round

What does NOT persist:

Initial contexts (only used in Phase 1)
Solutions from previous iterations (only keep most recent)
Critiques from previous iterations (only keep most recent)

4. Technical Implementation
Required Components:

PydanticAI for model management
Existing code from /llm directory
Parallel execution for critiques and voting

Data Models:
python# User Input
class ConsiliumRequest:
    models: List[str]              # ["gpt-4", "claude-3", ...]
    initial_contexts: List[str]    # Individual contexts
    problem: str                   # Problem to solve
    max_iterations: int            # >= 1

# Critique Structure  
class Critique:
    no_critique_needed: bool
    critiques: List[str]

# Solution Structure
class Solution:
    model_index: int              # Which model generated this
    content: str                  # The actual solution

# Final Result
class ConsiliumResult:
    final_solution: str
    iterations_used: int
    consensus_reached: bool
    winning_model_index: int
Key Functions:
pythonasync def run_consilium(request: ConsiliumRequest) -> ConsiliumResult:
    # Phase 1: Initial generation
    # Phase 2: Critique loop
    # Phase 3: Check consensus or iterate
    # Phase 4: Vote if needed
    # Return result
5. Edge Cases

Single model: System should still work with just one model (will always reach consensus)
All models agree immediately: Return solution after first iteration
No consensus after max iterations: Must use voting
Tie in voting: Random selection from tied solutions
Empty critiques: If no_critique_needed=true, critiques array must be empty

6. Performance Considerations

Parallelize all critique operations (N models critiquing N solutions)
Parallelize voting phase
Sequential execution for improvement iterations
No caching in initial version
No rate limiting in initial version (add later if needed)

7. Success Criteria
The system successfully:

Accepts variable number of models
Allows individual initial contexts
Iterates until consensus or max iterations
Returns a single solution
Uses structured critique format
Implements proper context management between iterations

Out of Scope for Initial Version

Authentication/authorization
Cost tracking
Result persistence
Caching
Rate limiting
Web UI
Error recovery beyond basic exception handling
Prompt optimization
Model-specific configurations
Streaming responses
Progress indicators