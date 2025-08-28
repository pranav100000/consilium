import asyncio
import random
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from .models import ConsiliumRequest, ConsiliumResult, Critique, Solution

# Load environment variables from .env.local
env_path = Path(__file__).parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try to load from current directory
    load_dotenv('.env.local')


class ConsiliumOrchestrator:
    def __init__(self, request: ConsiliumRequest):
        self.request = request
        self.agents: List[Agent] = []
        self.solutions: Dict[int, Solution] = {}
        self.critiques: Dict[Tuple[int, int], Critique] = {}
        self.iteration = 0
        self._initialize_agents()
    
    def _initialize_agents(self):
        for i, model in enumerate(self.request.models):
            agent = Agent(
                model=model,
                system_prompt=self.request.initial_contexts[i] if i < len(self.request.initial_contexts) else ""
            )
            self.agents.append(agent)
    
    async def run(self) -> ConsiliumResult:
        await self._phase_1_initial_generation()
        
        consensus_solution = None
        
        for iteration in range(1, self.request.max_iterations + 1):
            self.iteration = iteration
            
            await self._phase_2_critique()
            
            consensus_solution = self._check_consensus()
            if consensus_solution:
                return ConsiliumResult(
                    final_solution=consensus_solution.content,
                    iterations_used=iteration,
                    consensus_reached=True,
                    winning_model_index=consensus_solution.model_index,
                    all_solutions=list(self.solutions.values())
                )
            
            if iteration < self.request.max_iterations:
                await self._phase_4_improvement()
        
        winning_solution = await self._phase_5_voting()
        
        return ConsiliumResult(
            final_solution=winning_solution.content,
            iterations_used=self.request.max_iterations,
            consensus_reached=False,
            winning_model_index=winning_solution.model_index,
            all_solutions=list(self.solutions.values())
        )
    
    async def _phase_1_initial_generation(self):
        tasks = []
        for i, agent in enumerate(self.agents):
            prompt = f"{self.request.initial_contexts[i]}\n\nProblem: {self.request.problem}" if i < len(self.request.initial_contexts) else self.request.problem
            tasks.append(self._generate_solution(agent, i, prompt))
        
        results = await asyncio.gather(*tasks)
        for i, solution_text in enumerate(results):
            self.solutions[i] = Solution(
                model_index=i,
                content=solution_text,
                iteration=0
            )
    
    async def _generate_solution(self, agent: Agent, index: int, prompt: str, message_history: Optional[List[ModelMessage]] = None) -> str:
        result = await agent.run(prompt, message_history=message_history)
        return result.output
    
    async def _phase_2_critique(self):
        self.critiques.clear()
        tasks = []
        
        for critic_idx, critic_agent in enumerate(self.agents):
            for solution_idx, solution in self.solutions.items():
                tasks.append(self._critique_solution(critic_agent, critic_idx, solution_idx, solution))
        
        critiques = await asyncio.gather(*tasks)
        
        for (critic_idx, solution_idx), critique in critiques:
            self.critiques[(critic_idx, solution_idx)] = critique
    
    async def _critique_solution(self, agent: Agent, critic_idx: int, solution_idx: int, solution: Solution) -> Tuple[Tuple[int, int], Critique]:
        critique_agent = Agent(
            model=agent.model,
            output_type=Critique,
            system_prompt="You are a critical reviewer. Analyze the solution and provide structured feedback."
        )
        
        prompt = f"""Problem: {self.request.problem}

Solution to critique:
{solution.content}

Provide your critique. If the solution is perfect and needs no improvement, set no_critique_needed to true and leave critiques empty.
Otherwise, set no_critique_needed to false and list specific issues or improvements needed."""
        
        try:
            result = await critique_agent.run(prompt)
            critique = result.output
            if critique.no_critique_needed:
                critique.critiques = []
            return ((critic_idx, solution_idx), critique)
        except Exception as e:
            print(f"Error in critique from model {critic_idx} for solution {solution_idx}: {e}")
            return ((critic_idx, solution_idx), Critique(no_critique_needed=False, critiques=[f"Error during critique: {str(e)}"]))
    
    def _check_consensus(self) -> Optional[Solution]:
        for solution_idx, solution in self.solutions.items():
            all_approve = True
            for critic_idx in range(len(self.agents)):
                if critic_idx == solution_idx:
                    continue
                critique = self.critiques.get((critic_idx, solution_idx))
                if not critique or not critique.no_critique_needed:
                    all_approve = False
                    break
            
            if all_approve and len(self.agents) > 1:
                return solution
        
        if len(self.agents) == 1:
            critique = self.critiques.get((0, 0))
            if critique and critique.no_critique_needed:
                return self.solutions[0]
        
        return None
    
    async def _phase_4_improvement(self):
        tasks = []
        for i, agent in enumerate(self.agents):
            current_solution = self.solutions[i]
            critiques_for_solution = []
            
            for critic_idx in range(len(self.agents)):
                if critic_idx != i:
                    critique = self.critiques.get((critic_idx, i))
                    if critique and critique.critiques:
                        critiques_for_solution.extend(critique.critiques)
            
            if critiques_for_solution:
                prompt = f"""Problem: {self.request.problem}

Your previous solution:
{current_solution.content}

Critiques from other models:
{chr(10).join(f'- {c}' for c in critiques_for_solution)}

Please provide an improved solution addressing these critiques."""
            else:
                prompt = f"""Problem: {self.request.problem}

Your previous solution:
{current_solution.content}

No critiques were provided. Please review and potentially improve your solution."""
            
            tasks.append(self._generate_solution(agent, i, prompt))
        
        results = await asyncio.gather(*tasks)
        for i, solution_text in enumerate(results):
            self.solutions[i] = Solution(
                model_index=i,
                content=solution_text,
                iteration=self.iteration
            )
    
    async def _phase_5_voting(self) -> Solution:
        if len(self.agents) == 1:
            return self.solutions[0]
        
        votes = {}
        tasks = []
        
        for voter_idx, voter_agent in enumerate(self.agents):
            tasks.append(self._cast_vote(voter_agent, voter_idx))
        
        vote_results = await asyncio.gather(*tasks)
        
        for voted_idx in vote_results:
            if voted_idx is not None:
                votes[voted_idx] = votes.get(voted_idx, 0) + 1
        
        if not votes:
            return random.choice(list(self.solutions.values()))
        
        max_votes = max(votes.values())
        winners = [idx for idx, count in votes.items() if count == max_votes]
        
        if len(winners) == 1:
            return self.solutions[winners[0]]
        else:
            winner_idx = random.choice(winners)
            return self.solutions[winner_idx]
    
    async def _cast_vote(self, agent: Agent, voter_idx: int) -> Optional[int]:
        solutions_text = []
        valid_indices = []
        
        for idx, solution in self.solutions.items():
            if idx != voter_idx:
                solutions_text.append(f"Solution {idx}:\n{solution.content}")
                valid_indices.append(idx)
        
        if not valid_indices:
            return None
        
        prompt = f"""Problem: {self.request.problem}

Review the following solutions and select the best one by responding with ONLY the solution number.

{chr(10).join(solutions_text)}

Which solution is best? Reply with just the number."""
        
        try:
            result = await agent.run(prompt)
            response = result.output.strip()
            
            for idx in valid_indices:
                if str(idx) in response:
                    return idx
            
            return random.choice(valid_indices)
        except Exception as e:
            print(f"Error in voting from model {voter_idx}: {e}")
            return random.choice(valid_indices) if valid_indices else None


async def run_consilium(request: ConsiliumRequest) -> ConsiliumResult:
    orchestrator = ConsiliumOrchestrator(request)
    return await orchestrator.run()