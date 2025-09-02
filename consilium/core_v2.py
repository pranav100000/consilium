import asyncio
import random
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
import difflib

from .models import (
    ConsiliumRequest, ConsiliumResult, Critique, Solution, 
    CritiqueLevel, ModelPerformance
)

# Load environment variables from .env.local
env_path = Path(__file__).parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv('.env.local')


class EnhancedConsiliumOrchestrator:
    def __init__(self, request: ConsiliumRequest):
        self.request = request
        self.agents: List[Agent] = []
        self.solutions: Dict[int, Solution] = {}
        self.critiques: Dict[Tuple[int, int], Critique] = {}
        self.iteration = 0
        self.model_performance: Dict[int, ModelPerformance] = {}
        self.improvement_trajectory: List[float] = []
        self.previous_solutions: Dict[int, List[str]] = {}
        self._initialize_agents()
        self._initialize_performance_tracking()
    
    def _initialize_agents(self):
        for i, model in enumerate(self.request.models):
            agent = Agent(
                model=model,
                system_prompt=self.request.initial_contexts[i] if i < len(self.request.initial_contexts) else ""
            )
            self.agents.append(agent)
            self.previous_solutions[i] = []
    
    def _initialize_performance_tracking(self):
        for i in range(len(self.agents)):
            self.model_performance[i] = ModelPerformance(model_index=i)
    
    def _calculate_improvement_delta(self, old_content: str, new_content: str) -> float:
        """Calculate how much a solution improved using diff ratio"""
        if not old_content or not new_content:
            return 0.0
        ratio = difflib.SequenceMatcher(None, old_content, new_content).ratio()
        return 1.0 - ratio  # Higher number means more change
    
    def _aggregate_critiques(self, model_idx: int) -> Dict[str, any]:
        """Aggregate critiques for a model's solution"""
        critiques_for_model = []
        confidence_sum = 0
        count = 0
        
        for critic_idx in range(len(self.agents)):
            if critic_idx != model_idx:
                critique = self.critiques.get((critic_idx, model_idx))
                if critique:
                    critiques_for_model.append(critique)
                    confidence_sum += critique.confidence
                    count += 1
        
        if not critiques_for_model:
            return {"consensus_issues": [], "unique_insights": [], "priority_fixes": []}
        
        # Find consensus issues (mentioned by multiple critics)
        issue_counts = {}
        all_suggestions = []
        severity_levels = []
        
        for critique in critiques_for_model:
            for issue in critique.critiques:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            all_suggestions.extend(critique.suggestions)
            if critique.level:
                severity_levels.append(critique.level)
        
        consensus_issues = [issue for issue, count in issue_counts.items() if count > 1]
        unique_insights = [issue for issue, count in issue_counts.items() if count == 1]
        
        # Prioritize by severity
        priority_fixes = []
        if severity_levels:
            if CritiqueLevel.FUNDAMENTAL_FLAWS in severity_levels:
                priority_fixes.extend([c for c in critiques_for_model if c.level == CritiqueLevel.FUNDAMENTAL_FLAWS])
            elif CritiqueLevel.MAJOR_ISSUES in severity_levels:
                priority_fixes.extend([c for c in critiques_for_model if c.level == CritiqueLevel.MAJOR_ISSUES])
        
        return {
            "consensus_issues": consensus_issues[:5],
            "unique_insights": unique_insights[:3],
            "suggestions": list(set(all_suggestions))[:5],
            "average_confidence": confidence_sum / count if count > 0 else 0,
            "priority_level": max(severity_levels) if severity_levels else None
        }
    
    async def run(self) -> ConsiliumResult:
        await self._phase_1_initial_generation()
        
        consensus_solution = None
        termination_reason = "max_iterations"
        
        for iteration in range(1, self.request.max_iterations + 1):
            self.iteration = iteration
            
            await self._phase_2_critique_enhanced()
            
            # Check graduated consensus
            consensus_data = self._check_graduated_consensus()
            if consensus_data["solution"]:
                return ConsiliumResult(
                    final_solution=consensus_data["solution"].content,
                    iterations_used=iteration,
                    consensus_reached=True,
                    consensus_level=consensus_data["level"],
                    winning_model_index=consensus_data["solution"].model_index,
                    all_solutions=list(self.solutions.values()),
                    model_performance=list(self.model_performance.values()),
                    improvement_trajectory=self.improvement_trajectory,
                    termination_reason="consensus"
                )
            
            # Check for early stopping on minor issues only
            if self._should_stop_early():
                termination_reason = "minor_issues_only"
                break
            
            if iteration < self.request.max_iterations:
                improvement_made = await self._phase_4_improvement_enhanced()
                
                # Check for plateau
                if self.request.early_stop_on_plateau and not improvement_made:
                    termination_reason = "plateau"
                    break
        
        winning_solution = await self._phase_5_voting_enhanced()
        
        return ConsiliumResult(
            final_solution=winning_solution.content,
            iterations_used=self.iteration,
            consensus_reached=False,
            consensus_level=self._calculate_current_consensus_level(),
            winning_model_index=winning_solution.model_index,
            all_solutions=list(self.solutions.values()),
            model_performance=list(self.model_performance.values()),
            improvement_trajectory=self.improvement_trajectory,
            termination_reason=termination_reason
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
                iteration=0,
                previous_content=None,
                improvement_delta=0.0
            )
            self.previous_solutions[i].append(solution_text)
    
    async def _generate_solution(self, agent: Agent, index: int, prompt: str, message_history: Optional[List[ModelMessage]] = None) -> str:
        result = await agent.run(prompt, message_history=message_history)
        return result.output
    
    async def _phase_2_critique_enhanced(self):
        self.critiques.clear()
        tasks = []
        
        for critic_idx, critic_agent in enumerate(self.agents):
            for solution_idx, solution in self.solutions.items():
                tasks.append(self._critique_solution_enhanced(critic_agent, critic_idx, solution_idx, solution))
        
        critiques = await asyncio.gather(*tasks)
        
        for (critic_idx, solution_idx), critique in critiques:
            self.critiques[(critic_idx, solution_idx)] = critique
            # Update performance metrics
            self.model_performance[critic_idx].critiques_given += 1
            if critique.is_acceptable:
                self.model_performance[solution_idx].solutions_approved += 1
    
    async def _critique_solution_enhanced(self, agent: Agent, critic_idx: int, solution_idx: int, solution: Solution) -> Tuple[Tuple[int, int], Critique]:
        critique_agent = Agent(
            model=agent.model,
            output_type=Critique,
            system_prompt="""You are a critical reviewer. Analyze solutions and provide structured feedback with severity levels.
            
Severity Levels:
- PERFECT: No issues at all, solution is optimal
- MINOR_ISSUES: Small improvements possible (style, naming, minor optimizations)
- MAJOR_ISSUES: Significant problems that affect functionality or efficiency
- FUNDAMENTAL_FLAWS: Critical errors, wrong approach, or missing requirements

Provide:
1. Overall severity level
2. Specific issues (if any)
3. Constructive suggestions for improvement
4. Confidence score (0-1) in your assessment"""
        )
        
        # Include solution history if enabled
        history_context = ""
        if self.request.enable_solution_history and self.iteration > 1:
            prev_solutions = self.previous_solutions.get(solution_idx, [])
            if len(prev_solutions) > 1:
                history_context = f"\nPrevious iteration:\n{prev_solutions[-2]}\n\nCurrent iteration shows these changes."
        
        prompt = f"""Problem: {self.request.problem}

Solution to critique:
{solution.content}
{history_context}

Analyze this solution. Consider:
- Correctness and completeness
- Efficiency and performance
- Code quality and readability
- Error handling and edge cases

Set no_critique_needed=true and level=PERFECT only if the solution is optimal.
For any issues, set no_critique_needed=false and provide detailed feedback."""
        
        try:
            result = await critique_agent.run(prompt)
            critique = result.output
            
            # Ensure consistency between fields
            if critique.no_critique_needed:
                critique.level = CritiqueLevel.PERFECT
                critique.critiques = []
            elif not critique.level:
                # Infer level from critique content
                if len(critique.critiques) == 0:
                    critique.level = CritiqueLevel.PERFECT
                elif len(critique.critiques) <= 2:
                    critique.level = CritiqueLevel.MINOR_ISSUES
                else:
                    critique.level = CritiqueLevel.MAJOR_ISSUES
                    
            return ((critic_idx, solution_idx), critique)
        except Exception as e:
            print(f"Error in critique from model {critic_idx} for solution {solution_idx}: {e}")
            return ((critic_idx, solution_idx), Critique(
                no_critique_needed=False,
                critiques=[f"Error during critique: {str(e)}"],
                level=CritiqueLevel.MAJOR_ISSUES,
                confidence=0.5
            ))
    
    def _check_graduated_consensus(self) -> Dict[str, any]:
        """Check for graduated consensus based on threshold"""
        best_solution = None
        best_approval = 0.0
        
        for solution_idx, solution in self.solutions.items():
            approvals = 0
            confidence_sum = 0
            total_critics = 0
            
            for critic_idx in range(len(self.agents)):
                if critic_idx == solution_idx:
                    continue
                    
                critique = self.critiques.get((critic_idx, solution_idx))
                if critique:
                    total_critics += 1
                    if critique.is_acceptable:
                        approvals += critique.confidence
                        confidence_sum += critique.confidence
            
            if total_critics > 0:
                approval_rate = approvals / total_critics
                if approval_rate > best_approval:
                    best_approval = approval_rate
                    best_solution = solution
                
                if approval_rate >= self.request.consensus_threshold:
                    return {"solution": solution, "level": approval_rate}
        
        return {"solution": None, "level": best_approval}
    
    def _should_stop_early(self) -> bool:
        """Check if we should stop early due to minor issues only"""
        all_minor = True
        for critique in self.critiques.values():
            if critique.level and critique.level not in [CritiqueLevel.PERFECT, CritiqueLevel.MINOR_ISSUES]:
                all_minor = False
                break
        
        return all_minor and self.iteration > 1
    
    def _calculate_current_consensus_level(self) -> float:
        """Calculate the current consensus level across all solutions"""
        consensus_data = self._check_graduated_consensus()
        return consensus_data["level"]
    
    async def _phase_4_improvement_enhanced(self) -> bool:
        """Enhanced improvement with history and aggregated critiques"""
        tasks = []
        old_solutions = {}
        
        for i, agent in enumerate(self.agents):
            current_solution = self.solutions[i]
            old_solutions[i] = current_solution.content
            
            # Aggregate critiques
            aggregated = self._aggregate_critiques(i)
            
            # Build context based on configuration
            context_parts = [f"Problem: {self.request.problem}"]
            context_parts.append(f"\nYour current solution:\n{current_solution.content}")
            
            if self.request.enable_solution_history and len(self.previous_solutions[i]) > 1:
                context_parts.append(f"\nYour previous solution:\n{self.previous_solutions[i][-2]}")
            
            if aggregated["consensus_issues"]:
                context_parts.append(f"\nIssues multiple reviewers identified:\n" + 
                                   "\n".join(f"- {issue}" for issue in aggregated["consensus_issues"]))
            
            if aggregated["unique_insights"]:
                context_parts.append(f"\nAdditional feedback:\n" +
                                   "\n".join(f"- {insight}" for insight in aggregated["unique_insights"]))
            
            if aggregated["suggestions"]:
                context_parts.append(f"\nSuggestions for improvement:\n" +
                                   "\n".join(f"- {suggestion}" for suggestion in aggregated["suggestions"]))
            
            if aggregated["priority_level"]:
                context_parts.append(f"\nPriority: Focus on {aggregated['priority_level']} first")
            
            prompt = "\n".join(context_parts) + "\n\nProvide an improved solution addressing the feedback."
            
            tasks.append(self._generate_solution(agent, i, prompt))
        
        results = await asyncio.gather(*tasks)
        
        total_improvement = 0.0
        for i, solution_text in enumerate(results):
            delta = self._calculate_improvement_delta(old_solutions[i], solution_text)
            total_improvement += delta
            
            self.solutions[i] = Solution(
                model_index=i,
                content=solution_text,
                iteration=self.iteration,
                previous_content=old_solutions[i],
                improvement_delta=delta,
                critique_summary=self._aggregate_critiques(i)
            )
            self.previous_solutions[i].append(solution_text)
            
            # Track performance
            self.model_performance[i].improvement_rates.append(delta)
        
        avg_improvement = total_improvement / len(self.agents) if self.agents else 0
        self.improvement_trajectory.append(avg_improvement)
        
        # Return whether significant improvement was made
        return avg_improvement >= self.request.min_improvement_threshold
    
    async def _phase_5_voting_enhanced(self) -> Solution:
        """Enhanced voting with optional weighting"""
        if len(self.agents) == 1:
            return self.solutions[0]
        
        # Calculate vote weights if enabled
        if self.request.weighted_voting:
            self._calculate_vote_weights()
        
        votes = {}
        tasks = []
        
        for voter_idx, voter_agent in enumerate(self.agents):
            tasks.append(self._cast_vote_enhanced(voter_agent, voter_idx))
        
        vote_results = await asyncio.gather(*tasks)
        
        for voter_idx, (voted_idx, confidence) in enumerate(vote_results):
            if voted_idx is not None:
                weight = self.model_performance[voter_idx].vote_weight
                if self.request.weighted_voting:
                    votes[voted_idx] = votes.get(voted_idx, 0) + (weight * confidence)
                else:
                    votes[voted_idx] = votes.get(voted_idx, 0) + 1
        
        if not votes:
            return random.choice(list(self.solutions.values()))
        
        max_votes = max(votes.values())
        winners = [idx for idx, count in votes.items() if count == max_votes]
        
        if len(winners) == 1:
            return self.solutions[winners[0]]
        else:
            # Tie-breaking: choose the one with best critique consensus
            best_idx = winners[0]
            best_consensus = 0
            for idx in winners:
                consensus = sum(1 for c in self.critiques.values() if c.is_acceptable)
                if consensus > best_consensus:
                    best_consensus = consensus
                    best_idx = idx
            return self.solutions[best_idx]
    
    def _calculate_vote_weights(self):
        """Calculate voting weights based on model performance"""
        for idx, perf in self.model_performance.items():
            # Weight based on how often their solutions were approved
            approval_weight = (perf.solutions_approved + 1) / (self.iteration + 1)
            
            # Weight based on improvement rates
            if perf.improvement_rates:
                improvement_weight = sum(perf.improvement_rates) / len(perf.improvement_rates)
            else:
                improvement_weight = 0.5
            
            perf.vote_weight = (approval_weight + improvement_weight) / 2
    
    async def _cast_vote_enhanced(self, agent: Agent, voter_idx: int) -> Tuple[Optional[int], float]:
        """Enhanced voting with confidence scores"""
        solutions_text = []
        valid_indices = []
        
        for idx, solution in self.solutions.items():
            if idx != voter_idx:
                critique_summary = self._aggregate_critiques(idx)
                info = f"\nConsensus level: {critique_summary.get('average_confidence', 0):.2f}"
                if critique_summary.get('priority_level'):
                    info += f"\nSeverity: {critique_summary['priority_level']}"
                    
                solutions_text.append(f"Solution {idx}:{info}\n{solution.content}")
                valid_indices.append(idx)
        
        if not valid_indices:
            return (None, 0.0)
        
        prompt = f"""Problem: {self.request.problem}

Review these solutions and select the best one. Reply with:
1. The solution number (just the number)
2. Your confidence (0.0 to 1.0)

{chr(10).join(solutions_text)}

Format: [number] [confidence]
Example: 2 0.85"""
        
        try:
            result = await agent.run(prompt)
            response = result.output.strip()
            
            # Parse response for number and confidence
            parts = response.split()
            if parts:
                for idx in valid_indices:
                    if str(idx) in parts[0]:
                        confidence = 1.0
                        if len(parts) > 1:
                            try:
                                confidence = float(parts[1])
                            except:
                                confidence = 1.0
                        return (idx, min(max(confidence, 0.0), 1.0))
            
            return (random.choice(valid_indices), 0.5)
        except Exception as e:
            print(f"Error in voting from model {voter_idx}: {e}")
            return (random.choice(valid_indices) if valid_indices else None, 0.5)


async def run_consilium(request: ConsiliumRequest) -> ConsiliumResult:
    """Main entry point - uses enhanced orchestrator"""
    orchestrator = EnhancedConsiliumOrchestrator(request)
    return await orchestrator.run()