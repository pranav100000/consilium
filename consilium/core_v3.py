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


class StrictConsiliumOrchestrator:
    """Enhanced orchestrator with stricter consensus requirements"""
    
    def __init__(self, request: ConsiliumRequest):
        self.request = request
        self.agents: List[Agent] = []
        self.solutions: Dict[int, Solution] = {}
        self.critiques: Dict[Tuple[int, int], Critique] = {}
        self.iteration = 0
        self.model_performance: Dict[int, ModelPerformance] = {}
        self.improvement_trajectory: List[float] = []
        self.previous_solutions: Dict[int, List[str]] = {}
        self.critique_history: Dict[int, List[Dict]] = {}  # Track critique patterns
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
            self.critique_history[i] = []
    
    def _initialize_performance_tracking(self):
        for i in range(len(self.agents)):
            self.model_performance[i] = ModelPerformance(model_index=i)
    
    def _calculate_improvement_delta(self, old_content: str, new_content: str) -> float:
        """Calculate how much a solution improved using diff ratio"""
        if not old_content or not new_content:
            return 0.0
        ratio = difflib.SequenceMatcher(None, old_content, new_content).ratio()
        return 1.0 - ratio
    
    def _calculate_solution_diversity(self) -> float:
        """Calculate how different the solutions are from each other"""
        if len(self.solutions) < 2:
            return 1.0
        
        total_diff = 0
        comparisons = 0
        for i, sol1 in self.solutions.items():
            for j, sol2 in self.solutions.items():
                if i < j:
                    ratio = difflib.SequenceMatcher(None, sol1.content, sol2.content).ratio()
                    total_diff += (1.0 - ratio)
                    comparisons += 1
        
        return total_diff / comparisons if comparisons > 0 else 0
    
    def _is_critique_substantial(self, critique: Critique) -> bool:
        """Check if a critique is substantial enough"""
        if critique.no_critique_needed:
            return True  # Approval is always substantial
        
        # Require at least 2 specific issues for non-approval
        if len(critique.critiques) < 2:
            return False
        
        # Check if critiques are meaningful (not just nitpicks)
        trivial_keywords = ['could', 'might', 'perhaps', 'slightly', 'minor', 'style', 'preference']
        substantial_count = 0
        for crit in critique.critiques:
            if not any(keyword in crit.lower() for keyword in trivial_keywords):
                substantial_count += 1
        
        return substantial_count >= 1
    
    def _adjust_consensus_threshold(self) -> float:
        """Dynamically adjust consensus threshold based on iteration"""
        base_threshold = self.request.consensus_threshold
        
        # Stricter in early iterations to encourage exploration
        if self.iteration <= 1:
            return min(1.0, base_threshold + 0.15)  # Require near-perfect consensus early
        elif self.iteration == 2:
            return base_threshold + 0.05
        else:
            return base_threshold  # Normal threshold in later iterations
    
    async def run(self) -> ConsiliumResult:
        await self._phase_1_initial_generation()
        
        consensus_solution = None
        termination_reason = "max_iterations"
        forced_iterations = min(2, self.request.max_iterations)  # Force at least 2 iterations
        
        for iteration in range(1, self.request.max_iterations + 1):
            self.iteration = iteration
            
            await self._phase_2_critique_strict()
            
            # Don't check consensus in first iteration - force exploration
            if iteration > forced_iterations:
                consensus_data = self._check_strict_consensus()
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
                        termination_reason="strict_consensus"
                    )
            
            # Check solution diversity - if too similar, push for more variation
            diversity = self._calculate_solution_diversity()
            if diversity < 0.3 and iteration < self.request.max_iterations:
                print(f"Low diversity ({diversity:.2f}) - encouraging variation")
                await self._phase_4_improvement_with_diversity()
            elif iteration < self.request.max_iterations:
                improvement_made = await self._phase_4_improvement_strict()
                
                # Only stop on plateau after forced iterations
                if iteration > forced_iterations and self.request.early_stop_on_plateau and not improvement_made:
                    termination_reason = "plateau_after_exploration"
                    break
        
        winning_solution = await self._phase_5_voting_strict()
        
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
        """Initial generation with diversity encouragement"""
        tasks = []
        for i, agent in enumerate(self.agents):
            # Add diversity hints to initial prompts
            diversity_hint = ""
            if i == 0:
                diversity_hint = "\nFocus on your unique perspective and expertise."
            elif i == 1:
                diversity_hint = "\nProvide a solution that emphasizes your specific strengths."
            else:
                diversity_hint = "\nOffer a distinctive approach based on your expertise."
            
            prompt = f"{self.request.initial_contexts[i]}{diversity_hint}\n\nProblem: {self.request.problem}"
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
    
    async def _phase_2_critique_strict(self):
        """Stricter critique phase with quality requirements"""
        self.critiques.clear()
        tasks = []
        
        for critic_idx, critic_agent in enumerate(self.agents):
            for solution_idx, solution in self.solutions.items():
                tasks.append(self._critique_solution_strict(critic_agent, critic_idx, solution_idx, solution))
        
        critiques = await asyncio.gather(*tasks)
        
        # Validate critique quality
        for (critic_idx, solution_idx), critique in critiques:
            # Ensure critique is substantial
            if not self._is_critique_substantial(critique):
                # Force more detailed critique
                critique.critiques.append("Insufficient analysis depth - requires more specific feedback")
                critique.level = CritiqueLevel.MAJOR_ISSUES
                critique.confidence = max(0.5, critique.confidence - 0.2)
            
            self.critiques[(critic_idx, solution_idx)] = critique
            
            # Track critique patterns
            self.critique_history[solution_idx].append({
                "iteration": self.iteration,
                "critic": critic_idx,
                "level": critique.level,
                "issue_count": len(critique.critiques)
            })
            
            # Update performance metrics
            self.model_performance[critic_idx].critiques_given += 1
            if critique.is_acceptable:
                self.model_performance[solution_idx].solutions_approved += 1
    
    async def _critique_solution_strict(self, agent: Agent, critic_idx: int, solution_idx: int, solution: Solution) -> Tuple[Tuple[int, int], Critique]:
        """Stricter critique with higher standards"""
        
        # Adjust prompt based on iteration to be more critical early on
        strictness_level = "very critical" if self.iteration <= 2 else "thorough"
        
        critique_agent = Agent(
            model=agent.model,
            output_type=Critique,
            system_prompt=f"""You are a {strictness_level} code reviewer with high standards.
            
BE STRICT: Only mark no_critique_needed=true if the solution is truly exceptional.

Severity Levels (use these strictly):
- PERFECT: Truly flawless - elegant, efficient, complete, well-documented
- MINOR_ISSUES: Small improvements (naming, minor optimizations)  
- MAJOR_ISSUES: Significant problems (missing features, poor efficiency, inadequate error handling)
- FUNDAMENTAL_FLAWS: Critical errors, wrong approach, or major requirements missing

You MUST:
1. Find at least 2-3 specific issues unless the code is truly perfect
2. Be concrete and specific - no vague feedback
3. Consider edge cases that might not be handled
4. Check for efficiency improvements
5. Verify all requirements are fully met
6. Set confidence based on how thoroughly you analyzed it (0.0-1.0)"""
        )
        
        # Include comparison context if available
        comparison_context = ""
        if self.iteration > 1 and len(self.solutions) > 1:
            other_solutions = [s for idx, s in self.solutions.items() if idx != solution_idx]
            if other_solutions:
                comparison_context = f"\n\nNote: There are {len(other_solutions)} other solutions. Consider if this one offers unique value."
        
        # Include history of critiques for this solution
        history_context = ""
        if self.critique_history.get(solution_idx):
            prev_issues = self.critique_history[solution_idx][-1]["issue_count"] if self.critique_history[solution_idx] else 0
            if prev_issues > 0:
                history_context = f"\n\nThis solution previously had {prev_issues} issues identified. Verify they are addressed."
        
        prompt = f"""Problem: {self.request.problem}

Solution to critique:
{solution.content}
{comparison_context}
{history_context}

Provide a STRICT critique. Remember:
- Set no_critique_needed=true ONLY if truly exceptional
- Find specific, concrete issues
- Consider ALL aspects: correctness, efficiency, completeness, style, documentation
- Be especially critical in early iterations to push for excellence"""
        
        try:
            result = await critique_agent.run(prompt)
            critique = result.output
            
            # Enforce minimum critique standards
            if critique.no_critique_needed and self.iteration <= 1:
                # In first iteration, force finding at least something
                critique.no_critique_needed = False
                critique.level = CritiqueLevel.MINOR_ISSUES
                critique.critiques.append("Further refinement possible in early iteration")
                critique.confidence = min(0.7, critique.confidence)
            
            # Ensure consistency
            if critique.no_critique_needed:
                critique.level = CritiqueLevel.PERFECT
                critique.critiques = []
            elif not critique.level:
                # Infer level from critique count and content
                if len(critique.critiques) == 0:
                    critique.level = CritiqueLevel.PERFECT
                elif len(critique.critiques) <= 1:
                    critique.level = CritiqueLevel.MINOR_ISSUES
                elif len(critique.critiques) <= 3:
                    critique.level = CritiqueLevel.MAJOR_ISSUES
                else:
                    critique.level = CritiqueLevel.FUNDAMENTAL_FLAWS
                    
            return ((critic_idx, solution_idx), critique)
        except Exception as e:
            print(f"Error in critique from model {critic_idx} for solution {solution_idx}: {e}")
            return ((critic_idx, solution_idx), Critique(
                no_critique_needed=False,
                critiques=[f"Error during critique: {str(e)}", "Unable to properly evaluate"],
                level=CritiqueLevel.MAJOR_ISSUES,
                confidence=0.3
            ))
    
    def _check_strict_consensus(self) -> Dict[str, any]:
        """Check for consensus with stricter requirements"""
        threshold = self._adjust_consensus_threshold()
        best_solution = None
        best_approval = 0.0
        
        for solution_idx, solution in self.solutions.items():
            approvals = 0
            confidence_sum = 0
            total_critics = 0
            perfect_count = 0
            
            for critic_idx in range(len(self.agents)):
                if critic_idx == solution_idx:
                    continue
                    
                critique = self.critiques.get((critic_idx, solution_idx))
                if critique:
                    total_critics += 1
                    if critique.level == CritiqueLevel.PERFECT:
                        perfect_count += 1
                        approvals += critique.confidence
                    elif critique.level == CritiqueLevel.MINOR_ISSUES:
                        approvals += critique.confidence * 0.7  # Partial credit for minor issues
                    confidence_sum += critique.confidence
            
            if total_critics > 0:
                # Require at least one PERFECT rating for consensus
                if perfect_count == 0 and self.iteration <= 2:
                    continue
                
                approval_rate = approvals / total_critics
                if approval_rate > best_approval:
                    best_approval = approval_rate
                    best_solution = solution
                
                if approval_rate >= threshold:
                    return {"solution": solution, "level": approval_rate}
        
        return {"solution": None, "level": best_approval}
    
    def _calculate_current_consensus_level(self) -> float:
        """Calculate the current consensus level across all solutions"""
        consensus_data = self._check_strict_consensus()
        return consensus_data["level"]
    
    async def _phase_4_improvement_strict(self) -> bool:
        """Improvement phase with focus on addressing critiques thoroughly"""
        tasks = []
        old_solutions = {}
        
        for i, agent in enumerate(self.agents):
            current_solution = self.solutions[i]
            old_solutions[i] = current_solution.content
            
            # Aggregate critiques with emphasis on major issues
            aggregated = self._aggregate_critiques_strict(i)
            
            # Build improvement prompt
            prompt = self._build_improvement_prompt(i, current_solution, aggregated)
            
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
                critique_summary=self._aggregate_critiques_strict(i)
            )
            self.previous_solutions[i].append(solution_text)
            
            # Track performance
            self.model_performance[i].improvement_rates.append(delta)
        
        avg_improvement = total_improvement / len(self.agents) if self.agents else 0
        self.improvement_trajectory.append(avg_improvement)
        
        return avg_improvement >= self.request.min_improvement_threshold
    
    async def _phase_4_improvement_with_diversity(self) -> bool:
        """Improvement phase that encourages diverse solutions"""
        tasks = []
        old_solutions = {}
        
        for i, agent in enumerate(self.agents):
            current_solution = self.solutions[i]
            old_solutions[i] = current_solution.content
            
            # Get other solutions for comparison
            other_solutions = [s.content for idx, s in self.solutions.items() if idx != i]
            
            prompt = f"""Problem: {self.request.problem}

Your current solution:
{current_solution.content}

Other models have provided similar solutions. Please provide a DISTINCTLY DIFFERENT approach that:
1. Solves the problem correctly
2. Uses a different algorithm, structure, or methodology
3. Showcases your unique expertise: {self.request.initial_contexts[i] if i < len(self.request.initial_contexts) else 'your perspective'}

Be creative and differentiate your solution while maintaining quality."""
            
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
                improvement_delta=delta
            )
            self.previous_solutions[i].append(solution_text)
            
            self.model_performance[i].improvement_rates.append(delta)
        
        avg_improvement = total_improvement / len(self.agents) if self.agents else 0
        self.improvement_trajectory.append(avg_improvement)
        
        return True  # Always continue after diversity push
    
    def _aggregate_critiques_strict(self, model_idx: int) -> Dict[str, any]:
        """Aggregate critiques with focus on substantial issues"""
        critiques_for_model = []
        confidence_sum = 0
        count = 0
        severity_counts = {level: 0 for level in CritiqueLevel}
        
        for critic_idx in range(len(self.agents)):
            if critic_idx != model_idx:
                critique = self.critiques.get((critic_idx, model_idx))
                if critique:
                    critiques_for_model.append(critique)
                    confidence_sum += critique.confidence
                    count += 1
                    if critique.level:
                        severity_counts[critique.level] += 1
        
        if not critiques_for_model:
            return {"consensus_issues": [], "unique_insights": [], "priority_fixes": []}
        
        # Find consensus issues
        issue_counts = {}
        all_suggestions = []
        
        for critique in critiques_for_model:
            for issue in critique.critiques:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            all_suggestions.extend(critique.suggestions)
        
        consensus_issues = [issue for issue, count in issue_counts.items() if count > 1]
        unique_insights = [issue for issue, count in issue_counts.items() if count == 1]
        
        # Prioritize by severity
        priority_fixes = []
        for level in [CritiqueLevel.FUNDAMENTAL_FLAWS, CritiqueLevel.MAJOR_ISSUES]:
            priority_fixes.extend([c.critiques for c in critiques_for_model if c.level == level])
        
        return {
            "consensus_issues": consensus_issues[:5],
            "unique_insights": unique_insights[:3],
            "suggestions": list(set(all_suggestions))[:5],
            "average_confidence": confidence_sum / count if count > 0 else 0,
            "severity_distribution": severity_counts,
            "total_critics": count
        }
    
    def _build_improvement_prompt(self, model_idx: int, current_solution: Solution, aggregated: Dict) -> str:
        """Build a detailed improvement prompt"""
        context_parts = [f"Problem: {self.request.problem}"]
        context_parts.append(f"\nYour current solution:\n{current_solution.content}")
        
        # Add severity distribution
        if aggregated.get("severity_distribution"):
            dist = aggregated["severity_distribution"]
            if dist.get(CritiqueLevel.FUNDAMENTAL_FLAWS, 0) > 0:
                context_parts.append(f"\n⚠️ {dist[CritiqueLevel.FUNDAMENTAL_FLAWS]} critics found FUNDAMENTAL FLAWS - major rework needed")
            elif dist.get(CritiqueLevel.MAJOR_ISSUES, 0) > 0:
                context_parts.append(f"\n⚠️ {dist[CritiqueLevel.MAJOR_ISSUES]} critics found MAJOR ISSUES - significant improvements required")
        
        if aggregated["consensus_issues"]:
            context_parts.append(f"\n🔴 Issues ALL reviewers identified (MUST fix):\n" + 
                               "\n".join(f"- {issue}" for issue in aggregated["consensus_issues"]))
        
        if aggregated["unique_insights"]:
            context_parts.append(f"\n🟡 Additional concerns to consider:\n" +
                               "\n".join(f"- {insight}" for insight in aggregated["unique_insights"]))
        
        if aggregated["suggestions"]:
            context_parts.append(f"\n💡 Suggestions for improvement:\n" +
                               "\n".join(f"- {suggestion}" for suggestion in aggregated["suggestions"]))
        
        context_parts.append(f"\n\nProvide an improved solution that thoroughly addresses ALL feedback, especially consensus issues.")
        
        return "\n".join(context_parts)
    
    async def _phase_5_voting_strict(self) -> Solution:
        """Strict voting phase with quality requirements"""
        if len(self.agents) == 1:
            return self.solutions[0]
        
        votes = {}
        tasks = []
        
        for voter_idx, voter_agent in enumerate(self.agents):
            tasks.append(self._cast_vote_strict(voter_agent, voter_idx))
        
        vote_results = await asyncio.gather(*tasks)
        
        for voter_idx, (voted_idx, confidence, justification) in enumerate(vote_results):
            if voted_idx is not None:
                # Weight by confidence and performance
                weight = confidence
                if self.request.weighted_voting:
                    weight *= self.model_performance[voter_idx].vote_weight
                votes[voted_idx] = votes.get(voted_idx, 0) + weight
        
        if not votes:
            return random.choice(list(self.solutions.values()))
        
        max_votes = max(votes.values())
        winners = [idx for idx, count in votes.items() if count == max_votes]
        
        if len(winners) == 1:
            return self.solutions[winners[0]]
        else:
            # Tie-breaking based on critique consensus
            best_idx = winners[0]
            best_score = 0
            for idx in winners:
                aggregated = self._aggregate_critiques_strict(idx)
                score = aggregated.get("average_confidence", 0) * (1 - len(aggregated.get("consensus_issues", [])) / 10)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return self.solutions[best_idx]
    
    async def _cast_vote_strict(self, agent: Agent, voter_idx: int) -> Tuple[Optional[int], float, str]:
        """Strict voting with justification requirement"""
        solutions_text = []
        valid_indices = []
        
        for idx, solution in self.solutions.items():
            if idx != voter_idx:
                critique_summary = self._aggregate_critiques_strict(idx)
                severity = "No major issues"
                if critique_summary.get("severity_distribution"):
                    dist = critique_summary["severity_distribution"]
                    if dist.get(CritiqueLevel.FUNDAMENTAL_FLAWS, 0) > 0:
                        severity = "Has fundamental flaws"
                    elif dist.get(CritiqueLevel.MAJOR_ISSUES, 0) > 0:
                        severity = "Has major issues"
                    elif dist.get(CritiqueLevel.MINOR_ISSUES, 0) > 0:
                        severity = "Has minor issues"
                
                solutions_text.append(f"Solution {idx} ({severity}):\n{solution.content}")
                valid_indices.append(idx)
        
        if not valid_indices:
            return (None, 0.0, "")
        
        prompt = f"""Problem: {self.request.problem}

Review these solutions and select the BEST one. Consider:
- Correctness and completeness
- Code quality and efficiency
- How well critiques were addressed
- Overall excellence

{chr(10).join(solutions_text)}

Reply with:
1. Solution number
2. Confidence (0.0-1.0)
3. Brief justification (one line)

Format: [number] [confidence] [justification]"""
        
        try:
            result = await agent.run(prompt)
            response = result.output.strip()
            
            # Parse response
            parts = response.split(None, 2)
            if parts:
                for idx in valid_indices:
                    if str(idx) in parts[0]:
                        confidence = 0.7
                        justification = ""
                        if len(parts) > 1:
                            try:
                                confidence = float(parts[1])
                            except:
                                pass
                        if len(parts) > 2:
                            justification = parts[2]
                        return (idx, min(max(confidence, 0.0), 1.0), justification)
            
            return (random.choice(valid_indices), 0.5, "Unable to parse vote")
        except Exception as e:
            print(f"Error in voting from model {voter_idx}: {e}")
            return (random.choice(valid_indices) if valid_indices else None, 0.5, str(e))


async def run_consilium_strict(request: ConsiliumRequest) -> ConsiliumResult:
    """Run with stricter consensus requirements"""
    orchestrator = StrictConsiliumOrchestrator(request)
    return await orchestrator.run()