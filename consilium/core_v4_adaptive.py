import asyncio
import random
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
import difflib
import statistics

from .models import (
    ConsiliumRequest, ConsiliumResult, Critique, Solution, 
    CritiqueLevel, ModelPerformance
)

# Load environment variables
env_path = Path(__file__).parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv('.env.local')


class AdaptiveConsiliumOrchestrator:
    """
    Adaptive orchestrator that combines the speed of Enhanced mode 
    with the quality potential of Strict mode
    """
    
    def __init__(self, request: ConsiliumRequest):
        self.request = request
        self.agents: List[Agent] = []
        self.solutions: Dict[int, Solution] = {}
        self.critiques: Dict[Tuple[int, int], Critique] = {}
        self.iteration = 0
        self.model_performance: Dict[int, ModelPerformance] = {}
        self.improvement_trajectory: List[float] = []
        self.previous_solutions: Dict[int, List[str]] = {}
        self.critique_history: Dict[int, List[Dict]] = {}
        
        # Adaptive tracking
        self.quality_signals: List[float] = []  # Track quality indicators
        self.complexity_score: float = 0.0  # Problem complexity estimate
        self.convergence_rate: float = 0.0  # How fast models are converging
        self.mode: str = "adaptive"  # Current operating mode
        self.quality_threshold_met: bool = False
        
        self._initialize_agents()
        self._initialize_performance_tracking()
        self._estimate_problem_complexity()
    
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
    
    def _estimate_problem_complexity(self):
        """Estimate problem complexity to adapt strategy"""
        problem_lower = self.request.problem.lower()
        
        # Complexity indicators
        complexity_keywords = {
            'high': ['implement', 'design', 'optimize', 'efficient', 'complex', 'advanced', 'production', 'scalable', 'concurrent', 'cache', 'algorithm'],
            'medium': ['handle', 'validate', 'process', 'error', 'edge case', 'robust', 'complete'],
            'low': ['simple', 'basic', 'check', 'return', 'convert', 'find']
        }
        
        score = 0.5  # Base complexity
        
        # Check for complexity indicators
        for level, keywords in complexity_keywords.items():
            matches = sum(1 for kw in keywords if kw in problem_lower)
            if level == 'high':
                score += matches * 0.15
            elif level == 'medium':
                score += matches * 0.08
            else:
                score -= matches * 0.05
        
        # Problem length as complexity indicator
        if len(self.request.problem) > 200:
            score += 0.2
        elif len(self.request.problem) < 50:
            score -= 0.1
        
        self.complexity_score = min(max(score, 0.0), 1.0)
        
        # Adjust initial strategy based on complexity
        if self.complexity_score > 0.7:
            self.mode = "quality_focus"
        elif self.complexity_score < 0.3:
            self.mode = "speed_focus"
        else:
            self.mode = "balanced"
    
    def _calculate_improvement_delta(self, old_content: str, new_content: str) -> float:
        """Calculate solution improvement"""
        if not old_content or not new_content:
            return 0.0
        ratio = difflib.SequenceMatcher(None, old_content, new_content).ratio()
        return 1.0 - ratio
    
    def _calculate_solution_diversity(self) -> float:
        """Calculate diversity among solutions"""
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
    
    def _assess_quality_signals(self) -> float:
        """Assess current solution quality from multiple signals"""
        signals = []
        
        # Signal 1: Critique severity distribution
        severity_score = 0
        total_critiques = 0
        for critique in self.critiques.values():
            if critique.level:
                total_critiques += 1
                if critique.level == CritiqueLevel.PERFECT:
                    severity_score += 1.0
                elif critique.level == CritiqueLevel.MINOR_ISSUES:
                    severity_score += 0.7
                elif critique.level == CritiqueLevel.MAJOR_ISSUES:
                    severity_score += 0.3
                else:  # FUNDAMENTAL_FLAWS
                    severity_score += 0.0
        
        if total_critiques > 0:
            signals.append(severity_score / total_critiques)
        
        # Signal 2: Consensus level
        consensus_level = self._calculate_current_consensus_level()
        signals.append(consensus_level)
        
        # Signal 3: Solution diversity (higher is better early, lower is better late)
        diversity = self._calculate_solution_diversity()
        if self.iteration <= 1:
            signals.append(diversity)  # Want diversity early
        else:
            signals.append(1.0 - diversity * 0.5)  # Want convergence later
        
        # Signal 4: Improvement rate
        if self.improvement_trajectory:
            recent_improvement = statistics.mean(self.improvement_trajectory[-2:]) if len(self.improvement_trajectory) >= 2 else self.improvement_trajectory[-1]
            signals.append(1.0 - recent_improvement)  # Less improvement = higher quality
        
        # Weighted average of signals
        quality_score = statistics.mean(signals) if signals else 0.5
        self.quality_signals.append(quality_score)
        
        return quality_score
    
    def _adapt_strategy(self) -> Dict[str, any]:
        """Dynamically adapt strategy based on current state"""
        quality_score = self._assess_quality_signals()
        
        # Determine if we need more iterations
        force_continue = False
        adjusted_threshold = self.request.consensus_threshold
        
        # Complex problems need more scrutiny
        if self.complexity_score > 0.6:
            if self.iteration <= 1 and quality_score < 0.8:
                force_continue = True
                adjusted_threshold = min(0.95, self.request.consensus_threshold + 0.1)
            elif quality_score < 0.6:
                force_continue = True
        
        # Quality not meeting expectations
        if quality_score < 0.7 and self.iteration < self.request.max_iterations - 1:
            force_continue = True
            self.mode = "quality_boost"
        
        # Convergence too fast on complex problem
        if self.complexity_score > 0.5 and self.iteration == 1 and quality_score > 0.9:
            # Might be premature consensus
            force_continue = True
            adjusted_threshold = 0.95
        
        # Track if quality threshold is met
        if quality_score >= 0.85 or (quality_score >= 0.75 and self.complexity_score < 0.4):
            self.quality_threshold_met = True
        
        return {
            "force_continue": force_continue,
            "adjusted_threshold": adjusted_threshold,
            "quality_score": quality_score,
            "require_perfect": quality_score < 0.5 and self.iteration <= 2,
            "encourage_diversity": self.iteration <= 1 and self._calculate_solution_diversity() < 0.3
        }
    
    async def run(self) -> ConsiliumResult:
        await self._phase_1_initial_generation()
        
        termination_reason = "max_iterations"
        
        for iteration in range(1, self.request.max_iterations + 1):
            self.iteration = iteration
            
            # Adaptive critique phase
            await self._phase_2_critique_adaptive()
            
            # Adapt strategy based on current state
            strategy = self._adapt_strategy()
            
            # Check consensus with adaptive threshold
            if not strategy["force_continue"]:
                consensus_data = self._check_adaptive_consensus(strategy["adjusted_threshold"])
                if consensus_data["solution"]:
                    # Additional quality gate
                    if self.quality_threshold_met or iteration >= 2:
                        return ConsiliumResult(
                            final_solution=consensus_data["solution"].content,
                            iterations_used=iteration,
                            consensus_reached=True,
                            consensus_level=consensus_data["level"],
                            winning_model_index=consensus_data["solution"].model_index,
                            all_solutions=list(self.solutions.values()),
                            model_performance=list(self.model_performance.values()),
                            improvement_trajectory=self.improvement_trajectory,
                            termination_reason=f"adaptive_consensus_q{self._assess_quality_signals():.2f}"
                        )
            
            # Check for plateau but only after ensuring quality
            if iteration > 1 and self.request.early_stop_on_plateau:
                if self.quality_threshold_met and len(self.improvement_trajectory) >= 2:
                    if statistics.mean(self.improvement_trajectory[-2:]) < 0.05:
                        termination_reason = "quality_plateau"
                        break
            
            # Improvement phase
            if iteration < self.request.max_iterations:
                if strategy["encourage_diversity"]:
                    await self._phase_4_improvement_diverse()
                else:
                    improvement_made = await self._phase_4_improvement_adaptive(strategy)
                    
                    if not improvement_made and self.quality_threshold_met:
                        termination_reason = "adaptive_completion"
                        break
        
        winning_solution = await self._phase_5_voting_adaptive()
        
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
        """Initial generation with complexity-aware prompting"""
        tasks = []
        for i, agent in enumerate(self.agents):
            # Add quality hints for complex problems
            quality_hint = ""
            if self.complexity_score > 0.6:
                quality_hint = "\nThis is a complex problem. Ensure your solution is thorough, handles edge cases, and follows best practices."
            elif self.complexity_score < 0.3:
                quality_hint = "\nProvide a clear, straightforward solution."
            
            prompt = f"{self.request.initial_contexts[i]}{quality_hint}\n\nProblem: {self.request.problem}"
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
    
    async def _phase_2_critique_adaptive(self):
        """Adaptive critique phase with dynamic standards"""
        self.critiques.clear()
        tasks = []
        
        # Determine critique strictness based on mode and iteration
        if self.mode == "quality_focus" or self.mode == "quality_boost":
            strictness = "very strict"
        elif self.iteration <= 1 and self.complexity_score > 0.5:
            strictness = "strict"
        else:
            strictness = "balanced"
        
        for critic_idx, critic_agent in enumerate(self.agents):
            for solution_idx, solution in self.solutions.items():
                tasks.append(self._critique_solution_adaptive(
                    critic_agent, critic_idx, solution_idx, solution, strictness
                ))
        
        critiques = await asyncio.gather(*tasks)
        
        for (critic_idx, solution_idx), critique in critiques:
            self.critiques[(critic_idx, solution_idx)] = critique
            
            # Track performance
            self.model_performance[critic_idx].critiques_given += 1
            if critique.is_acceptable:
                self.model_performance[solution_idx].solutions_approved += 1
            
            # Track critique patterns
            self.critique_history[solution_idx].append({
                "iteration": self.iteration,
                "level": critique.level,
                "confidence": critique.confidence
            })
    
    async def _critique_solution_adaptive(self, agent: Agent, critic_idx: int, solution_idx: int, 
                                         solution: Solution, strictness: str) -> Tuple[Tuple[int, int], Critique]:
        """Adaptive critique with dynamic standards"""
        
        critique_agent = Agent(
            model=agent.model,
            output_type=Critique,
            system_prompt=f"""You are a {strictness} code reviewer.
            
Evaluate solutions based on the problem complexity and requirements.
{'Be especially thorough - this appears to be a complex problem requiring high quality.' if self.complexity_score > 0.6 else ''}

Severity Levels:
- PERFECT: Truly excellent - correct, efficient, complete, well-documented
- MINOR_ISSUES: Small improvements possible
- MAJOR_ISSUES: Significant problems that should be fixed
- FUNDAMENTAL_FLAWS: Critical errors or missing requirements

Provide specific, actionable feedback."""
        )
        
        prompt = f"""Problem: {self.request.problem}

Solution to critique:
{solution.content}

{'Note: This is iteration ' + str(self.iteration) + '. Previous critiques should be addressed.' if self.iteration > 1 else ''}

Evaluate thoroughly. Set appropriate severity level and confidence (0.0-1.0)."""
        
        try:
            result = await critique_agent.run(prompt)
            critique = result.output
            
            # Ensure consistency
            if critique.no_critique_needed:
                critique.level = CritiqueLevel.PERFECT
            elif not critique.level:
                if len(critique.critiques) == 0:
                    critique.level = CritiqueLevel.PERFECT
                elif len(critique.critiques) <= 2:
                    critique.level = CritiqueLevel.MINOR_ISSUES
                else:
                    critique.level = CritiqueLevel.MAJOR_ISSUES
            
            return ((critic_idx, solution_idx), critique)
        except Exception as e:
            return ((critic_idx, solution_idx), Critique(
                no_critique_needed=False,
                critiques=[f"Error: {str(e)}"],
                level=CritiqueLevel.MAJOR_ISSUES,
                confidence=0.5
            ))
    
    def _check_adaptive_consensus(self, threshold: float) -> Dict[str, any]:
        """Check consensus with adaptive threshold"""
        best_solution = None
        best_approval = 0.0
        
        for solution_idx, solution in self.solutions.items():
            approvals = 0
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
                        approvals += critique.confidence * 0.75
            
            if total_critics > 0:
                approval_rate = approvals / total_critics
                
                # Bonus for perfect ratings on complex problems
                if self.complexity_score > 0.6 and perfect_count > 0:
                    approval_rate += 0.05 * perfect_count
                
                if approval_rate > best_approval:
                    best_approval = approval_rate
                    best_solution = solution
                
                if approval_rate >= threshold:
                    return {"solution": solution, "level": approval_rate}
        
        return {"solution": None, "level": best_approval}
    
    def _calculate_current_consensus_level(self) -> float:
        """Calculate current consensus level"""
        consensus_data = self._check_adaptive_consensus(self.request.consensus_threshold)
        return consensus_data["level"]
    
    async def _phase_4_improvement_adaptive(self, strategy: Dict) -> bool:
        """Adaptive improvement phase"""
        tasks = []
        old_solutions = {}
        
        for i, agent in enumerate(self.agents):
            current_solution = self.solutions[i]
            old_solutions[i] = current_solution.content
            
            # Aggregate critiques
            aggregated = self._aggregate_critiques(i)
            
            # Build improvement prompt based on strategy
            emphasis = ""
            if strategy.get("require_perfect"):
                emphasis = "\n🔴 CRITICAL: Major issues identified. Significant improvements required."
            elif self.mode == "quality_boost":
                emphasis = "\n⚠️ Quality standards not met. Please enhance your solution significantly."
            
            prompt = f"""Problem: {self.request.problem}

Your current solution:
{current_solution.content}

Feedback from reviewers:
{self._format_aggregated_critiques(aggregated)}
{emphasis}

Provide an improved solution that thoroughly addresses all feedback."""
            
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
            self.model_performance[i].improvement_rates.append(delta)
        
        avg_improvement = total_improvement / len(self.agents) if self.agents else 0
        self.improvement_trajectory.append(avg_improvement)
        
        return avg_improvement >= self.request.min_improvement_threshold * 0.5  # More lenient
    
    async def _phase_4_improvement_diverse(self):
        """Force diverse solutions when needed"""
        tasks = []
        
        for i, agent in enumerate(self.agents):
            prompt = f"""Problem: {self.request.problem}

Current solutions are too similar. Provide a DIFFERENT approach that:
1. Solves the problem correctly
2. Uses a distinct algorithm or structure
3. Showcases your expertise: {self.request.initial_contexts[i] if i < len(self.request.initial_contexts) else 'your perspective'}

Be creative while maintaining quality."""
            
            tasks.append(self._generate_solution(agent, i, prompt))
        
        results = await asyncio.gather(*tasks)
        
        for i, solution_text in enumerate(results):
            self.solutions[i] = Solution(
                model_index=i,
                content=solution_text,
                iteration=self.iteration,
                previous_content=self.solutions[i].content if i in self.solutions else None,
                improvement_delta=1.0  # Assume significant change
            )
            self.previous_solutions[i].append(solution_text)
    
    def _aggregate_critiques(self, model_idx: int) -> Dict:
        """Aggregate and prioritize critiques"""
        critiques_list = []
        for critic_idx in range(len(self.agents)):
            if critic_idx != model_idx:
                critique = self.critiques.get((critic_idx, model_idx))
                if critique:
                    critiques_list.append(critique)
        
        if not critiques_list:
            return {}
        
        # Categorize by severity
        issues_by_severity = {
            CritiqueLevel.FUNDAMENTAL_FLAWS: [],
            CritiqueLevel.MAJOR_ISSUES: [],
            CritiqueLevel.MINOR_ISSUES: [],
            CritiqueLevel.PERFECT: []
        }
        
        for critique in critiques_list:
            if critique.level:
                issues_by_severity[critique.level].extend(critique.critiques)
        
        return {
            "fundamental": issues_by_severity[CritiqueLevel.FUNDAMENTAL_FLAWS],
            "major": issues_by_severity[CritiqueLevel.MAJOR_ISSUES],
            "minor": issues_by_severity[CritiqueLevel.MINOR_ISSUES],
            "suggestions": [s for c in critiques_list for s in c.suggestions],
            "confidence": statistics.mean([c.confidence for c in critiques_list])
        }
    
    def _format_aggregated_critiques(self, aggregated: Dict) -> str:
        """Format aggregated critiques for prompt"""
        if not aggregated:
            return "No specific critiques provided."
        
        parts = []
        if aggregated.get("fundamental"):
            parts.append("🔴 CRITICAL ISSUES:\n" + "\n".join(f"- {i}" for i in aggregated["fundamental"]))
        if aggregated.get("major"):
            parts.append("⚠️ Major issues:\n" + "\n".join(f"- {i}" for i in aggregated["major"]))
        if aggregated.get("minor"):
            parts.append("💡 Minor improvements:\n" + "\n".join(f"- {i}" for i in aggregated["minor"][:3]))
        if aggregated.get("suggestions"):
            parts.append("Suggestions:\n" + "\n".join(f"- {s}" for s in aggregated["suggestions"][:3]))
        
        return "\n\n".join(parts)
    
    async def _phase_5_voting_adaptive(self) -> Solution:
        """Adaptive voting with quality weighting"""
        if len(self.agents) == 1:
            return self.solutions[0]
        
        votes = {}
        
        for voter_idx, voter_agent in enumerate(self.agents):
            voted_idx, confidence = await self._cast_vote_adaptive(voter_agent, voter_idx)
            if voted_idx is not None:
                # Weight by confidence and performance
                weight = confidence
                if self.request.weighted_voting:
                    weight *= self.model_performance[voter_idx].vote_weight
                
                # Quality bonus for solutions with better critique scores
                aggregated = self._aggregate_critiques(voted_idx)
                if aggregated.get("confidence", 0) > 0.8:
                    weight *= 1.2
                
                votes[voted_idx] = votes.get(voted_idx, 0) + weight
        
        if not votes:
            return random.choice(list(self.solutions.values()))
        
        winner_idx = max(votes.items(), key=lambda x: x[1])[0]
        return self.solutions[winner_idx]
    
    async def _cast_vote_adaptive(self, agent: Agent, voter_idx: int) -> Tuple[Optional[int], float]:
        """Cast vote with quality awareness"""
        solutions_text = []
        valid_indices = []
        
        for idx, solution in self.solutions.items():
            if idx != voter_idx:
                aggregated = self._aggregate_critiques(idx)
                quality_indicator = "✓ High quality" if aggregated.get("confidence", 0) > 0.7 else "⚠️ Has issues"
                solutions_text.append(f"Solution {idx} ({quality_indicator}):\n{solution.content}")
                valid_indices.append(idx)
        
        if not valid_indices:
            return (None, 0.0)
        
        prompt = f"""Problem: {self.request.problem}

Select the BEST solution considering correctness, quality, and completeness:

{chr(10).join(solutions_text)}

Reply with: [number] [confidence 0.0-1.0]"""
        
        try:
            result = await agent.run(prompt)
            response = result.output.strip()
            parts = response.split()
            
            if parts:
                for idx in valid_indices:
                    if str(idx) in parts[0]:
                        confidence = float(parts[1]) if len(parts) > 1 else 0.7
                        return (idx, min(max(confidence, 0.0), 1.0))
            
            return (random.choice(valid_indices), 0.5)
        except:
            return (random.choice(valid_indices) if valid_indices else None, 0.5)


async def run_consilium_adaptive(request: ConsiliumRequest) -> ConsiliumResult:
    """Run with adaptive quality enhancement"""
    orchestrator = AdaptiveConsiliumOrchestrator(request)
    return await orchestrator.run()