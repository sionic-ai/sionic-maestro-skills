"""
Coordination Engine - Architecture Selection based on Task Structure

Implements collaboration rules from "Towards a Science of Scaling Agent Systems" (2.3):
- Rule A: Domain/task structure dependency is absolute
- Rule B: Decomposable → MAS, Sequential → SAS/minimal
- Rule C: Coordination overhead is a first-class cost function
- Rule D: Model family calibration is necessary

This engine decides WHEN to use multi-agent coordination and WHEN to fall back to
single-agent execution based on task characteristics and measured overhead.
"""

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque


# =============================================================================
# COORDINATION TOPOLOGY
# =============================================================================

class CoordinationTopology(Enum):
    """
    Coordination topologies from the paper.

    Each has different communication patterns and overhead characteristics.
    """
    # Single Agent System - No coordination overhead
    SAS = "sas"

    # Multi-Agent: Independent parallel execution + central synthesis
    # Best for: Decomposable tasks, candidate generation
    MAS_INDEPENDENT = "mas_independent"

    # Multi-Agent: Central orchestrator coordinates all agents
    # Best for: Tool-heavy tasks requiring supervision
    MAS_CENTRALIZED = "mas_centralized"

    # Multi-Agent: Hybrid with limited peer communication
    # Best for: Complex tasks needing both parallelism and coordination
    MAS_HYBRID = "mas_hybrid"

    # Multi-Agent: Decentralized peer-to-peer (DISABLED by default)
    # Paper shows this often increases overhead without benefit
    MAS_DECENTRALIZED = "mas_decentralized"


# =============================================================================
# TASK STRUCTURE FEATURES
# =============================================================================

@dataclass
class TaskStructureFeatures:
    """
    Features extracted from task analysis for architecture selection.

    These features are the INPUT to the architecture selection engine.
    """
    # How easily can the task be split into independent subtasks?
    # 0.0 = completely sequential, 1.0 = embarrassingly parallel
    decomposability_score: float = 0.5

    # How much does each step depend on previous results?
    # 0.0 = no dependency, 1.0 = strong sequential chain
    sequential_dependency_score: float = 0.5

    # Tool usage complexity (number of tools, failure risk)
    # 0.0 = no tools, 1.0 = many complex tools
    tool_complexity: float = 0.3

    # Estimated single-agent success rate (from calibration or heuristics)
    baseline_single_agent_success: float = 0.7

    # Risk level (production code, security implications)
    # 0.0 = low risk, 1.0 = critical production
    risk_level: float = 0.3

    # Budget remaining (normalized 0-1)
    budget_remaining: float = 1.0

    # Task domain hints
    domain: str = "general"

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationDecision:
    """
    Output of the architecture selection engine.
    """
    topology: CoordinationTopology
    confidence: float  # 0-1, how confident the selection is
    rationale: str

    # Recommended parameters for the selected topology
    max_agents: int = 3
    max_rounds: int = 3
    max_messages_per_agent: int = 2

    # Fallback topology if primary fails
    fallback_topology: Optional[CoordinationTopology] = None

    # Degradation threshold - if overhead exceeds this, fall back
    overhead_threshold: float = 2.0

    # Stage-specific recommendations
    stage_recommendations: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# COORDINATION METRICS (for Rule C - Overhead as first-class cost)
# =============================================================================

@dataclass
class CoordinationMetrics:
    """
    Metrics for measuring coordination overhead.

    Paper emphasizes these continuous metrics are more explanatory than
    discrete architecture labels.
    """
    # Message/round counts
    total_messages: int = 0
    total_rounds: int = 0
    synthesis_rounds: int = 0

    # Message density (messages per turn)
    message_density: float = 0.0

    # Redundancy rate (similarity of agent outputs)
    # High redundancy = wasted computation
    redundancy_rate: float = 0.0

    # Error amplification factor
    # > 1.0 means MAS is making more errors than SAS would
    error_amplification: float = 1.0

    # Efficiency (success per 1k tokens)
    success_per_1k_tokens: float = 0.0

    # Coordination overhead (extra work vs SAS)
    # (MAS_turns - SAS_turns) / SAS_turns
    coordination_overhead: float = 0.0

    # Time overhead
    wall_time_ms: float = 0.0

    # Token usage
    total_tokens: int = 0

    # Success/failure tracking
    successes: int = 0
    failures: int = 0

    def compute_overhead_ratio(self, baseline_tokens: int = 1000) -> float:
        """Compute overhead ratio vs baseline."""
        if baseline_tokens <= 0:
            return 0.0
        return self.total_tokens / baseline_tokens

    def should_degrade(self, threshold: float = 2.0) -> bool:
        """Check if we should degrade to simpler topology."""
        # Degrade if:
        # 1. Overhead is high and success rate is low
        # 2. Error amplification is > 1 (MAS making things worse)
        # 3. Redundancy is very high (wasted work)

        if self.error_amplification > 1.2:
            return True

        if self.coordination_overhead > threshold:
            success_rate = self.successes / max(self.successes + self.failures, 1)
            if success_rate < 0.5:
                return True

        if self.redundancy_rate > 0.8:
            return True

        return False


class MetricsTracker:
    """
    Tracks coordination metrics across runs for calibration.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)
        self._current: Optional[CoordinationMetrics] = None

        # Per-topology statistics
        self._topology_stats: Dict[CoordinationTopology, Dict[str, float]] = {}

    def start_run(self):
        """Start tracking a new run."""
        self._current = CoordinationMetrics()
        self._current.wall_time_ms = time.time() * 1000

    def record_message(self, tokens: int = 0):
        """Record a message sent."""
        if self._current:
            self._current.total_messages += 1
            self._current.total_tokens += tokens

    def record_round(self, is_synthesis: bool = False):
        """Record a coordination round."""
        if self._current:
            self._current.total_rounds += 1
            if is_synthesis:
                self._current.synthesis_rounds += 1

    def record_success(self):
        """Record a successful outcome."""
        if self._current:
            self._current.successes += 1

    def record_failure(self):
        """Record a failed outcome."""
        if self._current:
            self._current.failures += 1

    def record_redundancy(self, similarity_scores: List[float]):
        """Record output similarity for redundancy calculation."""
        if self._current and similarity_scores:
            self._current.redundancy_rate = sum(similarity_scores) / len(similarity_scores)

    def end_run(self, topology: CoordinationTopology) -> CoordinationMetrics:
        """End tracking and compute final metrics."""
        if not self._current:
            return CoordinationMetrics()

        metrics = self._current

        # Compute derived metrics
        metrics.wall_time_ms = time.time() * 1000 - metrics.wall_time_ms

        if metrics.total_rounds > 0:
            metrics.message_density = metrics.total_messages / metrics.total_rounds

        if metrics.total_tokens > 0:
            metrics.success_per_1k_tokens = (metrics.successes * 1000) / metrics.total_tokens

        # Store in history
        self._history.append((topology, metrics))

        # Update per-topology stats
        self._update_topology_stats(topology, metrics)

        self._current = None
        return metrics

    def _update_topology_stats(self, topology: CoordinationTopology, metrics: CoordinationMetrics):
        """Update rolling statistics for a topology."""
        if topology not in self._topology_stats:
            self._topology_stats[topology] = {
                "success_rate": 0.0,
                "avg_overhead": 0.0,
                "avg_tokens": 0.0,
                "count": 0,
            }

        stats = self._topology_stats[topology]
        n = stats["count"]

        # Incremental average update
        success_rate = metrics.successes / max(metrics.successes + metrics.failures, 1)
        stats["success_rate"] = (stats["success_rate"] * n + success_rate) / (n + 1)
        stats["avg_overhead"] = (stats["avg_overhead"] * n + metrics.coordination_overhead) / (n + 1)
        stats["avg_tokens"] = (stats["avg_tokens"] * n + metrics.total_tokens) / (n + 1)
        stats["count"] = n + 1

    def get_topology_stats(self, topology: CoordinationTopology) -> Dict[str, float]:
        """Get statistics for a topology."""
        return self._topology_stats.get(topology, {
            "success_rate": 0.5,
            "avg_overhead": 1.0,
            "avg_tokens": 1000,
            "count": 0,
        })

    def get_best_topology_for_features(
        self,
        features: TaskStructureFeatures,
    ) -> Optional[CoordinationTopology]:
        """
        Based on historical data, suggest the best topology for given features.

        Returns None if not enough data for recommendation.
        """
        # Need at least 10 samples for meaningful recommendation
        total_samples = sum(s["count"] for s in self._topology_stats.values())
        if total_samples < 10:
            return None

        # Simple heuristic: pick topology with best success rate
        # that has been used in similar conditions
        best_topology = None
        best_score = -1

        for topology, stats in self._topology_stats.items():
            if stats["count"] < 3:
                continue

            # Score = success_rate - overhead_penalty
            overhead_penalty = min(stats["avg_overhead"] * 0.1, 0.3)
            score = stats["success_rate"] - overhead_penalty

            if score > best_score:
                best_score = score
                best_topology = topology

        return best_topology


# =============================================================================
# ARCHITECTURE SELECTION ENGINE (Rules A-D)
# =============================================================================

class ArchitectureSelectionEngine:
    """
    Implements the collaboration rules from the paper.

    Rule A: Domain/task structure dependency is absolute
    Rule B: Decomposable → MAS, Sequential → SAS
    Rule C: Coordination overhead is first-class cost
    Rule D: Model family calibration is necessary
    """

    # Thresholds (configurable)
    SEQUENTIAL_THRESHOLD = 0.7  # Above this, use SAS
    DECOMPOSABLE_THRESHOLD = 0.6  # Above this, consider MAS
    TOOL_COMPLEXITY_HIGH = 0.6

    def __init__(
        self,
        metrics_tracker: Optional[MetricsTracker] = None,
        default_topology: CoordinationTopology = CoordinationTopology.SAS,
    ):
        self.metrics_tracker = metrics_tracker or MetricsTracker()
        self.default_topology = default_topology

        # Stage-specific defaults based on paper insights
        self.stage_defaults = {
            "analyze": CoordinationTopology.MAS_INDEPENDENT,  # Parallel info gathering
            "hypothesize": CoordinationTopology.MAS_INDEPENDENT,  # Parallel hypothesis gen
            "implement": CoordinationTopology.MAS_CENTRALIZED,  # Tool-heavy, need supervision
            "debug": CoordinationTopology.SAS,  # Sequential dependency high
            "improve": CoordinationTopology.MAS_INDEPENDENT,  # Parallel review/suggestions
        }

    def select_architecture(
        self,
        features: TaskStructureFeatures,
        stage: Optional[str] = None,
        force_topology: Optional[CoordinationTopology] = None,
    ) -> CoordinationDecision:
        """
        Select the best coordination topology based on task features.

        This is the main entry point implementing Rules A-D.
        """
        # Allow override
        if force_topology:
            return CoordinationDecision(
                topology=force_topology,
                confidence=1.0,
                rationale="Forced by user/policy",
            )

        # Rule D: Check if calibration data suggests a better choice
        calibrated = self.metrics_tracker.get_best_topology_for_features(features)
        if calibrated and features.budget_remaining > 0.3:
            stats = self.metrics_tracker.get_topology_stats(calibrated)
            if stats["success_rate"] > 0.7:
                return CoordinationDecision(
                    topology=calibrated,
                    confidence=min(stats["success_rate"], 0.9),
                    rationale=f"Calibrated from {stats['count']} historical runs",
                    fallback_topology=CoordinationTopology.SAS,
                )

        # Rule B: Sequential dependency is high → SAS
        if features.sequential_dependency_score > self.SEQUENTIAL_THRESHOLD:
            return CoordinationDecision(
                topology=CoordinationTopology.SAS,
                confidence=0.85,
                rationale=(
                    f"High sequential dependency ({features.sequential_dependency_score:.2f} > {self.SEQUENTIAL_THRESHOLD}). "
                    "Paper shows MAS degrades significantly in sequential tasks."
                ),
                max_agents=1,
                max_rounds=1,
            )

        # Rule B: Decomposable + tool complexity
        if features.decomposability_score > self.DECOMPOSABLE_THRESHOLD:
            if features.tool_complexity > self.TOOL_COMPLEXITY_HIGH:
                # High decomposability + high tool complexity → Centralized
                return CoordinationDecision(
                    topology=CoordinationTopology.MAS_CENTRALIZED,
                    confidence=0.75,
                    rationale=(
                        f"Decomposable ({features.decomposability_score:.2f}) with complex tools "
                        f"({features.tool_complexity:.2f}). Centralized coordination recommended."
                    ),
                    max_agents=3,
                    max_rounds=3,
                    max_messages_per_agent=3,
                    fallback_topology=CoordinationTopology.SAS,
                    overhead_threshold=2.5,
                )
            else:
                # High decomposability + low tool complexity → Independent
                return CoordinationDecision(
                    topology=CoordinationTopology.MAS_INDEPENDENT,
                    confidence=0.80,
                    rationale=(
                        f"Decomposable ({features.decomposability_score:.2f}) with low tool complexity. "
                        "Independent parallel generation + synthesis recommended."
                    ),
                    max_agents=3,
                    max_rounds=2,
                    max_messages_per_agent=1,
                    fallback_topology=CoordinationTopology.SAS,
                    overhead_threshold=2.0,
                )

        # Rule A: Use stage-specific defaults
        if stage and stage in self.stage_defaults:
            default = self.stage_defaults[stage]
            return CoordinationDecision(
                topology=default,
                confidence=0.65,
                rationale=f"Stage-specific default for '{stage}'",
                fallback_topology=CoordinationTopology.SAS,
            )

        # Conservative default: SAS
        return CoordinationDecision(
            topology=self.default_topology,
            confidence=0.5,
            rationale="No strong signal, using conservative SAS default",
        )

    def should_degrade(
        self,
        current_topology: CoordinationTopology,
        metrics: CoordinationMetrics,
        decision: CoordinationDecision,
    ) -> Tuple[bool, Optional[CoordinationTopology], str]:
        """
        Rule C: Check if we should degrade to a simpler topology.

        Returns:
            (should_degrade, new_topology, reason)
        """
        if current_topology == CoordinationTopology.SAS:
            return False, None, "Already at SAS"

        # Check overhead threshold
        if metrics.coordination_overhead > decision.overhead_threshold:
            return True, CoordinationTopology.SAS, (
                f"Coordination overhead ({metrics.coordination_overhead:.2f}) "
                f"exceeds threshold ({decision.overhead_threshold:.2f})"
            )

        # Check error amplification
        if metrics.error_amplification > 1.2:
            return True, CoordinationTopology.SAS, (
                f"Error amplification ({metrics.error_amplification:.2f}) > 1.2: "
                "MAS is making more errors than SAS would"
            )

        # Check redundancy (wasted computation)
        if metrics.redundancy_rate > 0.85:
            return True, decision.fallback_topology or CoordinationTopology.SAS, (
                f"High redundancy ({metrics.redundancy_rate:.2f}): "
                "Agents producing nearly identical outputs"
            )

        # Check success rate
        total = metrics.successes + metrics.failures
        if total >= 5:
            success_rate = metrics.successes / total
            if success_rate < 0.3 and metrics.coordination_overhead > 1.5:
                return True, CoordinationTopology.SAS, (
                    f"Low success rate ({success_rate:.2f}) with high overhead: "
                    "Degrading to SAS"
                )

        return False, None, "No degradation needed"

    def get_stage_recommendation(
        self,
        stage: str,
        features: Optional[TaskStructureFeatures] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed recommendations for a workflow stage.
        """
        features = features or TaskStructureFeatures()
        decision = self.select_architecture(features, stage)

        stage_configs = {
            "analyze": {
                "description": "Example Analysis - Parallel info gathering",
                "topology": decision.topology.value,
                "ensemble_strategy": "parallel_independent",
                "voting_mode": "score_based",  # Not test-based
                "max_candidates": 3,
                "red_flag_rules": ["too_long", "missing_fields"],
            },
            "hypothesize": {
                "description": "Hypothesis Generation - Parallel hypothesis gen + scoring",
                "topology": decision.topology.value,
                "ensemble_strategy": "parallel_independent",
                "voting_mode": "score_based",  # Falsifiability, explanation power
                "max_candidates": 5,
                "scoring_criteria": ["falsifiable", "low_experiment_cost", "explanation_coverage"],
                "red_flag_rules": ["hedging", "unfalsifiable"],
            },
            "implement": {
                "description": "Code Implementation - Parallel patch gen + test selection",
                "topology": decision.topology.value,
                "ensemble_strategy": "parallel_then_test",
                "voting_mode": "test_first",  # Tests are the judge
                "max_candidates": 3,
                "selection_priority": ["test_pass", "lint_score", "diff_size", "risk_score"],
                "red_flag_rules": ["invalid_diff", "multi_file", "dangerous_code"],
            },
            "debug": {
                "description": "Debug Loop - Sequential, minimize coordination",
                "topology": CoordinationTopology.SAS.value,  # Override to SAS
                "ensemble_strategy": "single_agent",
                "voting_mode": "test_first",
                "max_iterations": 5,
                "red_flag_rules": ["too_long", "self_contradiction", "no_progress"],
            },
            "improve": {
                "description": "Recursive Improvement - Parallel review + skill extraction",
                "topology": decision.topology.value,
                "ensemble_strategy": "parallel_independent",
                "voting_mode": "hybrid",
                "max_candidates": 3,
                "outputs": ["skill_template", "policy_update", "playbook_update"],
            },
        }

        return stage_configs.get(stage, {
            "description": "Unknown stage",
            "topology": CoordinationTopology.SAS.value,
        })


# =============================================================================
# TASK STRUCTURE CLASSIFIER
# =============================================================================

class TaskStructureClassifier:
    """
    Analyzes task description/context to extract structure features.

    This is used in Stage 1 (Analyze) to classify the task and inform
    architecture selection for subsequent stages.
    """

    # Keywords suggesting decomposability
    DECOMPOSABLE_KEYWORDS = [
        "multiple", "several", "each", "all", "parallel",
        "independent", "separate", "various", "different",
        "compare", "alternatives", "options", "candidates",
    ]

    # Keywords suggesting sequential dependency
    SEQUENTIAL_KEYWORDS = [
        "first", "then", "after", "before", "depends on",
        "requires", "must", "sequence", "order", "step by step",
        "chain", "pipeline", "workflow", "accumulate",
    ]

    # Keywords suggesting tool complexity
    TOOL_KEYWORDS = [
        "test", "run", "execute", "compile", "build",
        "deploy", "database", "api", "network", "file",
        "git", "docker", "kubernetes", "aws", "cloud",
    ]

    def classify(
        self,
        task_description: str,
        code_context: Optional[str] = None,
        error_logs: Optional[str] = None,
    ) -> TaskStructureFeatures:
        """
        Classify task structure from description and context.

        This is a heuristic classifier. For production, consider using
        an LLM to do more sophisticated classification.
        """
        text = f"{task_description} {code_context or ''} {error_logs or ''}".lower()

        # Count keyword occurrences
        decomp_count = sum(1 for kw in self.DECOMPOSABLE_KEYWORDS if kw in text)
        seq_count = sum(1 for kw in self.SEQUENTIAL_KEYWORDS if kw in text)
        tool_count = sum(1 for kw in self.TOOL_KEYWORDS if kw in text)

        # Normalize to 0-1 scores
        decomposability = min(decomp_count / 5, 1.0)
        sequential = min(seq_count / 5, 1.0)
        tool_complexity = min(tool_count / 5, 1.0)

        # Detect specific patterns
        is_debugging = any(kw in text for kw in ["debug", "fix", "error", "bug", "fail"])
        is_refactoring = any(kw in text for kw in ["refactor", "clean", "improve", "optimize"])
        is_feature = any(kw in text for kw in ["add", "implement", "create", "new feature"])

        # Adjust scores based on patterns
        if is_debugging:
            sequential = max(sequential, 0.7)  # Debugging is usually sequential
            tool_complexity = max(tool_complexity, 0.5)

        if is_refactoring:
            decomposability = max(decomposability, 0.5)  # Often can parallelize review

        if is_feature:
            decomposability = max(decomposability, 0.4)

        # Determine domain
        domain = "general"
        if is_debugging:
            domain = "debugging"
        elif is_refactoring:
            domain = "refactoring"
        elif is_feature:
            domain = "feature_implementation"

        return TaskStructureFeatures(
            decomposability_score=decomposability,
            sequential_dependency_score=sequential,
            tool_complexity=tool_complexity,
            domain=domain,
            metadata={
                "is_debugging": is_debugging,
                "is_refactoring": is_refactoring,
                "is_feature": is_feature,
                "keyword_counts": {
                    "decomposable": decomp_count,
                    "sequential": seq_count,
                    "tool": tool_count,
                },
            },
        )


# =============================================================================
# FALLBACK/DEGRADATION STRATEGY
# =============================================================================

class DegradationStrategy:
    """
    Manages automatic degradation from complex to simple topologies.

    When MAS isn't working, gracefully fall back to SAS.
    """

    def __init__(
        self,
        max_format_errors: int = 3,
        max_overhead_violations: int = 2,
    ):
        self.max_format_errors = max_format_errors
        self.max_overhead_violations = max_overhead_violations

        # Counters
        self._format_error_count = 0
        self._overhead_violation_count = 0
        self._current_topology = CoordinationTopology.SAS

    def record_format_error(self):
        """Record a format/parsing error."""
        self._format_error_count += 1

    def record_overhead_violation(self):
        """Record an overhead threshold violation."""
        self._overhead_violation_count += 1

    def reset_counters(self):
        """Reset error counters (e.g., on stage transition)."""
        self._format_error_count = 0
        self._overhead_violation_count = 0

    def should_degrade(self) -> Tuple[bool, str]:
        """
        Check if we should degrade based on error counts.

        Returns:
            (should_degrade, reason)
        """
        if self._format_error_count >= self.max_format_errors:
            return True, (
                f"Format errors ({self._format_error_count}) exceeded threshold. "
                "Degrading to SAS with simplified prompts."
            )

        if self._overhead_violation_count >= self.max_overhead_violations:
            return True, (
                f"Overhead violations ({self._overhead_violation_count}) exceeded threshold. "
                "Degrading to SAS to reduce coordination cost."
            )

        return False, "No degradation needed"

    def get_degraded_config(
        self,
        current_topology: CoordinationTopology,
    ) -> Dict[str, Any]:
        """
        Get configuration for degraded mode.

        When we degrade:
        1. Switch to SAS
        2. Simplify prompts
        3. Reduce context
        4. Increase timeout
        """
        return {
            "topology": CoordinationTopology.SAS,
            "simplify_prompts": True,
            "reduce_context": True,
            "context_reduction_factor": 0.7,
            "increase_timeout": True,
            "timeout_factor": 1.5,
            "single_model": True,  # Use only one model
            "preferred_model": "claude",  # Most reliable for complex reasoning
        }


# =============================================================================
# COORDINATION POLICY
# =============================================================================

@dataclass
class CoordinationPolicy:
    """
    Policy configuration for coordination engine.
    """
    # Architecture selection thresholds
    sequential_threshold: float = 0.7
    decomposable_threshold: float = 0.6
    tool_complexity_high: float = 0.6

    # Overhead thresholds
    max_coordination_overhead: float = 2.5
    max_error_amplification: float = 1.2
    max_redundancy_rate: float = 0.85

    # Budget limits
    max_tokens_per_stage: int = 50000
    max_rounds_per_stage: int = 5
    max_agents: int = 3

    # Degradation triggers
    max_format_errors: int = 3
    max_consecutive_failures: int = 3

    # Model preferences by topology
    model_preferences: Dict[str, List[str]] = field(default_factory=lambda: {
        "sas": ["claude"],
        "mas_independent": ["codex", "gemini", "claude"],
        "mas_centralized": ["claude", "codex", "gemini"],
    })

    @classmethod
    def conservative(cls) -> "CoordinationPolicy":
        """Conservative policy - prefer SAS, tight budgets."""
        return cls(
            sequential_threshold=0.5,
            decomposable_threshold=0.7,
            max_coordination_overhead=1.5,
            max_tokens_per_stage=30000,
            max_rounds_per_stage=3,
            max_agents=2,
        )

    @classmethod
    def aggressive(cls) -> "CoordinationPolicy":
        """Aggressive policy - more MAS, looser budgets."""
        return cls(
            sequential_threshold=0.8,
            decomposable_threshold=0.4,
            max_coordination_overhead=3.0,
            max_tokens_per_stage=100000,
            max_rounds_per_stage=7,
            max_agents=5,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_redundancy(outputs: List[str]) -> float:
    """
    Compute redundancy rate from multiple outputs.

    Simple implementation using character-level similarity.
    For production, use embedding-based similarity.
    """
    if len(outputs) < 2:
        return 0.0

    similarities = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            # Simple Jaccard similarity on words
            words_i = set(outputs[i].lower().split())
            words_j = set(outputs[j].lower().split())
            if not words_i or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            similarities.append(intersection / union if union > 0 else 0)

    return sum(similarities) / len(similarities) if similarities else 0.0


def estimate_error_amplification(
    mas_failures: int,
    mas_total: int,
    sas_failure_rate: float = 0.3,
) -> float:
    """
    Estimate error amplification factor.

    error_amplification = MAS_failure_rate / SAS_failure_rate
    > 1.0 means MAS is performing worse than SAS would
    """
    if mas_total == 0:
        return 1.0

    mas_failure_rate = mas_failures / mas_total

    if sas_failure_rate <= 0:
        return 1.0

    return mas_failure_rate / sas_failure_rate
