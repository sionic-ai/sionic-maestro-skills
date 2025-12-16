"""
5-Stage Workflow Engine implementing the coding loop:
1. Example Analysis - Freeze facts before guessing
2. Hypothesis - Generate competing explanations
3. Implementation - Apply changes safely
4. Debug Loop - Fix without divergence
5. Recursive Improvement - Refactor and stabilize

Based on "Towards a Science of Scaling Agent Systems":
- Stage-specific coordination policies
- Measured overhead control
- Error amplification prevention
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger("maestro.workflow")


class Stage(Enum):
    """The 5 stages of the Maestro workflow."""
    ANALYZE = "analyze"
    HYPOTHESIZE = "hypothesize"
    IMPLEMENT = "implement"
    DEBUG = "debug"
    IMPROVE = "improve"

    @property
    def display_name(self) -> str:
        return {
            Stage.ANALYZE: "Example Analysis",
            Stage.HYPOTHESIZE: "Hypothesis Formulation",
            Stage.IMPLEMENT: "Code Implementation",
            Stage.DEBUG: "Iterative Debugging",
            Stage.IMPROVE: "Recursive Improvement",
        }[self]

    @property
    def next_stage(self) -> Optional["Stage"]:
        """Get the next stage in the workflow."""
        stages = list(Stage)
        idx = stages.index(self)
        if idx < len(stages) - 1:
            return stages[idx + 1]
        return None

    @property
    def can_use_ensemble(self) -> bool:
        """Whether this stage benefits from ensemble generation."""
        # Paper insight: Tool-heavy stages (implement, debug) degrade with MAS
        return self in [Stage.ANALYZE, Stage.HYPOTHESIZE, Stage.IMPROVE]

    @property
    def prefer_single_agent(self) -> bool:
        """Whether this stage should prefer single-agent mode."""
        # Paper insight: Sequential tasks degrade with MAS
        return self in [Stage.IMPLEMENT, Stage.DEBUG]


@dataclass
class StagePolicy:
    """
    Coordination policy for a specific stage.

    Based on paper findings:
    - capability_threshold: Don't consult if baseline is high
    - max_consults: Limit overhead
    - use_ensemble: Whether to generate multiple candidates
    - require_verification: Whether to require test/validation before proceeding
    """
    max_consults: int = 2
    use_ensemble: bool = False
    ensemble_providers: List[str] = field(default_factory=lambda: ["codex", "gemini"])
    require_verification: bool = False
    allow_iteration: bool = False
    max_iterations: int = 3
    suggested_providers: List[str] = field(default_factory=list)


@dataclass
class StageContext:
    """Context passed between stages."""
    task: str
    files: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    selected_hypothesis: Optional[Dict[str, Any]] = None
    implementation: Optional[str] = None
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result from executing a stage."""
    stage: Stage
    success: bool
    output: Dict[str, Any]
    next_stage: Optional[Stage] = None
    consults_used: int = 0
    elapsed_ms: float = 0.0
    error: Optional[str] = None


# Default policies per stage (based on paper findings)
DEFAULT_STAGE_POLICIES: Dict[Stage, StagePolicy] = {
    Stage.ANALYZE: StagePolicy(
        max_consults=2,
        use_ensemble=False,  # Analysis benefits from diverse perspectives but not full ensemble
        suggested_providers=["gemini"],  # Good for large context reading
        require_verification=False,
    ),
    Stage.HYPOTHESIZE: StagePolicy(
        max_consults=2,
        use_ensemble=True,  # Multiple hypotheses are valuable
        ensemble_providers=["codex", "gemini"],
        require_verification=True,  # Must be testable
    ),
    Stage.IMPLEMENT: StagePolicy(
        max_consults=1,  # Paper: tool-heavy tasks degrade with MAS
        use_ensemble=False,  # One implementation, verify with tests
        suggested_providers=["codex"],  # Good for code generation
        require_verification=True,  # Must pass tests
    ),
    Stage.DEBUG: StagePolicy(
        max_consults=1,  # Paper: sequential tasks degrade with MAS
        use_ensemble=False,
        allow_iteration=True,
        max_iterations=5,
        suggested_providers=["codex"],
        require_verification=True,
    ),
    Stage.IMPROVE: StagePolicy(
        max_consults=2,
        use_ensemble=True,  # Can benefit from diverse improvement suggestions
        ensemble_providers=["claude", "gemini"],
        require_verification=True,
    ),
}


class WorkflowEngine:
    """
    Manages the 5-stage workflow with measured coordination.

    Key principles from the paper:
    1. Tool-Coordination Trade-off: Limit consults in tool-heavy stages
    2. Capability Saturation: Skip consults when baseline is high
    3. Error Amplification: Always verify before proceeding
    """

    def __init__(
        self,
        policies: Optional[Dict[Stage, StagePolicy]] = None,
        capability_threshold: float = 0.45,
    ):
        self.policies = policies or DEFAULT_STAGE_POLICIES.copy()
        self.capability_threshold = capability_threshold
        self.current_stage: Optional[Stage] = None
        self.context: Optional[StageContext] = None
        self.history: List[StageResult] = []
        self.total_consults: int = 0
        self.max_total_consults: int = 6  # Budget across entire workflow

    def start(self, task: str, initial_context: Optional[Dict[str, Any]] = None) -> StageContext:
        """Start a new workflow."""
        self.current_stage = Stage.ANALYZE
        self.context = StageContext(
            task=task,
            files=initial_context.get("files", []) if initial_context else [],
            facts=initial_context.get("facts", []) if initial_context else [],
            errors=initial_context.get("errors", []) if initial_context else [],
            constraints=initial_context.get("constraints", []) if initial_context else [],
        )
        self.history = []
        self.total_consults = 0
        logger.info(f"Started workflow: {task[:50]}...")
        return self.context

    def get_stage_policy(self, stage: Stage) -> StagePolicy:
        """Get the policy for a stage."""
        return self.policies.get(stage, StagePolicy())

    def should_use_ensemble(self, stage: Stage, baseline_confidence: float = 0.0) -> bool:
        """
        Determine if ensemble should be used for this stage.

        Paper insight: Capability saturation at ~45% baseline means
        coordination yields diminishing returns.
        """
        policy = self.get_stage_policy(stage)

        if not policy.use_ensemble:
            return False

        # Check capability saturation
        if baseline_confidence >= self.capability_threshold:
            logger.info(
                f"Skipping ensemble for {stage.value}: baseline confidence "
                f"{baseline_confidence:.2f} >= threshold {self.capability_threshold:.2f}"
            )
            return False

        # Check budget
        if self.total_consults >= self.max_total_consults:
            logger.warning(f"Consult budget exhausted ({self.total_consults}/{self.max_total_consults})")
            return False

        return True

    def should_consult(self, stage: Stage, consults_this_stage: int = 0) -> bool:
        """Determine if we should consult an external model."""
        policy = self.get_stage_policy(stage)

        if consults_this_stage >= policy.max_consults:
            return False

        if self.total_consults >= self.max_total_consults:
            return False

        return True

    def record_consult(self) -> None:
        """Record that a consult was made."""
        self.total_consults += 1

    def transition(self, result: StageResult) -> Optional[Stage]:
        """
        Handle stage transition based on result.

        Returns the next stage or None if workflow is complete.
        """
        self.history.append(result)

        if not result.success:
            # On failure, may retry current stage or go to debug
            if result.stage == Stage.IMPLEMENT and self.policies[Stage.DEBUG].allow_iteration:
                return Stage.DEBUG
            # Otherwise stay on current stage (caller handles retry)
            return result.stage

        # Normal progression
        next_stage = result.next_stage or result.stage.next_stage

        if next_stage:
            self.current_stage = next_stage
            logger.info(f"Transitioning to {next_stage.display_name}")

        return next_stage

    def get_stage_prompt_template(self, stage: Stage) -> str:
        """Get the prompt template for a stage."""
        templates = {
            Stage.ANALYZE: """## Example Analysis

**Task**: {task}

**Instructions**:
1. List ALL observations/facts (no guessing)
2. Identify reproduction steps
3. Note affected modules/files
4. Define invariants (what must NOT break)

**Context**:
{context}

**Output Format** (JSON):
```json
{{
  "observations": ["..."],
  "repro_steps": ["..."],
  "affected_modules": ["..."],
  "invariants": ["..."],
  "confidence": 0.0-1.0
}}
```""",

            Stage.HYPOTHESIZE: """## Hypothesis Formulation

**Task**: {task}

**Facts from Analysis**:
{facts}

**Instructions**:
1. Generate 2-4 competing hypotheses for the root cause
2. Each hypothesis must have a TESTABLE verification method
3. Rank by likelihood and ease of verification

**Output Format** (JSON):
```json
{{
  "hypotheses": [
    {{
      "id": "H1",
      "claim": "...",
      "evidence_for": ["..."],
      "evidence_against": ["..."],
      "verification_test": "...",
      "confidence": 0.0-1.0
    }}
  ],
  "recommended_order": ["H1", "H2", ...]
}}
```""",

            Stage.IMPLEMENT: """## Code Implementation

**Task**: {task}

**Selected Hypothesis**:
{hypothesis}

**Instructions**:
1. Generate the MINIMAL change to verify/fix the hypothesis
2. Output as unified diff format
3. List files that will be changed
4. Identify any risks or side effects

**Output Format**:
```json
{{
  "patch_plan": "Brief description of changes",
  "files_to_change": ["..."],
  "diff": "```diff\\n...\\n```",
  "risks": ["..."],
  "verification_command": "pytest test_x.py -k test_name"
}}
```""",

            Stage.DEBUG: """## Iterative Debugging

**Task**: {task}

**Current Error**:
{error}

**Previous Attempts**:
{attempts}

**Instructions**:
1. Analyze the NEW error (what changed from before?)
2. Update hypothesis confidence
3. Propose SINGLE next action (smallest possible change)
4. If stuck after 3 attempts, consider alternative hypothesis

**Output Format** (JSON):
```json
{{
  "error_analysis": "...",
  "hypothesis_update": {{
    "id": "H1",
    "new_confidence": 0.0-1.0,
    "reason": "..."
  }},
  "next_action": {{
    "type": "edit|test|rollback|escalate",
    "details": "..."
  }},
  "should_try_alternative": false
}}
```""",

            Stage.IMPROVE: """## Recursive Improvement

**Task**: {task}

**Implementation**:
{implementation}

**Test Results**: PASSING

**Instructions**:
1. Review for code quality (readability, maintainability)
2. Identify potential edge cases
3. Suggest documentation needs
4. Propose regression test additions
5. Do NOT over-engineer

**Output Format** (JSON):
```json
{{
  "improvements": [
    {{
      "type": "refactor|edge_case|docs|test",
      "description": "...",
      "risk": "low|medium|high",
      "benefit": "...",
      "diff": "..."
    }}
  ],
  "final_checklist": ["..."],
  "ready_for_merge": true
}}
```""",
        }
        return templates.get(stage, "")

    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state for serialization/display."""
        return {
            "current_stage": self.current_stage.value if self.current_stage else None,
            "total_consults": self.total_consults,
            "max_consults": self.max_total_consults,
            "history": [
                {
                    "stage": r.stage.value,
                    "success": r.success,
                    "consults_used": r.consults_used,
                    "elapsed_ms": r.elapsed_ms,
                }
                for r in self.history
            ],
            "context_summary": {
                "task": self.context.task[:100] if self.context else "",
                "files_count": len(self.context.files) if self.context else 0,
                "facts_count": len(self.context.facts) if self.context else 0,
                "errors_count": len(self.context.errors) if self.context else 0,
            } if self.context else {},
        }


class WorkflowRunner:
    """
    High-level runner that orchestrates the workflow with provider calls.

    This is the main interface for the MCP tools.
    """

    def __init__(
        self,
        engine: WorkflowEngine,
        registry: "ProviderRegistry",
        tracer: Optional["TraceStore"] = None,
    ):
        self.engine = engine
        self.registry = registry
        self.tracer = tracer

    async def run_stage(
        self,
        stage: Stage,
        context: StageContext,
        providers: Optional[List[str]] = None,
        baseline_confidence: float = 0.0,
        timeout_sec: Optional[int] = None,
    ) -> StageResult:
        """
        Execute a single stage of the workflow.

        Returns StageResult with outputs and next stage recommendation.
        """
        import time
        from .context import ContextPacker

        start_time = time.time()
        policy = self.engine.get_stage_policy(stage)
        consults_used = 0

        # Determine which providers to use
        if providers is None:
            if self.engine.should_use_ensemble(stage, baseline_confidence):
                providers = policy.ensemble_providers
            else:
                providers = policy.suggested_providers[:1] if policy.suggested_providers else ["claude"]

        # Build prompt
        template = self.engine.get_stage_prompt_template(stage)
        packed_context = ContextPacker.pack(
            files=context.files,
            facts=context.facts,
            errors=context.errors,
            constraints=context.constraints,
        )

        prompt = template.format(
            task=context.task,
            context=packed_context,
            facts="\n".join(f"- {f}" for f in context.facts),
            hypothesis=context.selected_hypothesis or "N/A",
            error=context.errors[-1] if context.errors else "N/A",
            attempts=len([h for h in self.engine.history if h.stage == stage]),
            implementation=context.implementation or "N/A",
        )

        # Execute
        try:
            if len(providers) > 1 and policy.use_ensemble:
                # Parallel execution for ensemble (pass timeout_sec to each request)
                requests = [
                    {"provider": p, "prompt": prompt, "timeout_sec": timeout_sec}
                    for p in providers
                ]
                responses = await self.registry.run_parallel(requests)
                consults_used = len(responses)
                output = {
                    "candidates": [
                        {
                            "provider": r.provider,
                            "content": r.stdout,
                            "ok": r.ok,
                            "elapsed_ms": r.elapsed_ms,
                        }
                        for r in responses
                    ]
                }
            else:
                # Single provider (pass timeout_sec)
                provider_name = providers[0] if providers else "claude"
                response = self.registry.run(provider_name, prompt, timeout_sec=timeout_sec)
                consults_used = 1
                output = {
                    "provider": response.provider,
                    "content": response.stdout,
                    "ok": response.ok,
                    "elapsed_ms": response.elapsed_ms,
                    "structured": response.structured,
                }

            # Record consults
            for _ in range(consults_used):
                self.engine.record_consult()

            elapsed_ms = (time.time() - start_time) * 1000

            result = StageResult(
                stage=stage,
                success=True,
                output=output,
                next_stage=stage.next_stage,
                consults_used=consults_used,
                elapsed_ms=elapsed_ms,
            )

            # Log to tracer
            if self.tracer:
                self.tracer.log_stage(result)

            return result

        except Exception as e:
            logger.exception(f"Stage {stage.value} failed")
            return StageResult(
                stage=stage,
                success=False,
                output={},
                error=str(e),
                consults_used=consults_used,
                elapsed_ms=(time.time() - start_time) * 1000,
            )
