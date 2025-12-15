"""
Zen Skills MCP Server

A Model Context Protocol server implementing:
- Multi-LLM orchestration (Claude/Codex/Gemini)
- 5-stage workflow (Analyze → Hypothesize → Implement → Debug → Improve)
- Measured coordination based on "Towards a Science of Scaling Agent Systems"
- Poetiq-style ensemble generation with tests_first selection
- Role-based persona prompting via cli_clients.yaml

Architecture: Centralized Consult Pattern
- Claude Code = Orchestrator (tool execution)
- Codex/Gemini/Claude CLI = Consultants (text advice only)
"""

from fastmcp import FastMCP
from typing import List, Dict, Optional, Literal, Any
import asyncio
import logging
import os
import json
from pathlib import Path

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from zen.config import ZenConfig
from zen.providers import ProviderRegistry, ProviderResponse
from zen.context import ContextPacker, PackingConfig
from zen.workflow import WorkflowEngine, Stage, StageContext, WorkflowRunner
from zen.selection import (
    SelectionEngine, SelectionMode, Candidate, TestSignal, LintSignal,
    RedFlagConfig, validate_candidate_content, RedFlagResult
)
from zen.tracing import TraceStore, Metrics, EvidenceType
from zen.verify import VerificationEngine, VerificationType, VerificationResult
from zen.workspace import WorkspaceManager, WorkspaceConfig, PatchResult
from zen.consensus import ConsensusEngine, ConsensusConfig, RedFlagConfig as ConsensusRedFlagConfig

# MAKER-style modules
from zen.maker import (
    StageType, MicroStepType, MicroStepSpec, MICRO_STEP_SPECS,
    RedFlagger, RedFlaggerConfig, RedFlagResult as MakerRedFlagResult,
    VoteStep, VoteResult, VoteCandidate,
    Calibrator, CalibrationResult,
    get_tools_for_step, get_tools_for_stage, get_all_micro_steps_for_stage,
    get_default_k_for_step, step_has_oracle,
)
from zen.skills import (
    SkillManifest, SkillDefinition, DynamicToolRegistry, ToolLoadState,
    SkillSession, SkillLoader, create_skill_session, get_recommended_tools_for_task,
)
from zen.coordination import (
    CoordinationTopology, TaskStructureFeatures, CoordinationDecision,
    CoordinationMetrics, MetricsTracker, ArchitectureSelectionEngine,
    TaskStructureClassifier, DegradationStrategy, CoordinationPolicy,
    compute_redundancy, estimate_error_amplification,
)


# ============================================================================
# Configuration Loading (cli_clients.yaml, roles, skills, schemas)
# ============================================================================

BASE_DIR = Path(__file__).parent


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not YAML_AVAILABLE:
        logging.warning("PyYAML not installed, using defaults")
        return {}
    if not path.exists():
        logging.warning(f"Config file not found: {path}")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_role_prompt(role_name: str) -> str:
    """Load a role/persona prompt from the roles directory."""
    role_file = BASE_DIR / "roles" / f"{role_name}.md"
    if not role_file.exists():
        return ""
    return role_file.read_text()


def load_skill_definition(stage: str) -> Dict[str, Any]:
    """Load a skill definition from the skills directory."""
    # Map stage names to skill files
    skill_map = {
        "analyze": "stage1_example_analysis.md",
        "hypothesize": "stage2_hypothesis.md",
        "implement": "stage3_implementation.md",
        "debug": "stage4_debug_loop.md",
        "improve": "stage5_recursive_improve.md",
    }
    skill_file = BASE_DIR / "skills" / skill_map.get(stage, "")
    if not skill_file.exists():
        return {}

    content = skill_file.read_text()

    # Parse YAML frontmatter if present
    result = {"content": content, "stage": stage}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3 and YAML_AVAILABLE:
            try:
                frontmatter = yaml.safe_load(parts[1])
                result["metadata"] = frontmatter
                result["content"] = parts[2].strip()
            except Exception:
                pass
    return result


def load_output_schema(schema_name: str) -> Dict[str, Any]:
    """Load a JSON schema from the schemas directory."""
    schema_file = BASE_DIR / "schemas" / f"{schema_name}.json"
    if not schema_file.exists():
        return {}
    try:
        with open(schema_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# Load CLI clients configuration
CLI_CONFIG = load_yaml_config(BASE_DIR / "conf" / "cli_clients.yaml")

# Configure logging
logging.basicConfig(
    level=os.getenv("ZEN_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("zen.server")

# Initialize configuration
config = ZenConfig.from_env()

# Initialize MCP server
mcp = FastMCP(
    "zen-skills-mcp",
    description=(
        "Multi-LLM orchestration with measured coordination. "
        "Implements 5-stage workflow (Analyze→Hypothesize→Implement→Debug→Improve) "
        "with Poetiq-style ensemble selection. Based on 'Towards a Science of Scaling Agent Systems'."
    ),
)

# Initialize components
registry = ProviderRegistry.from_config(config)
workflow_engine = WorkflowEngine(capability_threshold=config.policy.capability_threshold)
selection_engine = SelectionEngine()
trace_store = TraceStore(
    trace_dir=config.tracing.trace_dir,
    session_id=None,  # Auto-generate
)

# Initialize MAKER/Skill components
skill_manifest = SkillManifest(str(BASE_DIR / "conf" / "skill_manifest.yaml"))
disabled_tools = set(os.getenv("ZEN_DISABLED_TOOLS", "").split(",")) if os.getenv("ZEN_DISABLED_TOOLS") else set()
tool_registry = DynamicToolRegistry(
    manifest=skill_manifest,
    max_tools=int(os.getenv("ZEN_MAX_TOOLS", "10")),
    disabled_tools=disabled_tools,
)
skill_session = SkillSession(skill_manifest, tool_registry)
skill_loader = SkillLoader(str(BASE_DIR))
red_flagger = RedFlagger()
calibrator = Calibrator(red_flagger)

# Initialize Coordination components (Architecture Selection Engine)
coordination_policy = CoordinationPolicy()
metrics_tracker = MetricsTracker()
architecture_engine = ArchitectureSelectionEngine(
    metrics_tracker=metrics_tracker,
    default_topology=CoordinationTopology.SAS,
)
task_classifier = TaskStructureClassifier()
degradation_strategy = DegradationStrategy()


# ============================================================================
# TOOL: zen_consult - Single provider consultation
# ============================================================================

@mcp.tool()
def zen_consult(
    prompt: str,
    provider: Literal["codex", "gemini", "claude"] = "codex",
    model: Optional[str] = None,
    context_files: List[str] = [],
    context_facts: List[str] = [],
    context_errors: List[str] = [],
    context_constraints: List[str] = [],
    stage: Optional[str] = None,
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """
    Consult an external LLM CLI as a 'sub-agent'.

    Use for:
    - Example Analysis: Understanding code structure, summarizing logs
    - Hypothesis Generation: Getting diverse perspectives on root causes
    - Code Review: Safety, security, and maintainability checks
    - Improvement Suggestions: Refactoring ideas

    DO NOT use for:
    - Direct file editing (orchestrator should edit)
    - Running tests or commands (orchestrator should run)

    Paper Insight: "Tool-coordination trade-off" - limit consults in tool-heavy stages.

    Args:
        prompt: The specific question or task for the consultant.
        provider: Which CLI to use ('codex', 'gemini', 'claude').
        model: Specific model (uses default if not specified).
        context_files: File paths to include in context.
        context_facts: Established facts about the problem.
        context_errors: Error logs or stack traces.
        context_constraints: Requirements that must not be violated.
        stage: Current workflow stage (for context-aware packing).
        timeout_sec: Timeout for the CLI call.

    Returns:
        Dictionary with 'ok', 'content', 'provider', 'model', 'elapsed_ms'.
    """
    # Check if provider is enabled
    if provider not in registry.list_providers():
        return {
            "ok": False,
            "error": f"Provider '{provider}' is not enabled. Available: {registry.list_providers()}",
        }

    # Pack context with stage-specific strategy
    if stage:
        context_str = ContextPacker.for_stage(
            stage=stage,
            files=context_files,
            facts=context_facts,
            errors=context_errors,
            constraints=context_constraints,
            task=prompt,
        )
    else:
        context_str = ContextPacker.pack(
            files=context_files,
            facts=context_facts,
            errors=context_errors,
            constraints=context_constraints,
        )

    full_prompt = f"{context_str}\n\n## Task\n{prompt}" if context_str.strip() else prompt

    # Execute
    response = registry.run(
        provider_name=provider,
        prompt=full_prompt,
        model=model,
        timeout_sec=timeout_sec,
    )

    # Log to trace store
    trace_store.log_provider_call(
        provider=response.provider,
        model=response.model,
        prompt=full_prompt,
        response=response.stdout,
        elapsed_ms=response.elapsed_ms,
        success=response.ok,
        stage=stage,
        error=response.stderr if not response.ok else None,
    )

    return {
        "ok": response.ok,
        "content": response.stdout,
        "provider": response.provider,
        "model": response.model,
        "elapsed_ms": response.elapsed_ms,
        "stderr": response.stderr if not response.ok else None,
        "truncated": response.truncated,
    }


# ============================================================================
# TOOL: zen_ensemble_generate - Multi-provider candidate generation
# ============================================================================

@mcp.tool()
def zen_ensemble_generate(
    task: str,
    providers: List[str] = ["codex", "gemini"],
    context_files: List[str] = [],
    context_facts: List[str] = [],
    context_errors: List[str] = [],
    n_per_provider: int = 1,
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """
    Generate multiple candidate solutions using different LLMs.

    Paper Insight: "Independent agents allow diverse exploration"
    but "error amplification" means we MUST verify before accepting.

    Use for:
    - Hypothesis generation (get diverse root cause theories)
    - Solution exploration (multiple approaches to a problem)
    - Review collection (different perspectives on code quality)

    DO NOT use for:
    - Implementation (creates merge conflicts)
    - Sequential debugging (paper shows MAS degrades here)

    Args:
        task: The task to generate candidates for.
        providers: List of providers to use (default: codex, gemini).
        context_files: Files to include in context.
        context_facts: Known facts.
        context_errors: Error logs.
        n_per_provider: Number of candidates per provider.
        timeout_sec: Timeout per provider.

    Returns:
        Dictionary with 'candidates' list, each having id, provider, content.
    """
    # Check capability saturation
    if not workflow_engine.should_use_ensemble(Stage.HYPOTHESIZE):
        return {
            "ok": True,
            "candidates": [],
            "warning": "Ensemble skipped due to policy (budget or capability threshold)",
        }

    # Validate providers
    available = registry.list_providers()
    valid_providers = [p for p in providers if p in available]
    if not valid_providers:
        return {
            "ok": False,
            "error": f"No valid providers. Available: {available}",
        }

    # Pack context
    context_str = ContextPacker.pack(
        files=context_files,
        facts=context_facts,
        errors=context_errors,
        constraints=[],
    )

    full_prompt = f"{context_str}\n\n## Task\n{task}" if context_str.strip() else task

    # Build requests
    requests = []
    for provider in valid_providers:
        for i in range(n_per_provider):
            requests.append({
                "provider": provider,
                "prompt": full_prompt,
                "timeout_sec": timeout_sec,
            })

    # Execute in parallel
    try:
        responses = asyncio.get_event_loop().run_until_complete(
            registry.run_parallel(requests)
        )
    except RuntimeError:
        # No event loop - run sequentially
        responses = [
            registry.run(req["provider"], req["prompt"], timeout_sec=req.get("timeout_sec"))
            for req in requests
        ]

    # Format candidates
    candidates = []
    for i, (req, resp) in enumerate(zip(requests, responses)):
        cand_id = f"c{i+1}"
        candidates.append({
            "id": cand_id,
            "provider": resp.provider,
            "model": resp.model,
            "content": resp.stdout,
            "ok": resp.ok,
            "elapsed_ms": resp.elapsed_ms,
            "error": resp.stderr if not resp.ok else None,
        })

        # Log to trace store
        trace_store.log_provider_call(
            provider=resp.provider,
            model=resp.model,
            prompt=full_prompt,
            response=resp.stdout,
            elapsed_ms=resp.elapsed_ms,
            success=resp.ok,
            stage="ensemble",
        )

    return {
        "ok": True,
        "candidates": candidates,
        "count": len(candidates),
        "providers_used": valid_providers,
    }


# ============================================================================
# TOOL: zen_select_best - Select best candidate
# ============================================================================

@mcp.tool()
def zen_select_best(
    candidates: List[Dict[str, Any]],
    mode: Literal["tests_first", "llm_judge", "hybrid"] = "tests_first",
    test_results: Optional[List[Dict[str, Any]]] = None,
    lint_results: Optional[List[Dict[str, Any]]] = None,
    judge_provider: Literal["claude", "codex", "gemini"] = "claude",
    criteria: List[str] = ["correctness", "safety", "completeness"],
) -> Dict[str, Any]:
    """
    Select the best candidate from ensemble output.

    Paper Insight: "Centralized verification prevents error amplification."
    Independent agents amplify errors 17.2x without verification!

    Selection Priority (tests_first mode - RECOMMENDED):
    1. Automated tests (pytest, npm test) - HIGHEST PRIORITY
    2. Static analysis (lint scores)
    3. LLM Judge evaluation
    4. Provider trust scores

    Args:
        candidates: List of candidates from zen_ensemble_generate.
        mode: Selection strategy ('tests_first', 'llm_judge', 'hybrid').
        test_results: Optional test results [{candidate_id, passed, output}].
        lint_results: Optional lint results [{candidate_id, score, output}].
        judge_provider: Which LLM to use for judging.
        criteria: Evaluation criteria for LLM judge.

    Returns:
        Dictionary with winner_id, winner, rationale, scores.
    """
    if not candidates:
        return {"ok": False, "error": "No candidates provided"}

    # Convert to Candidate objects
    cand_objects = []
    for c in candidates:
        cand_objects.append(Candidate(
            id=c.get("id", f"c{len(cand_objects)+1}"),
            provider=c.get("provider", "unknown"),
            content=c.get("content", ""),
            model=c.get("model", ""),
        ))

    # Convert test signals
    test_signals = None
    if test_results:
        test_signals = [
            TestSignal(
                candidate_id=t["candidate_id"],
                passed=t.get("passed", False),
                output=t.get("output", ""),
            )
            for t in test_results
        ]

    # Convert lint signals
    lint_signals = None
    if lint_results:
        lint_signals = [
            LintSignal(
                candidate_id=l["candidate_id"],
                score=l.get("score", 0.5),
                output=l.get("output", ""),
            )
            for l in lint_results
        ]

    # Judge callback
    def judge_callback(provider, prompt):
        return registry.run(provider, prompt)

    # Select mode
    selection_mode = {
        "tests_first": SelectionMode.TESTS_FIRST,
        "llm_judge": SelectionMode.LLM_JUDGE,
        "hybrid": SelectionMode.HYBRID,
    }.get(mode, SelectionMode.TESTS_FIRST)

    # Update selection engine criteria
    selection_engine.judge_rubric = criteria
    selection_engine.judge_provider = judge_provider

    # Perform selection
    result = selection_engine.select(
        candidates=cand_objects,
        mode=selection_mode,
        test_signals=test_signals,
        lint_signals=lint_signals,
        judge_callback=judge_callback if mode in ["llm_judge", "hybrid"] else None,
    )

    # Log selection
    trace_store.log_selection(result)

    return {
        "ok": True,
        "winner_id": result.winner_id,
        "winner": {
            "id": result.winner.id,
            "provider": result.winner.provider,
            "content": result.winner.content,
        },
        "rationale": result.rationale,
        "scores": result.scores,
        "mode_used": result.mode_used.value,
        "requires_review": result.metadata.get("requires_review", False),
        "warning": result.metadata.get("warning"),
    }


# ============================================================================
# TOOL: zen_pack_context - Context engineering utility
# ============================================================================

@mcp.tool()
def zen_pack_context(
    files: List[str] = [],
    facts: List[str] = [],
    errors: List[str] = [],
    constraints: List[str] = [],
    stage: Optional[str] = None,
    max_chars: int = 40000,
) -> Dict[str, Any]:
    """
    Pack context for sub-agent calls with smart truncation.

    Paper Insight: "Information fragmentation increases coordination tax."
    This tool minimizes overhead by smart excerpting and prioritization.

    Stage-specific strategies:
    - analyze: Broad context, more files
    - hypothesize: Focused on errors and facts
    - implement: Narrow, specific files only
    - debug: Error-heavy, recent changes
    - improve: Full context for review

    Args:
        files: File paths (supports globs like 'src/*.py').
        facts: Known facts about the problem.
        errors: Error logs or stack traces.
        constraints: Requirements that must not be violated.
        stage: Current workflow stage for optimized packing.
        max_chars: Maximum context characters.

    Returns:
        Dictionary with 'packed_context', 'stats'.
    """
    if stage:
        packed = ContextPacker.for_stage(
            stage=stage,
            files=files,
            facts=facts,
            errors=errors,
            constraints=constraints,
        )
    else:
        config = PackingConfig(max_total_chars=max_chars)
        packed = ContextPacker.pack(files, facts, errors, constraints, config)

    return {
        "ok": True,
        "packed_context": packed,
        "stats": {
            "total_chars": len(packed),
            "truncated": len(packed) >= max_chars - 100,
            "files_requested": len(files),
            "facts_count": len(facts),
            "errors_count": len(errors),
        },
    }


# ============================================================================
# TOOL: zen_workflow_state - Get current workflow state
# ============================================================================

@mcp.tool()
def zen_workflow_state() -> Dict[str, Any]:
    """
    Get the current workflow state and metrics.

    Shows:
    - Current stage
    - Consult budget used/remaining
    - Stage history
    - Paper-aligned metrics (overhead, efficiency)

    Returns:
        Dictionary with workflow state and metrics.
    """
    state = workflow_engine.get_workflow_state()
    metrics = Metrics.compute(trace_store)

    return {
        "ok": True,
        "workflow": state,
        "metrics": metrics.to_dict(),
        "summary": metrics.summary(),
    }


# ============================================================================
# TOOL: zen_run_stage - Execute a workflow stage
# ============================================================================

@mcp.tool()
def zen_run_stage(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
    task: str,
    context_files: List[str] = [],
    context_facts: List[str] = [],
    context_errors: List[str] = [],
    providers: Optional[List[str]] = None,
    baseline_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Execute a single stage of the 5-stage workflow.

    Stages:
    1. analyze: Freeze facts before guessing
    2. hypothesize: Generate competing explanations
    3. implement: Apply changes safely (single agent preferred)
    4. debug: Fix without divergence (single agent preferred)
    5. improve: Refactor and stabilize

    Paper Insights Applied:
    - Tool-heavy stages (implement, debug) use single agent
    - Capability saturation: Skip ensemble if baseline > 45%
    - Error amplification: All outputs need verification

    Args:
        stage: Which stage to run.
        task: The task description.
        context_files: Relevant files.
        context_facts: Known facts.
        context_errors: Error logs.
        providers: Override default providers for this stage.
        baseline_confidence: Your confidence level (0.0-1.0).

    Returns:
        Stage result with output and next stage recommendation.
    """
    # Map string to Stage enum
    stage_enum = {
        "analyze": Stage.ANALYZE,
        "hypothesize": Stage.HYPOTHESIZE,
        "implement": Stage.IMPLEMENT,
        "debug": Stage.DEBUG,
        "improve": Stage.IMPROVE,
    }.get(stage)

    if not stage_enum:
        return {"ok": False, "error": f"Unknown stage: {stage}"}

    # Initialize context
    context = StageContext(
        task=task,
        files=context_files,
        facts=context_facts,
        errors=context_errors,
    )

    # Create runner
    runner = WorkflowRunner(
        engine=workflow_engine,
        registry=registry,
        tracer=trace_store,
    )

    # Run stage
    try:
        result = asyncio.get_event_loop().run_until_complete(
            runner.run_stage(
                stage=stage_enum,
                context=context,
                providers=providers,
                baseline_confidence=baseline_confidence,
            )
        )
    except RuntimeError:
        # No event loop - create one
        result = asyncio.run(
            runner.run_stage(
                stage=stage_enum,
                context=context,
                providers=providers,
                baseline_confidence=baseline_confidence,
            )
        )

    return {
        "ok": result.success,
        "stage": result.stage.value,
        "stage_display": result.stage.display_name,
        "output": result.output,
        "next_stage": result.next_stage.value if result.next_stage else None,
        "consults_used": result.consults_used,
        "elapsed_ms": result.elapsed_ms,
        "error": result.error,
    }


# ============================================================================
# TOOL: zen_get_metrics - Get detailed metrics
# ============================================================================

@mcp.tool()
def zen_get_metrics() -> Dict[str, Any]:
    """
    Get detailed metrics aligned with the paper.

    Paper Metrics:
    - Coordination Overhead (O%): Extra turns vs single-agent
    - Efficiency Score (Ec): Success rate / relative overhead
    - Consults per Stage: Average sub-agent calls per stage

    Use to:
    - Monitor coordination costs
    - Identify when to reduce/increase collaboration
    - Compare against paper benchmarks

    Returns:
        Detailed metrics dictionary.
    """
    metrics = Metrics.compute(trace_store)
    return {
        "ok": True,
        "metrics": metrics.to_dict(),
        "summary": metrics.summary(),
    }


# ============================================================================
# TOOL: zen_list_providers - List available providers
# ============================================================================

@mcp.tool()
def zen_list_providers() -> Dict[str, Any]:
    """
    List available CLI providers and their status.

    Returns:
        Dictionary with provider names and configurations.
    """
    providers = {}
    for name in registry.list_providers():
        provider = registry.get(name)
        if provider:
            providers[name] = {
                "enabled": True,
                "cmd": provider.cmd,
                "default_model": provider.default_model,
                "timeout_sec": provider.timeout_sec,
            }

    return {
        "ok": True,
        "providers": providers,
        "available": list(providers.keys()),
    }


# ============================================================================
# TOOL: zen_get_skill - Get skill definition for a stage
# ============================================================================

@mcp.tool()
def zen_get_skill(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
) -> Dict[str, Any]:
    """
    Get the skill definition and guidance for a workflow stage.

    Each stage has a detailed skill definition with:
    - Goal and purpose
    - Process steps
    - Output schema (JSON format)
    - Coordination policy (when to use ensemble)
    - Anti-patterns to avoid
    - Exit criteria

    Use this to understand how to execute a stage properly.

    Args:
        stage: Which stage's skill to retrieve.

    Returns:
        Skill definition with metadata, content, and output schema.
    """
    skill = load_skill_definition(stage)
    if not skill:
        return {"ok": False, "error": f"Skill not found for stage: {stage}"}

    # Load the output schema for this stage
    schema_map = {
        "analyze": "stage1_output",
        "hypothesize": "stage2_output",
        "implement": "stage3_output",
        "debug": "stage4_output",
        "improve": "stage5_output",
    }
    output_schema = load_output_schema(schema_map.get(stage, ""))

    # Get the role for this stage from CLI config
    stage_config = CLI_CONFIG.get("stage_mapping", {}).get(stage, {})
    primary_role = stage_config.get("primary_role", "")

    return {
        "ok": True,
        "stage": stage,
        "metadata": skill.get("metadata", {}),
        "content": skill.get("content", ""),
        "output_schema": output_schema,
        "primary_role": primary_role,
        "coordination_policy": {
            "topology": stage_config.get("topology", "SAS"),
            "max_consults": stage_config.get("max_consults", 2),
            "ensemble_allowed": stage_config.get("ensemble_allowed", False),
        },
    }


# ============================================================================
# TOOL: zen_get_role - Get role/persona prompt
# ============================================================================

@mcp.tool()
def zen_get_role(
    role: Literal[
        "example_analyst",
        "hypothesis_scientist",
        "implementer",
        "debugger",
        "refiner",
        "judge",
    ],
) -> Dict[str, Any]:
    """
    Get the role/persona prompt for a specific role.

    Roles provide system prompts that shape how an LLM approaches a task.
    Each role has specific traits, focus areas, and output expectations.

    Available roles:
    - example_analyst: Extracts facts, observations, constraints
    - hypothesis_scientist: Generates testable hypotheses
    - implementer: Makes minimal, testable code changes
    - debugger: Systematic debugging with iteration limits
    - refiner: Quality improvement and regression testing
    - judge: Objective candidate selection

    Args:
        role: Which role's prompt to retrieve.

    Returns:
        Role definition with prompt content and metadata.
    """
    prompt_content = load_role_prompt(role)
    if not prompt_content:
        return {"ok": False, "error": f"Role not found: {role}"}

    # Get role metadata from CLI config
    role_config = CLI_CONFIG.get("roles", {}).get(role, {})

    return {
        "ok": True,
        "role": role,
        "name": role_config.get("name", role.replace("_", " ").title()),
        "stage": role_config.get("stage", "any"),
        "traits": role_config.get("traits", []),
        "focus": role_config.get("focus", []),
        "prompt": prompt_content,
    }


# ============================================================================
# TOOL: zen_get_schema - Get output schema for validation
# ============================================================================

@mcp.tool()
def zen_get_schema(
    schema: Literal[
        "stage1_output",
        "stage2_output",
        "stage3_output",
        "stage4_output",
        "stage5_output",
        "judge_output",
        "tools",
    ],
) -> Dict[str, Any]:
    """
    Get a JSON schema for output validation.

    Schemas define the expected structure of outputs from each stage.
    Use these to validate that your outputs conform to the expected format.

    Available schemas:
    - stage1_output: Example Analysis output
    - stage2_output: Hypothesis output
    - stage3_output: Implementation output
    - stage4_output: Debug Loop output
    - stage5_output: Recursive Improvement output
    - judge_output: Candidate selection output
    - tools: All tool input/output schemas

    Args:
        schema: Which schema to retrieve.

    Returns:
        JSON Schema definition.
    """
    schema_content = load_output_schema(schema)
    if not schema_content:
        return {"ok": False, "error": f"Schema not found: {schema}"}

    return {
        "ok": True,
        "schema_name": schema,
        "schema": schema_content,
    }


# ============================================================================
# TOOL: zen_consult_with_role - Consult with role-based prompting
# ============================================================================

@mcp.tool()
def zen_consult_with_role(
    prompt: str,
    role: Literal[
        "example_analyst",
        "hypothesis_scientist",
        "implementer",
        "debugger",
        "refiner",
        "judge",
    ],
    provider: Literal["codex", "gemini", "claude"] = "codex",
    context_files: List[str] = [],
    context_facts: List[str] = [],
    context_errors: List[str] = [],
    stage: Optional[str] = None,
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """
    Consult an LLM with a specific role/persona system prompt.

    This combines zen_consult with role-based prompting from cli_clients.yaml.
    The role's system prompt is prepended to shape the LLM's response style.

    Roles and their purposes:
    - example_analyst: Factual observation extraction (Stage 1)
    - hypothesis_scientist: Testable hypothesis generation (Stage 2)
    - implementer: Minimal code changes (Stage 3)
    - debugger: Systematic iteration (Stage 4)
    - refiner: Quality improvement (Stage 5)
    - judge: Candidate selection (Any stage)

    Args:
        prompt: The specific question or task.
        role: Which persona to use.
        provider: Which CLI to use.
        context_files: File paths to include.
        context_facts: Established facts.
        context_errors: Error logs or stack traces.
        stage: Current workflow stage.
        timeout_sec: Timeout for the CLI call.

    Returns:
        Dictionary with response and metadata.
    """
    # Load role prompt
    role_prompt = load_role_prompt(role)

    # Check if provider is enabled
    if provider not in registry.list_providers():
        return {
            "ok": False,
            "error": f"Provider '{provider}' is not enabled. Available: {registry.list_providers()}",
        }

    # Pack context
    if stage:
        context_str = ContextPacker.for_stage(
            stage=stage,
            files=context_files,
            facts=context_facts,
            errors=context_errors,
            constraints=[],
            task=prompt,
        )
    else:
        context_str = ContextPacker.pack(
            files=context_files,
            facts=context_facts,
            errors=context_errors,
            constraints=[],
        )

    # Build full prompt with role system prompt
    full_prompt_parts = []
    if role_prompt:
        full_prompt_parts.append(f"## System Instructions\n{role_prompt}\n")
    if context_str.strip():
        full_prompt_parts.append(f"## Context\n{context_str}\n")
    full_prompt_parts.append(f"## Task\n{prompt}")

    full_prompt = "\n".join(full_prompt_parts)

    # Execute
    response = registry.run(
        provider_name=provider,
        prompt=full_prompt,
        timeout_sec=timeout_sec,
    )

    # Log to trace store
    trace_store.log_provider_call(
        provider=response.provider,
        model=response.model,
        prompt=full_prompt,
        response=response.stdout,
        elapsed_ms=response.elapsed_ms,
        success=response.ok,
        stage=stage,
        error=response.stderr if not response.ok else None,
    )

    return {
        "ok": response.ok,
        "content": response.stdout,
        "provider": response.provider,
        "model": response.model,
        "role": role,
        "elapsed_ms": response.elapsed_ms,
        "stderr": response.stderr if not response.ok else None,
    }


# ============================================================================
# TOOL: zen_get_coordination_policy - Get paper-aligned coordination rules
# ============================================================================

@mcp.tool()
def zen_get_coordination_policy() -> Dict[str, Any]:
    """
    Get the coordination policies based on the paper.

    Returns the rules from "Towards a Science of Scaling Agent Systems":
    - Capability threshold: When to skip ensemble (~45%)
    - Tool-coordination trade-off: Which stages prefer single agent
    - Error amplification: Why Independent topology is forbidden
    - Sequential task handling: Why debug uses single agent

    Use this to understand WHEN to use multi-agent coordination.

    Returns:
        Coordination policies and topologies.
    """
    policies = CLI_CONFIG.get("policies", {})
    topologies = CLI_CONFIG.get("topologies", {})
    stage_mapping = CLI_CONFIG.get("stage_mapping", {})

    return {
        "ok": True,
        "policies": {
            "capability_threshold": policies.get("capability_threshold", 0.45),
            "skip_ensemble_conditions": policies.get("skip_ensemble_conditions", []),
            "error_amplification": policies.get("error_amplification", {}),
            "budgets": policies.get("budgets", {}),
        },
        "topologies": topologies,
        "stage_defaults": {
            stage: {
                "topology": cfg.get("topology"),
                "ensemble_allowed": cfg.get("ensemble_allowed", False),
                "max_consults": cfg.get("max_consults", 2),
            }
            for stage, cfg in stage_mapping.items()
        },
        "paper_insights": {
            "tool_coordination_tradeoff": "Tool-heavy tasks suffer from MAS overhead (β=-0.330). Implement/Debug prefer single agent.",
            "capability_saturation": "When baseline ≥45%, MAS returns diminish (β=-0.408). Skip ensemble if confident.",
            "error_amplification": "Independent agents amplify errors 17.2x. Centralized contains to 4.4x. Use tests, not voting.",
            "sequential_degradation": "Sequential tasks degrade 39-70% with MAS. Debug loops must use single agent.",
        },
    }


# ============================================================================
# TOOL: zen_verify - Run verification (tests/lint/type-check)
# ============================================================================

@mcp.tool()
def zen_verify(
    commands: List[Dict[str, Any]],
    cwd: Optional[str] = None,
    stop_on_failure: bool = False,
    parallel: bool = False,
) -> Dict[str, Any]:
    """
    Run verification commands (tests, lint, type-check).

    Paper Insight: "Test, don't vote" - Deterministic signals (test pass/fail)
    are more reliable than LLM consensus for code correctness.

    This is the foundation of the "tests_first" selection mode.

    Args:
        commands: List of command specs:
                  [{"command": "pytest -v", "type": "unit_test"}, ...]
                  Types: unit_test, lint, type_check, format, build, custom
        cwd: Working directory for commands.
        stop_on_failure: Stop after first failure.
        parallel: Run commands in parallel (for independent checks).

    Returns:
        VerificationReport with pass/fail status for each command.

    Example:
        zen_verify([
            {"command": "python -m pytest tests/ -v", "type": "unit_test"},
            {"command": "python -m ruff check src/", "type": "lint"}
        ])
    """
    engine = VerificationEngine()

    if parallel:
        try:
            report = asyncio.get_event_loop().run_until_complete(
                engine.run_parallel(commands, cwd)
            )
        except RuntimeError:
            report = asyncio.run(engine.run_parallel(commands, cwd))
    else:
        report = engine.run_multiple(commands, cwd, stop_on_failure)

    # Log verification results
    for result in report.results:
        trace_store.log_verification(
            verification_type=result.type.value,
            command=result.command,
            passed=result.passed,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            output=result.stdout,
        )

    return {
        "ok": report.all_passed,
        "summary": report.summary,
        "all_passed": report.all_passed,
        "total_duration_ms": report.total_duration_ms,
        "critical_failures": report.critical_failures,
        "results": [r.to_dict() for r in report.results],
    }


# ============================================================================
# TOOL: zen_apply_patch - Apply a unified diff patch safely
# ============================================================================

@mcp.tool()
def zen_apply_patch(
    patch: str,
    workspace_root: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Apply a unified diff patch to the workspace safely.

    Paper Insight: "Tool execution is restricted to single Executor"
    This tool provides safe, auditable file modifications with:
    - Path validation (no escape from workspace)
    - Allowlist-based file patterns
    - Automatic backups before modification
    - Structured logging of all changes

    Args:
        patch: Unified diff content (git diff or standard diff format).
        workspace_root: Root directory for patch application.
                        Defaults to current working directory.
        dry_run: If True, validate patch but don't apply.

    Returns:
        PatchResult with files changed, created, failed, and backup location.

    Example:
        zen_apply_patch('''
        --- a/src/main.py
        +++ b/src/main.py
        @@ -10,6 +10,7 @@
         def main():
        +    print("Hello")
             pass
        ''')
    """
    root = workspace_root or os.getcwd()
    manager = WorkspaceManager(root)

    result = manager.apply_patch(patch, dry_run=dry_run)

    # Log patch application
    if not dry_run and result.success:
        trace_store.log_patch(
            files_changed=result.files_changed,
            files_created=result.files_created,
            stage="implement",
            backup_dir=result.backup_dir,
        )

    return {
        "ok": result.success,
        "dry_run": dry_run,
        "files_changed": result.files_changed,
        "files_created": result.files_created,
        "files_failed": result.files_failed,
        "backup_dir": result.backup_dir,
        "error": result.error,
        "details": result.details,
    }


# ============================================================================
# TOOL: zen_consensus_vote - MAKER-style micro-decision voting
# ============================================================================

@mcp.tool()
def zen_consensus_vote(
    question: str,
    k: int = 3,
    max_rounds: int = 12,
    providers: List[str] = ["codex", "gemini", "claude"],
    require_json: bool = False,
    max_response_chars: int = 2000,
) -> Dict[str, Any]:
    """
    Run MAKER-style first-to-ahead-by-k voting on a micro-decision.

    Paper Insight: From "Solving a Million-Step LLM Task With Zero Errors":
    - For micro-decisions, voting with error correction dramatically reduces errors
    - Red-flagging (rejecting malformed responses) improves accuracy
    - First-to-ahead-by-k is more efficient than fixed-round voting

    Use for micro-decisions like:
    - "Which file is most likely the source of the bug?" (selection)
    - "Is this patch safe to apply?" (binary yes/no)
    - "What's the key error signal in this log?" (extraction)

    DO NOT use for:
    - Code generation (too complex for voting)
    - Implementation decisions (need context, not consensus)

    Args:
        question: The micro-decision question (keep it focused and answerable).
        k: Votes ahead needed to win (higher = more confident, slower).
        max_rounds: Maximum voting rounds before picking plurality winner.
        providers: Which providers to query (cycled through).
        require_json: If True, responses must be valid JSON.
        max_response_chars: Maximum response length (red-flag if exceeded).

    Returns:
        ConsensusResult with winner, confidence, and vote trace.
    """
    # Provider callback
    def provider_callback(provider: str, prompt: str):
        return registry.run(provider, prompt)

    # Configure red-flagging
    red_flags = ConsensusRedFlagConfig(
        max_chars=max_response_chars,
        require_json=require_json,
    )

    consensus_config = ConsensusConfig(
        k=k,
        max_rounds=max_rounds,
        providers=[p for p in providers if p in registry.list_providers()],
        red_flags=red_flags,
    )

    engine = ConsensusEngine(consensus_config, provider_callback)
    result = engine.vote(question)

    # Log consensus result
    trace_store.log_consensus(
        question=question,
        result=result,
        providers_used=consensus_config.providers,
    )

    return {
        "ok": result.winner is not None,
        "winner": result.winner,
        "confidence": result.confidence,
        "votes_for_winner": result.votes_for_winner,
        "total_valid_votes": result.total_valid_votes,
        "total_votes": result.total_votes,
        "stop_reason": result.stop_reason.value,
        "vote_distribution": result.vote_distribution,
        "red_flagged_count": sum(
            1 for v in result.vote_trace if v.status.value == "red_flagged"
        ),
        "rounds": result.metadata.get("rounds", 0),
    }


# ============================================================================
# TOOL: zen_validate_content - Red-flag validation for any content
# ============================================================================

@mcp.tool()
def zen_validate_content(
    content: str,
    content_type: Literal["general", "diff", "json"] = "general",
    max_chars: int = 15000,
    require_json_fields: List[str] = [],
    forbidden_patterns: List[str] = [],
) -> Dict[str, Any]:
    """
    Validate content against MAKER-style red-flag criteria.

    Paper Insight: "Format errors are signals of reasoning errors"
    Rejecting malformed responses BEFORE using them improves overall accuracy.

    Red-flag criteria:
    - Content too long (indicates rambling, loss of focus)
    - Content too short (indicates incomplete response)
    - Hedging language ("I'm not sure", "it's unclear")
    - Missing required JSON fields
    - Invalid diff format (for patches)

    Args:
        content: The content to validate.
        content_type: Type of content ("general", "diff", "json").
        max_chars: Maximum allowed characters.
        require_json_fields: Required fields if content is JSON.
        forbidden_patterns: Additional patterns to reject.

    Returns:
        Validation result with is_valid and reason if invalid.
    """
    red_flag_config = RedFlagConfig(
        max_chars=max_chars,
        require_json=content_type == "json",
        json_required_fields=require_json_fields,
        require_diff_format=content_type == "diff",
    )

    # Add custom forbidden patterns
    if forbidden_patterns:
        red_flag_config.forbidden_patterns.extend(forbidden_patterns)

    is_valid, result, reason = validate_candidate_content(
        content, red_flag_config, content_type
    )

    return {
        "ok": is_valid,
        "is_valid": is_valid,
        "result": result.value,
        "reason": reason,
        "content_length": len(content),
        "content_type": content_type,
    }


# ============================================================================
# TOOL: zen_log_evidence - Log evidence to the chain
# ============================================================================

@mcp.tool()
def zen_log_evidence(
    evidence_type: Literal["observation", "hypothesis", "decision", "verification"],
    stage: str,
    content: Dict[str, Any],
    source: str,
    confidence: float = 1.0,
    linked_evidence_ids: List[str] = [],
) -> Dict[str, Any]:
    """
    Log evidence to the reasoning chain for auditability.

    Evidence-first approach: Every decision should link to concrete evidence.
    This enables post-hoc analysis of why decisions were made.

    Evidence types:
    - observation: Facts observed from code/logs
    - hypothesis: Proposed explanation for an issue
    - decision: Choice made with rationale
    - verification: Test/lint result

    Args:
        evidence_type: Type of evidence being logged.
        stage: Which workflow stage this evidence belongs to.
        content: The actual evidence content (structured dict).
        source: Where the evidence came from.
        confidence: How reliable is this evidence (0.0-1.0).
        linked_evidence_ids: IDs of related evidence to link.

    Returns:
        The evidence ID for future reference.
    """
    type_map = {
        "observation": EvidenceType.OBSERVATION,
        "hypothesis": EvidenceType.HYPOTHESIS,
        "decision": EvidenceType.DECISION,
        "verification": EvidenceType.VERIFICATION,
    }

    evidence_id = trace_store.log_evidence(
        evidence_type=type_map[evidence_type],
        stage=stage,
        content=content,
        source=source,
        confidence=confidence,
        links=linked_evidence_ids,
    )

    return {
        "ok": True,
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "stage": stage,
    }


# ============================================================================
# TOOL: zen_get_evidence_chain - Query the evidence chain
# ============================================================================

@mcp.tool()
def zen_get_evidence_chain(
    evidence_type: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Query the evidence chain for audit and analysis.

    Use to:
    - Review what evidence led to a decision
    - Trace back through the reasoning chain
    - Identify gaps in evidence

    Args:
        evidence_type: Filter by type (observation, hypothesis, decision, etc.)
        stage: Filter by workflow stage.
        limit: Maximum number of entries to return.

    Returns:
        List of evidence entries matching the filters.
    """
    type_map = {
        "observation": EvidenceType.OBSERVATION,
        "hypothesis": EvidenceType.HYPOTHESIS,
        "decision": EvidenceType.DECISION,
        "verification": EvidenceType.VERIFICATION,
        "red_flag": EvidenceType.RED_FLAG,
        "consensus": EvidenceType.CONSENSUS,
        "patch": EvidenceType.PATCH,
        "rollback": EvidenceType.ROLLBACK,
    }

    ev_type = type_map.get(evidence_type) if evidence_type else None
    chain = trace_store.get_evidence_chain(evidence_type=ev_type, stage=stage)

    # Limit and convert
    chain = chain[-limit:]

    return {
        "ok": True,
        "count": len(chain),
        "evidence": [e.to_dict() for e in chain],
        "red_flag_summary": trace_store.get_red_flag_summary(),
    }


# ============================================================================
# TOOL: zen_restore_from_backup - Restore files from backup
# ============================================================================

@mcp.tool()
def zen_restore_from_backup(
    backup_session: str,
    workspace_root: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Restore files from a backup session.

    Use when a patch introduced bugs and you need to rollback.

    Args:
        backup_session: The backup session ID (from zen_apply_patch result).
        workspace_root: Root directory for restoration.
        files: Optional list of specific files to restore.

    Returns:
        List of restored files and any failures.
    """
    root = workspace_root or os.getcwd()
    manager = WorkspaceManager(root)

    success, restored, failed = manager.restore_from_backup(backup_session, files)

    if success:
        trace_store.log_rollback(
            files_restored=restored,
            reason="Manual rollback via zen_restore_from_backup",
            stage="debug",
        )

    return {
        "ok": success,
        "restored_files": restored,
        "failed": failed,
        "backup_session": backup_session,
    }


# ============================================================================
# TOOL: zen_enter_stage - Enter a workflow stage (dynamic tool loading)
# ============================================================================

@mcp.tool()
def zen_enter_stage(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
) -> Dict[str, Any]:
    """
    Enter a workflow stage and load appropriate tools dynamically.

    MAKER Insight: "Minimize context overhead" - Only load tools needed for current stage.
    This reduces token usage and improves response quality.

    Tool loading by stage:
    - analyze: context packing, evidence logging
    - hypothesize: ensemble, voting, validation
    - implement: patch, verify
    - debug: verify, rollback, context
    - improve: ensemble, patch, verify

    Args:
        stage: Which stage to enter.

    Returns:
        Loaded tools list = available tools for this stage.
    """
    stage_map = {
        "analyze": StageType.ANALYZE,
        "hypothesize": StageType.HYPOTHESIZE,
        "implement": StageType.IMPLEMENT,
        "debug": StageType.DEBUG,
        "improve": StageType.IMPROVE,
    }

    stage_enum = stage_map.get(stage)
    if not stage_enum:
        return {"ok": False, "error": f"Unknown stage: {stage}"}

    loaded = skill_session.enter_stage(stage_enum)
    state = skill_session.get_state()

    return {
        "ok": True,
        "stage": stage,
        "loaded_tools": state["loaded_tools"],
        "newly_loaded": loaded,
        "available_skills": skill_session.get_available_skills(),
        "context_cost": state["context_cost"],
    }


# ============================================================================
# TOOL: zen_enter_skill - Enter a specific skill (fine-grained tool loading)
# ============================================================================

@mcp.tool()
def zen_enter_skill(
    skill_name: str,
) -> Dict[str, Any]:
    """
    Enter a specific skill for fine-grained tool loading.

    Skills are sub-tasks within a stage with specific tool requirements.
    This provides the most context-efficient tool loading.

    Example skills:
    - spec_extraction (analyze) - Extract I/O specs
    - edge_case_generation (analyze) - Generate edge cases
    - root_cause_analysis (hypothesize) - Find root causes
    - patch_generation (implement) - Generate patches
    - failure_classification (debug) - Classify errors
    - refactoring (improve) - Apply refactorings

    Args:
        skill_name: Name of the skill to enter.

    Returns:
        Skill info and loaded tools.
    """
    success, loaded = skill_session.enter_skill(skill_name)

    if not success:
        return {
            "ok": False,
            "error": f"Skill not found or invalid for current stage: {skill_name}",
            "available_skills": skill_session.get_available_skills(),
        }

    state = skill_session.get_state()

    return {
        "ok": True,
        "skill": skill_name,
        "stage": state["current_stage"],
        "loaded_tools": state["loaded_tools"],
        "newly_loaded": loaded,
        "micro_steps": skill_session.get_available_micro_steps(),
        "context_cost": state["context_cost"],
    }


# ============================================================================
# TOOL: zen_get_micro_steps - Get micro-steps for current stage/skill
# ============================================================================

@mcp.tool()
def zen_get_micro_steps(
    stage: Optional[Literal["analyze", "hypothesize", "implement", "debug", "improve"]] = None,
) -> Dict[str, Any]:
    """
    Get available micro-steps for MAKER-style decomposition.

    MAKER Insight: "Maximal Agentic Decomposition" - Break tasks into atomic steps.
    Each micro-step is small enough for voting to be effective.

    Micro-step types by stage:
    - Analyze: s1_spec_extract, s2_edge_case, s3_mre
    - Hypothesize: h1_root_cause, h2_verification
    - Implement: c1_minimal_patch, c2_compile_check
    - Debug: d1_failure_label, d2_next_experiment
    - Improve: r1_refactor, r2_perf

    Args:
        stage: Optional stage filter. Uses current stage if not specified.

    Returns:
        List of micro-step specifications.
    """
    if stage:
        stage_map = {
            "analyze": StageType.ANALYZE,
            "hypothesize": StageType.HYPOTHESIZE,
            "implement": StageType.IMPLEMENT,
            "debug": StageType.DEBUG,
            "improve": StageType.IMPROVE,
        }
        stage_enum = stage_map.get(stage)
        if not stage_enum:
            return {"ok": False, "error": f"Unknown stage: {stage}"}

        micro_steps = get_all_micro_steps_for_stage(stage_enum)
    else:
        # Get from current skill session
        state = skill_session.get_state()
        if state["current_stage"]:
            stage_enum = StageType(state["current_stage"])
            micro_steps = get_all_micro_steps_for_stage(stage_enum)
        else:
            # Return all micro-steps
            micro_steps = list(MICRO_STEP_SPECS.values())

    return {
        "ok": True,
        "micro_steps": [
            {
                "type": spec.step_type.value,
                "stage": spec.stage.value,
                "description": spec.description,
                "default_k": spec.default_k,
                "has_oracle": spec.has_oracle,
                "required_tools": spec.required_tools,
                "red_flag_rules": spec.red_flag_rules,
            }
            for spec in micro_steps
        ],
        "count": len(micro_steps),
    }


# ============================================================================
# TOOL: zen_vote_micro_step - MAKER-style voting on a micro-step
# ============================================================================

@mcp.tool()
def zen_vote_micro_step(
    step_type: Literal[
        "s1_spec_extract", "s2_edge_case", "s3_mre",
        "h1_root_cause", "h2_verification",
        "c1_minimal_patch", "c2_compile_check",
        "d1_failure_label", "d2_next_experiment",
        "r1_refactor", "r2_perf",
    ],
    prompt: str,
    context: str = "",
    k: Optional[int] = None,
    max_rounds: int = 15,
    providers: List[str] = ["codex", "gemini"],
) -> Dict[str, Any]:
    """
    Run MAKER-style first-to-ahead-by-k voting on a micro-step.

    MAKER Paper Insight:
    - Step-level voting with small k dramatically reduces error rate
    - Red-flagging rejects format errors BEFORE voting
    - Different step types have different default k values

    This is THE core mechanism for error correction in long-horizon tasks.

    Args:
        step_type: Type of micro-step (determines validation rules).
        prompt: The task prompt for this micro-step.
        context: Additional context to include.
        k: Votes ahead needed to win. Uses step-specific default if not set.
        max_rounds: Maximum voting rounds.
        providers: Which LLM providers to use (cycled through).

    Returns:
        VoteResult with winner content, confidence, and voting trace.
    """
    # Get step specification
    step_enum = MicroStepType(step_type)
    spec = MICRO_STEP_SPECS.get(step_enum)

    if not spec:
        return {"ok": False, "error": f"Unknown step type: {step_type}"}

    # Use step-specific k if not provided
    effective_k = k if k is not None else spec.default_k

    # Filter to available providers
    available_providers = registry.list_providers()
    valid_providers = [p for p in providers if p in available_providers]
    if not valid_providers:
        return {"ok": False, "error": f"No valid providers. Available: {available_providers}"}

    # Build full prompt
    full_prompt = prompt
    if context:
        full_prompt = f"## Context\n{context}\n\n## Task\n{prompt}"

    # Provider index for round-robin
    provider_idx = [0]

    def sample_fn():
        """Sample from providers in round-robin fashion."""
        provider = valid_providers[provider_idx[0] % len(valid_providers)]
        provider_idx[0] += 1

        response = registry.run(provider, full_prompt)
        return response.stdout, provider, response.stdout

    # Create vote step with step-specific red-flagger
    vote_step = VoteStep(
        k=effective_k,
        max_rounds=max_rounds,
        red_flagger=red_flagger,
    )

    # Run voting
    result = vote_step.vote(sample_fn, step_type=step_enum)

    # Log to trace store
    trace_store.log_evidence(
        evidence_type=EvidenceType.CONSENSUS,
        stage=spec.stage.value,
        content={
            "step_type": step_type,
            "winner": result.winner.content if result.winner else None,
            "converged": result.converged,
            "total_rounds": result.total_rounds,
            "vote_distribution": dict(result.vote_distribution),
        },
        source="zen_vote_micro_step",
        confidence=result.final_margin / max(result.total_samples, 1) if result.winner else 0,
        links=[],
    )

    return {
        "ok": result.winner is not None,
        "winner": result.winner.content if result.winner else None,
        "winner_provider": result.winner.provider if result.winner else None,
        "converged": result.converged,
        "final_margin": result.final_margin,
        "total_rounds": result.total_rounds,
        "total_samples": result.total_samples,
        "red_flagged_count": result.red_flagged_count,
        "vote_distribution": dict(result.vote_distribution),
        "step_type": step_type,
        "k_used": effective_k,
        "has_oracle": spec.has_oracle,
    }


# ============================================================================
# TOOL: zen_calibrate - Calibrate voting parameters
# ============================================================================

@mcp.tool()
def zen_calibrate(
    step_type: Literal[
        "s1_spec_extract", "s2_edge_case", "s3_mre",
        "h1_root_cause", "h2_verification",
        "c1_minimal_patch", "c2_compile_check",
        "d1_failure_label", "d2_next_experiment",
        "r1_refactor", "r2_perf",
    ],
    test_prompt: str,
    oracle_command: Optional[str] = None,
    target_success_rate: float = 0.99,
    estimated_total_steps: int = 100,
    num_samples: int = 10,
    providers: List[str] = ["codex", "gemini"],
) -> Dict[str, Any]:
    """
    Calibrate voting parameters (k) for a step type.

    MAKER Paper Insight:
    - Different step types have different accuracy p
    - k should be calibrated based on p and target success rate
    - With proper calibration, even low-accuracy steps can achieve high overall success

    Run this before a long workflow to optimize k for each step type.

    Args:
        step_type: Which micro-step type to calibrate.
        test_prompt: A representative prompt for sampling.
        oracle_command: Optional command to verify correctness.
        target_success_rate: Desired overall workflow success rate.
        estimated_total_steps: Expected total steps in workflow.
        num_samples: Number of samples for estimation.
        providers: Which providers to sample from.

    Returns:
        CalibrationResult with recommended k and accuracy estimates.
    """
    step_enum = MicroStepType(step_type)
    spec = MICRO_STEP_SPECS.get(step_enum)

    if not spec:
        return {"ok": False, "error": f"Unknown step type: {step_type}"}

    # Filter providers
    available_providers = registry.list_providers()
    valid_providers = [p for p in providers if p in available_providers]
    if not valid_providers:
        return {"ok": False, "error": "No valid providers"}

    provider_idx = [0]

    def sample_fn():
        provider = valid_providers[provider_idx[0] % len(valid_providers)]
        provider_idx[0] += 1
        response = registry.run(provider, test_prompt)
        return response.stdout, provider, response.stdout

    # Oracle function
    def oracle_fn(content: str) -> bool:
        if not oracle_command:
            # No oracle - assume valid non-red-flagged responses are correct
            return True
        # Run oracle command
        engine = VerificationEngine()
        result = engine.run(oracle_command, VerificationType.CUSTOM)
        return result.passed

    # Run calibration
    result = calibrator.calibrate(
        sample_fn=sample_fn,
        oracle_fn=oracle_fn,
        total_steps=estimated_total_steps,
        target_success_rate=target_success_rate,
        step_type=step_enum,
        num_samples=num_samples,
    )

    return {
        "ok": True,
        "step_type": step_type,
        "estimated_p": result.estimated_p,
        "estimated_v": result.estimated_v,
        "recommended_k": result.recommended_k,
        "target_success_rate": result.target_success_rate,
        "expected_cost_multiplier": result.expected_cost_multiplier,
        "calibration_samples": result.calibration_samples,
        "insight": (
            f"With p={result.estimated_p:.2f} and k={result.recommended_k}, "
            f"expected {result.total_steps_estimate} steps can achieve "
            f"{result.target_success_rate*100:.1f}% overall success rate."
        ),
    }


# ============================================================================
# TOOL: zen_red_flag_check - Check content for red flags
# ============================================================================

@mcp.tool()
def zen_red_flag_check(
    content: str,
    step_type: Optional[Literal[
        "s1_spec_extract", "s2_edge_case", "s3_mre",
        "h1_root_cause", "h2_verification",
        "c1_minimal_patch", "c2_compile_check",
        "d1_failure_label", "d2_next_experiment",
        "r1_refactor", "r2_perf",
    ]] = None,
    rules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check content for MAKER-style red flags.

    MAKER Paper Insight: "Format errors signal reasoning errors"
    - Don't try to repair malformed responses
    - Discard and resample instead
    - This reduces CORRELATED errors, not just average errors

    Red flag types:
    - too_long: Response exceeds max length (indicates rambling)
    - too_short: Response too short (incomplete)
    - hedging: Contains hedging language ("I'm not sure")
    - missing_fields: Missing required JSON fields
    - invalid_diff: Invalid diff format
    - multi_file: Patch affects too many files
    - dangerous_code: Contains dangerous patterns
    - dangerous_command: Contains dangerous commands
    - forbidden_file: References forbidden files (.env, secrets)

    Args:
        content: Content to validate.
        step_type: Optional step type for step-specific rules.
        rules: Optional explicit rules to check.

    Returns:
        Red flag result with is_flagged status and reasons.
    """
    step_enum = MicroStepType(step_type) if step_type else None
    result = red_flagger.validate(content, step_enum, rules)

    return {
        "ok": True,
        "is_flagged": result.is_flagged,
        "should_discard": result.is_flagged,  # MAKER: discard, don't repair
        "reasons": [r.value for r in result.reasons],
        "details": result.details,
        "content_length": len(content),
        "step_type": step_type,
        "recommendation": (
            "DISCARD and resample - format errors correlate with reasoning errors"
            if result.is_flagged else "Content passes red-flag validation"
        ),
    }


# ============================================================================
# TOOL: zen_get_loaded_tools - Get currently loaded tools
# ============================================================================

@mcp.tool()
def zen_get_loaded_tools() -> Dict[str, Any]:
    """
    Get the list of currently loaded tools.

    Use to understand what tools are available in current context.
    Tools are dynamically loaded based on current stage/skill.

    Returns:
        List of loaded tool names and context cost.
    """
    state = skill_session.get_state()

    return {
        "ok": True,
        "loaded_tools": state["loaded_tools"],
        "current_stage": state["current_stage"],
        "current_skill": state["current_skill"],
        "context_cost": state["context_cost"],
        "max_tools": tool_registry.max_tools,
        "usage_stats": tool_registry.get_usage_stats(),
    }


# ============================================================================
# TOOL: zen_recommend_tools - Get recommended tools for a task
# ============================================================================

@mcp.tool()
def zen_recommend_tools(
    task_description: str,
) -> Dict[str, Any]:
    """
    Get recommended tools for a task description.

    Analyzes the task and recommends which tools to load.
    Useful for bootstrapping before entering a specific stage.

    Args:
        task_description: Description of what you want to accomplish.

    Returns:
        Recommended tools and suggested stage.
    """
    recommended = get_recommended_tools_for_task(task_description, skill_manifest)

    # Determine likely stage
    task_lower = task_description.lower()
    suggested_stage = "analyze"  # Default

    if any(w in task_lower for w in ["fix", "implement", "add", "patch"]):
        suggested_stage = "implement"
    elif any(w in task_lower for w in ["debug", "error", "fail", "broken"]):
        suggested_stage = "debug"
    elif any(w in task_lower for w in ["why", "cause", "hypothesis"]):
        suggested_stage = "hypothesize"
    elif any(w in task_lower for w in ["refactor", "improve", "optimize"]):
        suggested_stage = "improve"

    return {
        "ok": True,
        "recommended_tools": recommended,
        "suggested_stage": suggested_stage,
        "tool_count": len(recommended),
        "hint": f"Use zen_enter_stage('{suggested_stage}') to load these tools",
    }


# ============================================================================
# TOOL: zen_exit_stage - Exit current stage and unload tools
# ============================================================================

@mcp.tool()
def zen_exit_stage() -> Dict[str, Any]:
    """
    Exit current stage and unload non-core tools.

    Call this when transitioning between stages to minimize context.
    Only core tools (zen_list_providers, zen_get_skill, zen_workflow_state)
    remain loaded.

    Returns:
        Updated tool state after unloading.
    """
    skill_session.exit_stage()
    state = skill_session.get_state()

    return {
        "ok": True,
        "message": "Stage exited, tools unloaded to core set",
        "loaded_tools": state["loaded_tools"],
        "context_cost": state["context_cost"],
    }


# ============================================================================
# TOOL: zen_classify_task - Analyze task structure for architecture selection
# ============================================================================

@mcp.tool()
def zen_classify_task(
    task_description: str,
    code_context: Optional[str] = None,
    error_logs: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify task structure to determine optimal coordination topology.

    Paper Rule A: "Domain/task structure dependency is absolute"
    This tool analyzes the task to extract features that determine
    whether to use MAS (multi-agent) or SAS (single-agent).

    Key features extracted:
    - decomposability_score: Can task be split into parallel subtasks?
    - sequential_dependency_score: How much does each step depend on previous?
    - tool_complexity: How many/complex tools are needed?

    Use this FIRST in Stage 1 (Analyze) to inform architecture selection
    for subsequent stages.

    Args:
        task_description: Description of the task/problem.
        code_context: Optional code snippets for context.
        error_logs: Optional error logs or stack traces.

    Returns:
        TaskStructureFeatures with scores and recommendations.
    """
    features = task_classifier.classify(
        task_description=task_description,
        code_context=code_context,
        error_logs=error_logs,
    )

    # Get architecture recommendation
    decision = architecture_engine.select_architecture(features)

    return {
        "ok": True,
        "features": {
            "decomposability_score": features.decomposability_score,
            "sequential_dependency_score": features.sequential_dependency_score,
            "tool_complexity": features.tool_complexity,
            "domain": features.domain,
            "baseline_single_agent_success": features.baseline_single_agent_success,
        },
        "metadata": features.metadata,
        "recommended_topology": decision.topology.value,
        "topology_confidence": decision.confidence,
        "topology_rationale": decision.rationale,
        "paper_rule_applied": (
            "Rule B: Sequential → SAS" if features.sequential_dependency_score > 0.7
            else "Rule B: Decomposable → MAS" if features.decomposability_score > 0.6
            else "Rule A: Domain-specific default"
        ),
    }


# ============================================================================
# TOOL: zen_select_architecture - Select coordination topology for a stage
# ============================================================================

@mcp.tool()
def zen_select_architecture(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
    decomposability_score: float = 0.5,
    sequential_dependency_score: float = 0.5,
    tool_complexity: float = 0.3,
    force_topology: Optional[Literal["sas", "mas_independent", "mas_centralized"]] = None,
) -> Dict[str, Any]:
    """
    Select the optimal coordination topology for a workflow stage.

    Paper Rules Applied:
    - Rule A: Architecture depends on task structure
    - Rule B: Decomposable → MAS, Sequential → SAS
    - Rule C: Coordination overhead is a cost to minimize
    - Rule D: Use calibration data when available

    The selected topology determines:
    - How many agents to use
    - How they communicate
    - When to fall back to simpler topology

    Args:
        stage: Current workflow stage.
        decomposability_score: 0-1, how parallelizable is the task.
        sequential_dependency_score: 0-1, how sequential are dependencies.
        tool_complexity: 0-1, how complex is tool usage.
        force_topology: Optional override ("sas", "mas_independent", "mas_centralized").

    Returns:
        CoordinationDecision with topology, parameters, and fallback plan.
    """
    features = TaskStructureFeatures(
        decomposability_score=decomposability_score,
        sequential_dependency_score=sequential_dependency_score,
        tool_complexity=tool_complexity,
    )

    # Parse force_topology
    forced = None
    if force_topology:
        topology_map = {
            "sas": CoordinationTopology.SAS,
            "mas_independent": CoordinationTopology.MAS_INDEPENDENT,
            "mas_centralized": CoordinationTopology.MAS_CENTRALIZED,
        }
        forced = topology_map.get(force_topology)

    decision = architecture_engine.select_architecture(
        features=features,
        stage=stage,
        force_topology=forced,
    )

    # Get stage-specific recommendations
    stage_rec = architecture_engine.get_stage_recommendation(stage, features)

    return {
        "ok": True,
        "topology": decision.topology.value,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "parameters": {
            "max_agents": decision.max_agents,
            "max_rounds": decision.max_rounds,
            "max_messages_per_agent": decision.max_messages_per_agent,
            "overhead_threshold": decision.overhead_threshold,
        },
        "fallback_topology": decision.fallback_topology.value if decision.fallback_topology else None,
        "stage_recommendations": stage_rec,
        "paper_insight": (
            "Debug stage forced to SAS - paper shows MAS degrades in sequential tasks"
            if stage == "debug" else
            f"Stage '{stage}' using {decision.topology.value} based on task features"
        ),
    }


# ============================================================================
# TOOL: zen_check_degradation - Check if should fall back to simpler topology
# ============================================================================

@mcp.tool()
def zen_check_degradation(
    current_topology: Literal["sas", "mas_independent", "mas_centralized"],
    total_messages: int = 0,
    total_rounds: int = 0,
    successes: int = 0,
    failures: int = 0,
    redundancy_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Check if coordination should degrade to a simpler topology.

    Paper Rule C: "Coordination overhead is a first-class cost function"
    When MAS isn't improving results, automatically fall back to SAS.

    Degradation triggers:
    - High coordination overhead without success improvement
    - Error amplification > 1.0 (MAS making more errors than SAS would)
    - High redundancy (agents producing identical outputs)
    - Consecutive format errors

    Args:
        current_topology: Current coordination topology.
        total_messages: Total messages sent in coordination.
        total_rounds: Total coordination rounds.
        successes: Number of successful outcomes.
        failures: Number of failed outcomes.
        redundancy_rate: 0-1, similarity of agent outputs.

    Returns:
        Degradation decision with new topology if needed.
    """
    # Build metrics
    metrics = CoordinationMetrics(
        total_messages=total_messages,
        total_rounds=total_rounds,
        successes=successes,
        failures=failures,
        redundancy_rate=redundancy_rate,
    )

    # Compute derived metrics
    total = successes + failures
    if total > 0:
        sas_failure_rate = 0.3  # Baseline assumption
        metrics.error_amplification = estimate_error_amplification(
            failures, total, sas_failure_rate
        )

    # Get decision for threshold
    topology_map = {
        "sas": CoordinationTopology.SAS,
        "mas_independent": CoordinationTopology.MAS_INDEPENDENT,
        "mas_centralized": CoordinationTopology.MAS_CENTRALIZED,
    }
    current = topology_map.get(current_topology, CoordinationTopology.SAS)

    decision = CoordinationDecision(
        topology=current,
        confidence=0.5,
        rationale="Current topology",
        overhead_threshold=2.0,
    )

    should_degrade, new_topology, reason = architecture_engine.should_degrade(
        current, metrics, decision
    )

    # Also check format error counter
    format_degrade, format_reason = degradation_strategy.should_degrade()
    if format_degrade:
        should_degrade = True
        reason = format_reason

    return {
        "ok": True,
        "should_degrade": should_degrade,
        "current_topology": current_topology,
        "recommended_topology": new_topology.value if new_topology else current_topology,
        "reason": reason,
        "metrics": {
            "error_amplification": metrics.error_amplification,
            "redundancy_rate": redundancy_rate,
            "success_rate": successes / total if total > 0 else 0,
            "coordination_overhead": total_messages / max(total, 1),
        },
        "action": (
            f"DEGRADE to {new_topology.value}: {reason}" if should_degrade
            else "CONTINUE with current topology"
        ),
    }


# ============================================================================
# TOOL: zen_record_coordination_result - Record result for calibration
# ============================================================================

@mcp.tool()
def zen_record_coordination_result(
    topology: Literal["sas", "mas_independent", "mas_centralized"],
    success: bool,
    tokens_used: int = 0,
    messages_sent: int = 0,
    rounds: int = 1,
    outputs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Record coordination result for calibration (Rule D).

    Paper Rule D: "Model family calibration is necessary"
    By recording results, the system learns which topology works best
    for your specific codebase and task types.

    Call this after each coordination attempt to build calibration data.

    Args:
        topology: Which topology was used.
        success: Whether the coordination succeeded.
        tokens_used: Tokens consumed.
        messages_sent: Messages sent between agents.
        rounds: Coordination rounds.
        outputs: Optional list of agent outputs (for redundancy calculation).

    Returns:
        Updated statistics for the topology.
    """
    topology_map = {
        "sas": CoordinationTopology.SAS,
        "mas_independent": CoordinationTopology.MAS_INDEPENDENT,
        "mas_centralized": CoordinationTopology.MAS_CENTRALIZED,
    }
    topo = topology_map.get(topology, CoordinationTopology.SAS)

    # Start tracking
    metrics_tracker.start_run()

    # Record metrics
    for _ in range(messages_sent):
        metrics_tracker.record_message(tokens_used // max(messages_sent, 1))

    for i in range(rounds):
        metrics_tracker.record_round(is_synthesis=(i == rounds - 1))

    if success:
        metrics_tracker.record_success()
    else:
        metrics_tracker.record_failure()
        degradation_strategy.record_format_error()  # Track for degradation

    # Record redundancy if outputs provided
    if outputs and len(outputs) > 1:
        redundancy = compute_redundancy(outputs)
        metrics_tracker.record_redundancy([redundancy])

    # End tracking
    final_metrics = metrics_tracker.end_run(topo)

    # Get updated stats
    stats = metrics_tracker.get_topology_stats(topo)

    return {
        "ok": True,
        "recorded": {
            "topology": topology,
            "success": success,
            "tokens_used": tokens_used,
            "messages_sent": messages_sent,
            "rounds": rounds,
        },
        "run_metrics": {
            "coordination_overhead": final_metrics.coordination_overhead,
            "message_density": final_metrics.message_density,
            "redundancy_rate": final_metrics.redundancy_rate,
        },
        "topology_stats": {
            "total_runs": stats["count"],
            "success_rate": stats["success_rate"],
            "avg_overhead": stats["avg_overhead"],
            "avg_tokens": stats["avg_tokens"],
        },
        "calibration_hint": (
            f"After {stats['count']} runs, {topology} has {stats['success_rate']*100:.1f}% success rate"
        ),
    }


# ============================================================================
# TOOL: zen_get_coordination_stats - Get calibration statistics
# ============================================================================

@mcp.tool()
def zen_get_coordination_stats() -> Dict[str, Any]:
    """
    Get coordination calibration statistics.

    Paper Rule D: Use calibration data to inform architecture selection.

    Returns statistics for each topology:
    - Success rate
    - Average overhead
    - Average token usage
    - Number of samples

    Use this to understand which topologies work best for your tasks.

    Returns:
        Statistics per topology and recommendations.
    """
    stats = {}
    for topo in [CoordinationTopology.SAS, CoordinationTopology.MAS_INDEPENDENT, CoordinationTopology.MAS_CENTRALIZED]:
        topo_stats = metrics_tracker.get_topology_stats(topo)
        stats[topo.value] = {
            "success_rate": topo_stats["success_rate"],
            "avg_overhead": topo_stats["avg_overhead"],
            "avg_tokens": topo_stats["avg_tokens"],
            "sample_count": topo_stats["count"],
        }

    # Find best topology
    best_topo = None
    best_score = -1
    for topo_name, topo_stats in stats.items():
        if topo_stats["sample_count"] >= 3:
            score = topo_stats["success_rate"] - (topo_stats["avg_overhead"] * 0.1)
            if score > best_score:
                best_score = score
                best_topo = topo_name

    return {
        "ok": True,
        "topology_stats": stats,
        "best_topology": best_topo,
        "best_topology_score": best_score if best_topo else None,
        "total_samples": sum(s["sample_count"] for s in stats.values()),
        "recommendation": (
            f"Based on {sum(s['sample_count'] for s in stats.values())} samples, "
            f"'{best_topo}' performs best with {stats[best_topo]['success_rate']*100:.1f}% success rate"
            if best_topo else "Not enough data for recommendation (need 3+ samples per topology)"
        ),
    }


# ============================================================================
# TOOL: zen_get_stage_strategy - Get recommended strategy for a stage
# ============================================================================

@mcp.tool()
def zen_get_stage_strategy(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
) -> Dict[str, Any]:
    """
    Get the recommended coordination strategy for a workflow stage.

    Returns stage-specific guidance based on paper insights:
    - analyze: Parallel info gathering, score-based selection
    - hypothesize: Parallel hypothesis gen, falsifiability scoring
    - implement: Parallel patch gen, TEST-FIRST selection
    - debug: SEQUENTIAL (SAS), minimize coordination
    - improve: Parallel review, skill extraction

    Args:
        stage: The workflow stage.

    Returns:
        Detailed strategy with topology, voting mode, and red-flag rules.
    """
    features = TaskStructureFeatures()  # Default features
    strategy = architecture_engine.get_stage_recommendation(stage, features)

    # Add paper insights
    paper_insights = {
        "analyze": (
            "Stage 1 uses parallel independent generation. "
            "Multiple models extract observations, orchestrator synthesizes. "
            "Voting is score-based (not test-based)."
        ),
        "hypothesize": (
            "Stage 2 uses parallel hypothesis generation. "
            "Selection criteria: falsifiability > low experiment cost > explanation coverage. "
            "Paper: 'MAS excels at decomposable information generation tasks.'"
        ),
        "implement": (
            "Stage 3 generates patches in parallel, then TEST-FIRST selection. "
            "Voting is secondary to test results. "
            "Paper: 'Tests are stronger judges than consensus for code.'"
        ),
        "debug": (
            "Stage 4 MUST use SAS (single-agent). "
            "Paper: 'Sequential tasks with state accumulation degrade significantly with MAS.' "
            "Max 5 iterations before human escalation."
        ),
        "improve": (
            "Stage 5 extracts reusable patterns from successful fixes. "
            "Parallel review is OK for suggestions. "
            "Output: skill templates, policy updates, playbook improvements."
        ),
    }

    return {
        "ok": True,
        "stage": stage,
        "strategy": strategy,
        "paper_insight": paper_insights.get(stage, ""),
        "key_rules": {
            "topology": strategy.get("topology", "sas"),
            "ensemble_strategy": strategy.get("ensemble_strategy", "single_agent"),
            "voting_mode": strategy.get("voting_mode", "test_first"),
            "red_flags": strategy.get("red_flag_rules", []),
        },
        "warning": (
            "DEBUG STAGE: Do NOT use multi-agent coordination here. "
            "Paper shows 39-70% degradation with MAS in sequential tasks."
            if stage == "debug" else None
        ),
    }


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Zen Skills MCP Server...")
    logger.info(f"Available providers: {registry.list_providers()}")
    logger.info(f"Trace directory: {config.tracing.trace_dir}")
    logger.info(f"Max tools per stage: {tool_registry.max_tools}")
    logger.info(f"Disabled tools: {disabled_tools}")
    mcp.run()
