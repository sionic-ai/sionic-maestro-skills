"""
Maestro MCP Server

A Model Context Protocol server implementing multi-LLM orchestration
with a "centralized consult" architecture.

Key Features:
- Multi-LLM orchestration (Claude/Codex/Gemini)
- 5-stage workflow (Analyze → Hypothesize → Implement → Debug → Improve)
- Measured coordination based on "Towards a Science of Scaling Agent Systems"
- MAKER-style error correction (voting, red-flagging, calibration)
- Dynamic tool loading for context optimization

Architecture: Centralized Consult Pattern
- Claude Code = Maestro/Orchestrator (tool execution)
- Codex/Gemini/Claude CLI = Consultants (text advice only)

Named "Maestro" because like a conductor orchestrating an orchestra,
Claude Code coordinates multiple models to produce harmonious output.
"""

from fastmcp import FastMCP
from typing import List, Dict, Optional, Literal, Any
import asyncio
import logging
import os
import json
from pathlib import Path

# Enable nested event loops for MCP server compatibility
import nest_asyncio
nest_asyncio.apply()

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from maestro.config import MaestroConfig
from maestro.providers import ProviderRegistry, ProviderResponse
from maestro.context import ContextPacker, PackingConfig
from maestro.workflow import WorkflowEngine, Stage, StageContext, WorkflowRunner
from maestro.selection import (
    SelectionEngine, SelectionMode, Candidate, TestSignal, LintSignal,
    RedFlagConfig, validate_candidate_content, RedFlagResult
)
from maestro.tracing import TraceStore, Metrics, EvidenceType
from maestro.verify import VerificationEngine, VerificationType, VerificationResult
from maestro.workspace import WorkspaceManager, WorkspaceConfig, PatchResult
from maestro.consensus import ConsensusEngine, ConsensusConfig, RedFlagConfig as ConsensusRedFlagConfig

# MAKER-style modules
from maestro.maker import (
    StageType, MicroStepType, MicroStepSpec, MICRO_STEP_SPECS,
    RedFlagger, RedFlaggerConfig, RedFlagResult as MakerRedFlagResult,
    VoteStep, VoteResult, VoteCandidate,
    Calibrator, CalibrationResult,
    get_tools_for_step, get_tools_for_stage, get_all_micro_steps_for_stage,
    get_default_k_for_step, step_has_oracle,
)
from maestro.skills import (
    SkillManifest, SkillDefinition, DynamicToolRegistry, ToolLoadState,
    SkillSession, SkillLoader, create_skill_session, get_recommended_tools_for_task,
)
from maestro.coordination import (
    CoordinationTopology, TaskStructureFeatures, CoordinationDecision,
    CoordinationMetrics, MetricsTracker, ArchitectureSelectionEngine,
    TaskStructureClassifier, DegradationStrategy, CoordinationPolicy,
    compute_redundancy, estimate_error_amplification,
)

# Human-in-the-Loop module
from maestro.human_loop import (
    HumanLoopManager, ApprovalRequest, ApprovalStatus, StageReport,
    ReviewQuestion, ReviewPriority, format_approval_request_for_display,
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

# Load SKILLS.md for on-demand documentation
SKILLS_MD_PATH = BASE_DIR / "SKILLS.md"
SKILLS_MD_CONTENT = SKILLS_MD_PATH.read_text() if SKILLS_MD_PATH.exists() else ""


def parse_skills_md(content: str) -> Dict[str, str]:
    """Parse SKILLS.md into topic -> content sections."""
    sections = {}
    current_topic = None
    current_content = []

    for line in content.split("\n"):
        # Match "## Tool: name" or "## Topic: name"
        if line.startswith("## Tool: ") or line.startswith("## Topic: "):
            if current_topic:
                sections[current_topic] = "\n".join(current_content).strip()
            current_topic = line.split(": ", 1)[1].strip()
            current_content = [line]
        elif line.startswith("---") and current_topic:
            sections[current_topic] = "\n".join(current_content).strip()
            current_topic = None
            current_content = []
        elif current_topic:
            current_content.append(line)

    if current_topic:
        sections[current_topic] = "\n".join(current_content).strip()

    return sections


SKILLS_SECTIONS = parse_skills_md(SKILLS_MD_CONTENT)

# Configure logging
logging.basicConfig(
    level=os.getenv("MAESTRO_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("maestro.server")

# Initialize configuration
config = MaestroConfig.from_env()

# Initialize MCP server
mcp = FastMCP(
    "maestro-mcp",
    instructions=(
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
disabled_tools = set(os.getenv("MAESTRO_DISABLED_TOOLS", "").split(",")) if os.getenv("MAESTRO_DISABLED_TOOLS") else set()
tool_registry = DynamicToolRegistry(
    manifest=skill_manifest,
    max_tools=int(os.getenv("MAESTRO_MAX_TOOLS", "10")),
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

# Initialize Human-in-the-Loop manager
human_loop_manager = HumanLoopManager()


# ============================================================================
# TOOL: maestro_help - On-demand documentation (token-efficient)
# ============================================================================

@mcp.tool()
def maestro_help(
    topic: str,
) -> Dict[str, Any]:
    """Get detailed guidance for any maestro tool or topic. Call this before using unfamiliar tools."""
    # Normalize topic name
    topic_normalized = topic.lower().replace("maestro_", "").strip()

    # Check exact match first
    if topic in SKILLS_SECTIONS:
        return {
            "ok": True,
            "topic": topic,
            "content": SKILLS_SECTIONS[topic],
        }

    # Check normalized match
    for key in SKILLS_SECTIONS:
        if key.lower().replace("maestro_", "") == topic_normalized:
            return {
                "ok": True,
                "topic": key,
                "content": SKILLS_SECTIONS[key],
            }

    # List available topics
    available = sorted(SKILLS_SECTIONS.keys())
    return {
        "ok": False,
        "error": f"Topic '{topic}' not found",
        "available_topics": available,
        "hint": "Try tool names like 'maestro_consult' or topics like 'workflow', 'paper_insights'",
    }


# ============================================================================
# TOOL: maestro_consult - Single provider consultation
# ============================================================================

@mcp.tool()
def maestro_consult(
    prompt: str,
    provider: Literal["codex", "gemini", "claude"] = "codex",
    model: Optional[str] = None,
    context_files: Optional[List[str]] = None,
    context_facts: Optional[List[str]] = None,
    context_errors: Optional[List[str]] = None,
    context_constraints: Optional[List[str]] = None,
    stage: Optional[str] = None,
    timeout_sec: int = 1800,  # 30 min default - generous timeout with liveness checking
) -> Dict[str, Any]:
    """Consult external LLM CLI as sub-agent. Call maestro_help('maestro_consult') for details."""
    # Initialize mutable defaults (Python footgun: = [] is shared between calls)
    context_files = context_files or []
    context_facts = context_facts or []
    context_errors = context_errors or []
    context_constraints = context_constraints or []

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
# TOOL: maestro_ensemble_generate - Multi-provider candidate generation
# ============================================================================

@mcp.tool()
def maestro_ensemble_generate(
    task: str,
    providers: Optional[List[str]] = None,
    context_files: Optional[List[str]] = None,
    context_facts: Optional[List[str]] = None,
    context_errors: Optional[List[str]] = None,
    n_per_provider: int = 1,
    timeout_sec: int = 1800,  # 30 min default - generous timeout with liveness checking
) -> Dict[str, Any]:
    """Generate multiple candidates using different LLMs. Call maestro_help('maestro_ensemble_generate') for details."""
    # Initialize mutable defaults
    providers = providers or ["codex", "gemini"]
    context_files = context_files or []
    context_facts = context_facts or []
    context_errors = context_errors or []

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
# TOOL: maestro_select_best - Select best candidate
# ============================================================================

@mcp.tool()
def maestro_select_best(
    candidates: List[Dict[str, Any]],
    mode: Literal["tests_first", "llm_judge", "hybrid"] = "tests_first",
    test_results: Optional[List[Dict[str, Any]]] = None,
    lint_results: Optional[List[Dict[str, Any]]] = None,
    judge_provider: Literal["claude", "codex", "gemini"] = "claude",
    criteria: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select best candidate from ensemble. Call maestro_help('maestro_select_best') for details."""
    # Initialize mutable defaults
    criteria = criteria or ["correctness", "safety", "completeness"]

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
# TOOL: maestro_pack_context - Context engineering utility
# ============================================================================

@mcp.tool()
def maestro_pack_context(
    files: Optional[List[str]] = None,
    facts: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    stage: Optional[str] = None,
    max_chars: int = 40000,
) -> Dict[str, Any]:
    """Pack context with smart truncation. Call maestro_help('maestro_pack_context') for details."""
    # Initialize mutable defaults
    files = files or []
    facts = facts or []
    errors = errors or []
    constraints = constraints or []

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
# TOOL: maestro_workflow_state - Get current workflow state
# ============================================================================

@mcp.tool()
def maestro_workflow_state() -> Dict[str, Any]:
    """Get current workflow state and metrics. Call maestro_help('maestro_workflow_state') for details."""
    state = workflow_engine.get_workflow_state()
    metrics = Metrics.compute(trace_store)

    return {
        "ok": True,
        "workflow": state,
        "metrics": metrics.to_dict(),
        "summary": metrics.summary(),
    }


# ============================================================================
# TOOL: maestro_run_stage - Execute a workflow stage
# ============================================================================

@mcp.tool()
def maestro_run_stage(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
    task: str,
    context_files: Optional[List[str]] = None,
    context_facts: Optional[List[str]] = None,
    context_errors: Optional[List[str]] = None,
    providers: Optional[List[str]] = None,
    baseline_confidence: float = 0.0,
    timeout_sec: int = 1800,  # 30 min default - generous timeout with liveness checking
) -> Dict[str, Any]:
    """Execute a workflow stage. Call maestro_help('maestro_run_stage') for details."""
    # Initialize mutable defaults
    context_files = context_files or []
    context_facts = context_facts or []
    context_errors = context_errors or []

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
                timeout_sec=timeout_sec,
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
                timeout_sec=timeout_sec,
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
# TOOL: maestro_get_metrics - Get detailed metrics
# ============================================================================

@mcp.tool()
def maestro_get_metrics() -> Dict[str, Any]:
    """Get detailed paper-aligned metrics. Call maestro_help('maestro_get_metrics') for details."""
    metrics = Metrics.compute(trace_store)
    return {
        "ok": True,
        "metrics": metrics.to_dict(),
        "summary": metrics.summary(),
    }


# ============================================================================
# TOOL: maestro_list_providers - List available providers
# ============================================================================

@mcp.tool()
def maestro_list_providers() -> Dict[str, Any]:
    """List available CLI providers. Call maestro_help('maestro_list_providers') for details."""
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
# TOOL: maestro_get_skill - Get skill definition for a stage
# ============================================================================

@mcp.tool()
def maestro_get_skill(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
) -> Dict[str, Any]:
    """Get skill definition for a stage. Call maestro_help('maestro_get_skill') for details."""
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
# TOOL: maestro_get_role - Get role/persona prompt
# ============================================================================

@mcp.tool()
def maestro_get_role(
    role: Literal[
        "example_analyst",
        "hypothesis_scientist",
        "implementer",
        "debugger",
        "refiner",
        "judge",
    ],
) -> Dict[str, Any]:
    """Get role/persona prompt. Call maestro_help('maestro_get_role') for details."""
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
# TOOL: maestro_get_schema - Get output schema for validation
# ============================================================================

@mcp.tool()
def maestro_get_schema(
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
    """Get JSON schema for validation. Call maestro_help('maestro_get_schema') for details."""
    schema_content = load_output_schema(schema)
    if not schema_content:
        return {"ok": False, "error": f"Schema not found: {schema}"}

    return {
        "ok": True,
        "schema_name": schema,
        "schema": schema_content,
    }


# ============================================================================
# TOOL: maestro_consult_with_role - Consult with role-based prompting
# ============================================================================

@mcp.tool()
def maestro_consult_with_role(
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
    context_files: Optional[List[str]] = None,
    context_facts: Optional[List[str]] = None,
    context_errors: Optional[List[str]] = None,
    stage: Optional[str] = None,
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """Consult LLM with role-based prompting. Call maestro_help('maestro_consult_with_role') for details."""
    # Initialize mutable defaults
    context_files = context_files or []
    context_facts = context_facts or []
    context_errors = context_errors or []

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
# TOOL: maestro_get_coordination_policy - Get paper-aligned coordination rules
# ============================================================================

@mcp.tool()
def maestro_get_coordination_policy() -> Dict[str, Any]:
    """Get paper-based coordination policies. Call maestro_help('maestro_get_coordination_policy') for details."""
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
# TOOL: maestro_verify - Run verification (tests/lint/type-check)
# ============================================================================

@mcp.tool()
def maestro_verify(
    commands: List[Dict[str, Any]],
    cwd: Optional[str] = None,
    stop_on_failure: bool = False,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Run verification commands (tests, lint, type-check). Call maestro_help('maestro_verify') for details."""
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
# TOOL: maestro_apply_patch - Apply a unified diff patch safely
# ============================================================================

@mcp.tool()
def maestro_apply_patch(
    patch: str,
    workspace_root: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Apply unified diff patch safely with backups. Call maestro_help('maestro_apply_patch') for details."""
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
# TOOL: maestro_consensus_vote - MAKER-style micro-decision voting
# ============================================================================

@mcp.tool()
def maestro_consensus_vote(
    question: str,
    k: int = 3,
    max_rounds: int = 12,
    providers: Optional[List[str]] = None,
    require_json: bool = False,
    max_response_chars: int = 2000,
) -> Dict[str, Any]:
    """MAKER-style first-to-ahead-by-k voting for micro-decisions. Call maestro_help('maestro_consensus_vote') for details."""
    # Initialize mutable defaults
    providers = providers or ["codex", "gemini", "claude"]

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
# TOOL: maestro_validate_content - Red-flag validation for any content
# ============================================================================

@mcp.tool()
def maestro_validate_content(
    content: str,
    content_type: Literal["general", "diff", "json"] = "general",
    max_chars: int = 15000,
    require_json_fields: Optional[List[str]] = None,
    forbidden_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate content against MAKER red-flag criteria. Call maestro_help('maestro_validate_content') for details."""
    # Initialize mutable defaults
    require_json_fields = require_json_fields or []
    forbidden_patterns = forbidden_patterns or []

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
# TOOL: maestro_log_evidence - Log evidence to the chain
# ============================================================================

@mcp.tool()
def maestro_log_evidence(
    evidence_type: Literal["observation", "hypothesis", "decision", "verification"],
    stage: str,
    content: Dict[str, Any],
    source: str,
    confidence: float = 1.0,
    linked_evidence_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Log evidence to reasoning chain for auditability. Call maestro_help('maestro_log_evidence') for details."""
    # Initialize mutable defaults
    linked_evidence_ids = linked_evidence_ids or []

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
# TOOL: maestro_get_evidence_chain - Query the evidence chain
# ============================================================================

@mcp.tool()
def maestro_get_evidence_chain(
    evidence_type: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Query evidence chain for audit and analysis. Call maestro_help('maestro_get_evidence_chain') for details."""
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
# TOOL: maestro_restore_from_backup - Restore files from backup
# ============================================================================

@mcp.tool()
def maestro_restore_from_backup(
    backup_session: str,
    workspace_root: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Restore files from backup session. Call maestro_help('maestro_restore_from_backup') for details."""
    root = workspace_root or os.getcwd()
    manager = WorkspaceManager(root)

    success, restored, failed = manager.restore_from_backup(backup_session, files)

    if success:
        trace_store.log_rollback(
            files_restored=restored,
            reason="Manual rollback via maestro_restore_from_backup",
            stage="debug",
        )

    return {
        "ok": success,
        "restored_files": restored,
        "failed": failed,
        "backup_session": backup_session,
    }


# ============================================================================
# TOOL: maestro_enter_stage - Enter a workflow stage (dynamic tool loading)
# ============================================================================

@mcp.tool()
def maestro_enter_stage(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
) -> Dict[str, Any]:
    """Enter stage and load appropriate tools dynamically. Call maestro_help('maestro_enter_stage') for details."""
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
# TOOL: maestro_enter_skill - Enter a specific skill (fine-grained tool loading)
# ============================================================================

@mcp.tool()
def maestro_enter_skill(
    skill_name: str,
) -> Dict[str, Any]:
    """Enter specific skill for fine-grained tool loading. Call maestro_help('maestro_enter_skill') for details."""
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
# TOOL: maestro_get_micro_steps - Get micro-steps for current stage/skill
# ============================================================================

@mcp.tool()
def maestro_get_micro_steps(
    stage: Optional[Literal["analyze", "hypothesize", "implement", "debug", "improve"]] = None,
) -> Dict[str, Any]:
    """Get available micro-steps for MAKER decomposition. Call maestro_help('maestro_get_micro_steps') for details."""
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
# TOOL: maestro_vote_micro_step - MAKER-style voting on a micro-step
# ============================================================================

@mcp.tool()
def maestro_vote_micro_step(
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
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run MAKER voting on micro-step with error correction. Call maestro_help('maestro_vote_micro_step') for details."""
    # Initialize mutable defaults
    providers = providers or ["codex", "gemini"]

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
        source="maestro_vote_micro_step",
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
# TOOL: maestro_calibrate - Calibrate voting parameters
# ============================================================================

@mcp.tool()
def maestro_calibrate(
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
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calibrate voting parameters for step type. Call maestro_help('maestro_calibrate') for details."""
    # Initialize mutable defaults
    providers = providers or ["codex", "gemini"]

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
# TOOL: maestro_red_flag_check - Check content for red flags
# ============================================================================

@mcp.tool()
def maestro_red_flag_check(
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
    """Check content for MAKER red flags. Call maestro_help('maestro_red_flag_check') for details."""
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
# TOOL: maestro_get_loaded_tools - Get currently loaded tools
# ============================================================================

@mcp.tool()
def maestro_get_loaded_tools() -> Dict[str, Any]:
    """Get currently loaded tools list. Call maestro_help('maestro_get_loaded_tools') for details."""
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
# TOOL: maestro_recommend_tools - Get recommended tools for a task
# ============================================================================

@mcp.tool()
def maestro_recommend_tools(
    task_description: str,
) -> Dict[str, Any]:
    """Get recommended tools for task. Call maestro_help('maestro_recommend_tools') for details."""
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
        "hint": f"Use maestro_enter_stage('{suggested_stage}') to load these tools",
    }


# ============================================================================
# TOOL: maestro_exit_stage - Exit current stage and unload tools
# ============================================================================

@mcp.tool()
def maestro_exit_stage() -> Dict[str, Any]:
    """Exit stage and unload non-core tools. Call maestro_help('maestro_exit_stage') for details."""
    skill_session.exit_stage()
    state = skill_session.get_state()

    return {
        "ok": True,
        "message": "Stage exited, tools unloaded to core set",
        "loaded_tools": state["loaded_tools"],
        "context_cost": state["context_cost"],
    }


# ============================================================================
# TOOL: maestro_classify_task - Analyze task structure for architecture selection
# ============================================================================

@mcp.tool()
def maestro_classify_task(
    task_description: str,
    code_context: Optional[str] = None,
    error_logs: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify task structure for optimal coordination topology. Call maestro_help('maestro_classify_task') for details."""
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
# TOOL: maestro_select_architecture - Select coordination topology for a stage
# ============================================================================

@mcp.tool()
def maestro_select_architecture(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
    decomposability_score: float = 0.5,
    sequential_dependency_score: float = 0.5,
    tool_complexity: float = 0.3,
    force_topology: Optional[Literal["sas", "mas_independent", "mas_centralized"]] = None,
) -> Dict[str, Any]:
    """Select optimal coordination topology for stage. Call maestro_help('maestro_select_architecture') for details."""
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
# TOOL: maestro_check_degradation - Check if should fall back to simpler topology
# ============================================================================

@mcp.tool()
def maestro_check_degradation(
    current_topology: Literal["sas", "mas_independent", "mas_centralized"],
    total_messages: int = 0,
    total_rounds: int = 0,
    successes: int = 0,
    failures: int = 0,
    redundancy_rate: float = 0.0,
) -> Dict[str, Any]:
    """Check if should degrade to simpler topology. Call maestro_help('maestro_check_degradation') for details."""
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
# TOOL: maestro_record_coordination_result - Record result for calibration
# ============================================================================

@mcp.tool()
def maestro_record_coordination_result(
    topology: Literal["sas", "mas_independent", "mas_centralized"],
    success: bool,
    tokens_used: int = 0,
    messages_sent: int = 0,
    rounds: int = 1,
    outputs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record coordination result for calibration. Call maestro_help('maestro_record_coordination_result') for details."""
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
# TOOL: maestro_get_coordination_stats - Get calibration statistics
# ============================================================================

@mcp.tool()
def maestro_get_coordination_stats() -> Dict[str, Any]:
    """Get coordination calibration statistics. Call maestro_help('maestro_get_coordination_stats') for details."""
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
# TOOL: maestro_get_stage_strategy - Get recommended strategy for a stage
# ============================================================================

@mcp.tool()
def maestro_get_stage_strategy(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
) -> Dict[str, Any]:
    """Get recommended coordination strategy for stage. Call maestro_help('maestro_get_stage_strategy') for details."""
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
# TOOL: maestro_request_approval - Request human approval for a stage
# ============================================================================

@mcp.tool()
def maestro_request_approval(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
    outputs: Dict[str, Any],
    duration_ms: float = 0.0,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Request human approval before proceeding to the next stage.

    This tool creates a detailed report with questions for the human reviewer.
    The workflow will NOT proceed until approval is submitted.

    Call maestro_help('maestro_request_approval') for details.
    """
    request = human_loop_manager.request_approval(
        stage=stage,
        outputs=outputs,
        duration_ms=duration_ms,
        metrics=metrics,
    )

    # Format for display
    display_text = format_approval_request_for_display(request)

    return {
        "ok": True,
        "request_id": request.request_id,
        "stage": stage,
        "status": request.status.value,
        "display": display_text,
        "report": request.report.to_dict(),
        "questions_count": len(request.report.questions),
        "critical_questions": [
            q.to_dict() for q in request.report.questions
            if q.priority == ReviewPriority.CRITICAL
        ],
        "action_required": "Please review the report and use maestro_submit_approval to approve/reject",
        "action_required_ko": "리포트를 검토하고 maestro_submit_approval을 사용하여 승인/거부해 주세요",
    }


# ============================================================================
# TOOL: maestro_submit_approval - Submit approval decision
# ============================================================================

@mcp.tool()
def maestro_submit_approval(
    request_id: str,
    approved: bool,
    feedback: Optional[str] = None,
    question_responses: Optional[Dict[str, str]] = None,
    revision_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit approval decision for a pending stage.

    Args:
        request_id: The approval request ID from maestro_request_approval
        approved: True to approve and proceed, False to reject
        feedback: General feedback or comments
        question_responses: Responses to specific review questions (keyed by question ID)
        revision_instructions: If not approved, instructions for revision

    Call maestro_help('maestro_submit_approval') for details.
    """
    result = human_loop_manager.submit_approval(
        request_id=request_id,
        approved=approved,
        feedback=feedback,
        question_responses=question_responses,
        revision_instructions=revision_instructions,
    )

    return result


# ============================================================================
# TOOL: maestro_get_pending_approvals - Get pending approval requests
# ============================================================================

@mcp.tool()
def maestro_get_pending_approvals() -> Dict[str, Any]:
    """
    Get all pending approval requests.

    Returns a list of all requests awaiting human approval.
    Call maestro_help('maestro_get_pending_approvals') for details.
    """
    pending = human_loop_manager.get_pending_requests()

    return {
        "ok": True,
        "count": len(pending),
        "pending_requests": pending,
        "message": f"{len(pending)} approval(s) pending" if pending else "No pending approvals",
        "message_ko": f"{len(pending)}개의 승인 대기 중" if pending else "대기 중인 승인 없음",
    }


# ============================================================================
# TOOL: maestro_get_approval_history - Get approval history
# ============================================================================

@mcp.tool()
def maestro_get_approval_history(
    stage: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get history of approval decisions.

    Args:
        stage: Filter by stage name (optional)
        limit: Maximum number of records to return

    Call maestro_help('maestro_get_approval_history') for details.
    """
    history = human_loop_manager.get_approval_history()

    # Filter by stage if specified
    if stage:
        history = [h for h in history if h.get("stage") == stage]

    # Apply limit
    history = history[-limit:]

    # Summary stats
    approved_count = sum(1 for h in history if h.get("status") == "approved")
    rejected_count = sum(1 for h in history if h.get("status") == "rejected")
    revision_count = sum(1 for h in history if h.get("status") == "revision_requested")

    return {
        "ok": True,
        "count": len(history),
        "history": history,
        "summary": {
            "approved": approved_count,
            "rejected": rejected_count,
            "revision_requested": revision_count,
        },
    }


# ============================================================================
# TOOL: maestro_run_stage_with_approval - Run stage and request approval
# ============================================================================

@mcp.tool()
def maestro_run_stage_with_approval(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
    task: str,
    context_files: Optional[List[str]] = None,
    context_facts: Optional[List[str]] = None,
    context_errors: Optional[List[str]] = None,
    providers: Optional[List[str]] = None,
    baseline_confidence: float = 0.0,
    timeout_sec: int = 1800,  # 30 min default - generous timeout with liveness checking
) -> Dict[str, Any]:
    """
    Execute a workflow stage AND automatically request human approval.

    This is the recommended way to run stages with human-in-the-loop.
    It combines maestro_run_stage and maestro_request_approval.

    The workflow will pause until the human approves or rejects.

    Call maestro_help('maestro_run_stage_with_approval') for details.
    """
    # Initialize mutable defaults
    context_files = context_files or []
    context_facts = context_facts or []
    context_errors = context_errors or []

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
                timeout_sec=timeout_sec,
            )
        )
    except RuntimeError:
        result = asyncio.run(
            runner.run_stage(
                stage=stage_enum,
                context=context,
                providers=providers,
                baseline_confidence=baseline_confidence,
                timeout_sec=timeout_sec,
            )
        )

    if not result.success:
        return {
            "ok": False,
            "stage": stage,
            "error": result.error,
            "elapsed_ms": result.elapsed_ms,
        }

    # Create approval request
    approval_request = human_loop_manager.request_approval(
        stage=stage,
        outputs=result.output,
        duration_ms=result.elapsed_ms,
        metrics={
            "consults_used": result.consults_used,
            "providers": providers or ["default"],
        },
    )

    # Format for display
    display_text = format_approval_request_for_display(approval_request)

    return {
        "ok": True,
        "stage": stage,
        "stage_display": result.stage.display_name,
        "stage_result": {
            "success": result.success,
            "output": result.output,
            "next_stage": result.next_stage.value if result.next_stage else None,
            "consults_used": result.consults_used,
            "elapsed_ms": result.elapsed_ms,
        },
        "approval": {
            "request_id": approval_request.request_id,
            "status": "pending",
            "display": display_text,
            "questions_count": len(approval_request.report.questions),
        },
        "action_required": (
            f"Stage '{stage}' completed. Please review the results above and use "
            f"maestro_submit_approval(request_id='{approval_request.request_id}', approved=True/False, ...) "
            "to approve or reject before proceeding to the next stage."
        ),
        "action_required_ko": (
            f"스테이지 '{stage}' 완료. 위의 결과를 검토하고 "
            f"maestro_submit_approval(request_id='{approval_request.request_id}', approved=True/False, ...) "
            "를 사용하여 다음 스테이지로 진행하기 전에 승인 또는 거부해 주세요."
        ),
    }


# ============================================================================
# TOOL: maestro_get_stage_questions - Get review questions for a stage
# ============================================================================

@mcp.tool()
def maestro_get_stage_questions(
    stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"],
) -> Dict[str, Any]:
    """
    Get the review questions that will be asked for a specific stage.

    Use this to preview what questions the human will need to answer.
    Call maestro_help('maestro_get_stage_questions') for details.
    """
    from maestro.human_loop import STAGE_QUESTIONS

    questions = STAGE_QUESTIONS.get(stage, [])

    formatted_questions = []
    for q in questions:
        formatted_questions.append({
            "id": q["id"],
            "question": q["question"],
            "question_ko": q["question_ko"],
            "priority": q["priority"].value,
            "options": q.get("options", []),
            "requires_text_response": q.get("requires_text_response", False),
        })

    # Group by priority
    critical = [q for q in formatted_questions if q["priority"] == "critical"]
    high = [q for q in formatted_questions if q["priority"] == "high"]
    medium = [q for q in formatted_questions if q["priority"] == "medium"]
    low = [q for q in formatted_questions if q["priority"] == "low"]

    return {
        "ok": True,
        "stage": stage,
        "total_questions": len(formatted_questions),
        "questions": formatted_questions,
        "by_priority": {
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
        },
        "summary": {
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "low_count": len(low),
        },
    }


# ============================================================================
# TOOL: maestro_workflow_with_hitl - Full workflow with human-in-the-loop
# ============================================================================

@mcp.tool()
def maestro_workflow_with_hitl(
    task: str,
    start_stage: Literal["analyze", "hypothesize", "implement", "debug", "improve"] = "analyze",
    auto_approve_low_risk: bool = False,
) -> Dict[str, Any]:
    """
    Start a workflow that requires human approval at each stage.

    This tool initializes a human-in-the-loop workflow where:
    1. Each stage must be explicitly approved before proceeding
    2. Detailed questions are asked at each stage
    3. Feedback is collected and incorporated

    Args:
        task: The task description
        start_stage: Which stage to start from (default: analyze)
        auto_approve_low_risk: If True, automatically approve stages with no critical issues

    Returns guidance on how to proceed with the HITL workflow.

    Call maestro_help('maestro_workflow_with_hitl') for details.
    """
    # Initialize workflow
    context = workflow_engine.start(task)

    # Build guidance
    stage_order = ["analyze", "hypothesize", "implement", "debug", "improve"]
    start_idx = stage_order.index(start_stage)
    remaining_stages = stage_order[start_idx:]

    guidance = {
        "workflow_started": True,
        "task": task,
        "start_stage": start_stage,
        "remaining_stages": remaining_stages,
        "auto_approve_low_risk": auto_approve_low_risk,
        "instructions": {
            "en": [
                f"1. Run the first stage with: maestro_run_stage_with_approval(stage='{start_stage}', task='{task[:50]}...')",
                "2. Review the detailed report and questions",
                "3. Use maestro_submit_approval() to approve/reject",
                "4. If approved, proceed to the next stage",
                "5. Repeat until all stages are complete",
            ],
            "ko": [
                f"1. 첫 번째 스테이지 실행: maestro_run_stage_with_approval(stage='{start_stage}', task='{task[:50]}...')",
                "2. 상세 리포트와 질문을 검토",
                "3. maestro_submit_approval()을 사용하여 승인/거부",
                "4. 승인되면 다음 스테이지로 진행",
                "5. 모든 스테이지가 완료될 때까지 반복",
            ],
        },
        "stage_questions_preview": {
            stage: len(STAGE_QUESTIONS.get(stage, []))
            for stage in remaining_stages
        },
        "tips": {
            "en": [
                "Use maestro_get_stage_questions() to preview questions for any stage",
                "Use maestro_get_pending_approvals() to see pending requests",
                "Use maestro_get_approval_history() to review past decisions",
                "Provide detailed feedback to improve future iterations",
            ],
            "ko": [
                "maestro_get_stage_questions()를 사용하여 각 스테이지의 질문을 미리 볼 수 있습니다",
                "maestro_get_pending_approvals()를 사용하여 대기 중인 요청을 확인할 수 있습니다",
                "maestro_get_approval_history()를 사용하여 과거 결정을 검토할 수 있습니다",
                "상세한 피드백을 제공하여 향후 반복을 개선하세요",
            ],
        },
    }

    return {
        "ok": True,
        "message": "Human-in-the-loop workflow initialized",
        "message_ko": "Human-in-the-loop 워크플로우가 초기화되었습니다",
        **guidance,
    }


# Import STAGE_QUESTIONS for the above tool
from maestro.human_loop import STAGE_QUESTIONS


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Maestro Skills Server...")
    logger.info(f"Available providers: {registry.list_providers()}")
    logger.info(f"Trace directory: {config.tracing.trace_dir}")
    logger.info(f"Max tools per stage: {tool_registry.max_tools}")
    logger.info(f"Disabled tools: {disabled_tools}")
    logger.info("Human-in-the-Loop enabled for all stages")
    mcp.run()
