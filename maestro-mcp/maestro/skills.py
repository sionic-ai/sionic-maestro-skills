"""
Dynamic Skill Loader with Tool Management

Implements skill-based dynamic MCP tool loading to minimize context overhead.
Only loads tools required for the current stage/micro-step.

Architecture:
1. SkillManifest - Defines skills and their tool requirements
2. SkillLoader - Loads skill definitions on-demand
3. DynamicToolRegistry - Manages which tools are exposed
4. SkillSession - Tracks current skill context and loaded tools

IMPORTANT LIMITATION (2025-01):
==================================
The current implementation provides a CONCEPTUAL registry of tools and their
context costs, but does NOT actually wire into FastMCP's tool add/remove APIs.

Tools are still registered via @mcp.tool() decorators at startup, meaning
MCP clients see the FULL tool list regardless of the current skill/stage.

To fully implement dynamic tool loading, you would need to:
1. Use FastMCP's dynamic tool registration (if supported)
2. OR implement tool visibility filtering at the MCP protocol level
3. OR use multiple MCP server instances with different tool sets

The current registry is useful for:
- Tracking which tools SHOULD be loaded (for documentation)
- Computing context cost estimates
- Planning which tools to expose in future implementations

This is tracked as a known issue in CRITICAL_ANALYSIS.md.
"""

import os
import yaml
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .maker import (
    StageType, MicroStepType, MicroStepSpec,
    MICRO_STEP_SPECS, get_tools_for_stage, get_tools_for_step
)


# =============================================================================
# SKILL MANIFEST
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    handler: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    # Context cost (approximate tokens added to context)
    context_cost: int = 100


@dataclass
class SkillDefinition:
    """Definition of a skill with its tool requirements."""
    name: str
    stage: StageType
    description: str
    # Required tools for this skill
    required_tools: List[str] = field(default_factory=list)
    # Optional tools that enhance the skill
    optional_tools: List[str] = field(default_factory=list)
    # Micro-steps this skill can execute
    micro_steps: List[MicroStepType] = field(default_factory=list)
    # Skill prompt file path
    prompt_file: Optional[str] = None
    # Output schema file path
    schema_file: Optional[str] = None
    # Role/persona for this skill
    role: Optional[str] = None
    # Priority for tool loading (higher = load first)
    priority: int = 0


class SkillManifest:
    """
    Manages the skill manifest - a registry of all available skills
    and their tool requirements.
    """

    # Core tools that are always loaded (minimal set)
    CORE_TOOLS = [
        "maestro_list_providers",    # Always need to know available providers
        "maestro_get_skill",         # Load skill definitions on-demand
        "maestro_workflow_state",    # Check current workflow state
    ]

    # Stage-specific tool sets
    STAGE_TOOLS: Dict[StageType, List[str]] = {
        StageType.ANALYZE: [
            "maestro_pack_context",
            "maestro_log_evidence",
            "maestro_consult",
        ],
        StageType.HYPOTHESIZE: [
            "maestro_pack_context",
            "maestro_consult",
            "maestro_ensemble_generate",
            "maestro_consensus_vote",
            "maestro_validate_content",
            "maestro_log_evidence",
        ],
        StageType.IMPLEMENT: [
            "maestro_consult",
            "maestro_apply_patch",
            "maestro_verify",
            "maestro_log_evidence",
        ],
        StageType.DEBUG: [
            "maestro_pack_context",
            "maestro_consult",
            "maestro_verify",
            "maestro_restore_from_backup",
            "maestro_log_evidence",
        ],
        StageType.IMPROVE: [
            "maestro_consult",
            "maestro_apply_patch",
            "maestro_verify",
            "maestro_ensemble_generate",
            "maestro_select_best",
            "maestro_log_evidence",
        ],
    }

    def __init__(self, manifest_path: Optional[str] = None):
        """
        Initialize skill manifest.

        Args:
            manifest_path: Path to skill_manifest.yaml (optional)
        """
        self.manifest_path = manifest_path
        self.skills: Dict[str, SkillDefinition] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}

        if manifest_path and os.path.exists(manifest_path):
            self._load_from_file(manifest_path)
        else:
            self._init_default_skills()

    def _load_from_file(self, path: str):
        """Load manifest from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        for skill_data in data.get('skills', []):
            stage = StageType(skill_data['stage'])
            micro_steps = [
                MicroStepType(ms) for ms in skill_data.get('micro_steps', [])
            ]

            skill = SkillDefinition(
                name=skill_data['name'],
                stage=stage,
                description=skill_data.get('description', ''),
                required_tools=skill_data.get('required_tools', []),
                optional_tools=skill_data.get('optional_tools', []),
                micro_steps=micro_steps,
                prompt_file=skill_data.get('prompt_file'),
                schema_file=skill_data.get('schema_file'),
                role=skill_data.get('role'),
                priority=skill_data.get('priority', 0),
            )
            self.skills[skill.name] = skill

        for tool_data in data.get('tools', []):
            tool = ToolDefinition(
                name=tool_data['name'],
                description=tool_data.get('description', ''),
                context_cost=tool_data.get('context_cost', 100),
            )
            self.tool_definitions[tool.name] = tool

    def _init_default_skills(self):
        """Initialize default skill definitions based on MAKER micro-steps."""
        # Create skills for each stage
        stage_skills = {
            StageType.ANALYZE: [
                ("spec_extraction", "Extract input/output specifications", [MicroStepType.S1_SPEC_EXTRACT]),
                ("edge_case_generation", "Generate edge cases for testing", [MicroStepType.S2_EDGE_CASE]),
                ("mre_creation", "Create minimal reproducible examples", [MicroStepType.S3_MRE]),
            ],
            StageType.HYPOTHESIZE: [
                ("root_cause_analysis", "Identify potential root causes", [MicroStepType.H1_ROOT_CAUSE]),
                ("verification_design", "Design verification experiments", [MicroStepType.H2_VERIFICATION]),
            ],
            StageType.IMPLEMENT: [
                ("patch_generation", "Generate minimal code patches", [MicroStepType.C1_MINIMAL_PATCH]),
                ("compile_verification", "Verify patches compile correctly", [MicroStepType.C2_COMPILE_CHECK]),
            ],
            StageType.DEBUG: [
                ("failure_classification", "Classify failure types", [MicroStepType.D1_FAILURE_LABEL]),
                ("experiment_planning", "Plan next debugging experiment", [MicroStepType.D2_NEXT_EXPERIMENT]),
            ],
            StageType.IMPROVE: [
                ("refactoring", "Apply safe refactorings", [MicroStepType.R1_REFACTOR]),
                ("performance_optimization", "Optimize performance", [MicroStepType.R2_PERF]),
            ],
        }

        for stage, skills in stage_skills.items():
            for name, description, micro_steps in skills:
                # Collect required tools from micro-steps
                required_tools = set()
                for ms in micro_steps:
                    required_tools.update(get_tools_for_step(ms))

                skill = SkillDefinition(
                    name=name,
                    stage=stage,
                    description=description,
                    required_tools=list(required_tools),
                    optional_tools=[],
                    micro_steps=micro_steps,
                )
                self.skills[name] = skill

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get skill definition by name."""
        return self.skills.get(name)

    def get_skills_for_stage(self, stage: StageType) -> List[SkillDefinition]:
        """Get all skills for a stage."""
        return [s for s in self.skills.values() if s.stage == stage]

    def get_tools_for_skill(self, skill_name: str) -> Set[str]:
        """Get required tools for a skill."""
        skill = self.skills.get(skill_name)
        if not skill:
            return set()
        return set(skill.required_tools) | set(skill.optional_tools)

    def get_tools_for_stage(self, stage: StageType) -> Set[str]:
        """Get all tools needed for a stage (union of all skills)."""
        tools = set(self.CORE_TOOLS)
        tools.update(self.STAGE_TOOLS.get(stage, []))

        for skill in self.get_skills_for_stage(stage):
            tools.update(skill.required_tools)

        return tools

    def get_minimal_tools(self) -> Set[str]:
        """Get minimal tool set (core tools only)."""
        return set(self.CORE_TOOLS)


# =============================================================================
# DYNAMIC TOOL REGISTRY
# =============================================================================

class ToolLoadState(Enum):
    """State of a tool in the registry."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    DISABLED = "disabled"


@dataclass
class RegisteredTool:
    """A tool registered in the dynamic registry."""
    name: str
    definition: ToolDefinition
    handler: Optional[Callable]
    state: ToolLoadState = ToolLoadState.UNLOADED
    load_count: int = 0  # Track usage for optimization


class DynamicToolRegistry:
    """
    Manages dynamic tool loading based on current skill/stage.

    Key features:
    1. Lazy loading - tools loaded only when needed
    2. Stage-based loading - batch load tools for current stage
    3. Usage tracking - optimize based on actual usage
    4. Context budget - limit total tools to save context
    """

    DEFAULT_MAX_TOOLS = 10  # Maximum tools to expose at once

    def __init__(
        self,
        manifest: SkillManifest,
        max_tools: int = DEFAULT_MAX_TOOLS,
        disabled_tools: Optional[Set[str]] = None,
    ):
        """
        Args:
            manifest: SkillManifest with tool definitions
            max_tools: Maximum tools to expose at once
            disabled_tools: Set of tool names to never load
        """
        self.manifest = manifest
        self.max_tools = max_tools
        self.disabled_tools = disabled_tools or set()

        # Registry of all tools
        self.registry: Dict[str, RegisteredTool] = {}
        # Currently exposed tools
        self.exposed_tools: Set[str] = set()
        # Load history for optimization
        self.load_history: List[str] = []

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        context_cost: int = 100,
    ):
        """Register a tool handler."""
        if name in self.disabled_tools:
            state = ToolLoadState.DISABLED
        else:
            state = ToolLoadState.UNLOADED

        self.registry[name] = RegisteredTool(
            name=name,
            definition=ToolDefinition(
                name=name,
                description=description,
                handler=handler,
                context_cost=context_cost,
            ),
            handler=handler,
            state=state,
        )

    def load_tools_for_stage(self, stage: StageType) -> List[str]:
        """
        Load tools required for a stage.

        Returns:
            List of newly loaded tool names
        """
        required = self.manifest.get_tools_for_stage(stage)
        core = self.manifest.get_minimal_tools()

        # Always include core tools
        to_load = core | required

        # Filter out disabled tools
        to_load = to_load - self.disabled_tools

        # Apply max_tools limit
        if len(to_load) > self.max_tools:
            # Prioritize core tools, then by usage
            core_list = list(to_load & core)
            other_list = list(to_load - core)

            # Sort others by load_count (most used first)
            other_list.sort(
                key=lambda t: self.registry.get(t, RegisteredTool(t, ToolDefinition(t, ""), None)).load_count,
                reverse=True
            )

            to_load = set(core_list + other_list[:self.max_tools - len(core_list)])

        newly_loaded = []
        for tool_name in to_load:
            if tool_name in self.registry:
                tool = self.registry[tool_name]
                if tool.state == ToolLoadState.UNLOADED:
                    tool.state = ToolLoadState.LOADED
                    tool.load_count += 1
                    newly_loaded.append(tool_name)
                    self.load_history.append(tool_name)

        self.exposed_tools = to_load
        return newly_loaded

    def load_tools_for_skill(self, skill_name: str) -> List[str]:
        """
        Load tools required for a specific skill.

        Returns:
            List of newly loaded tool names
        """
        skill = self.manifest.get_skill(skill_name)
        if not skill:
            return []

        core = self.manifest.get_minimal_tools()
        required = set(skill.required_tools)
        optional = set(skill.optional_tools)

        # Core + required are must-have
        must_have = core | required

        # Calculate remaining budget
        remaining_budget = self.max_tools - len(must_have)

        # Add optional tools within budget
        if remaining_budget > 0:
            to_load = must_have | set(list(optional)[:remaining_budget])
        else:
            to_load = must_have

        # Filter out disabled
        to_load = to_load - self.disabled_tools

        newly_loaded = []
        for tool_name in to_load:
            if tool_name in self.registry:
                tool = self.registry[tool_name]
                if tool.state == ToolLoadState.UNLOADED:
                    tool.state = ToolLoadState.LOADED
                    tool.load_count += 1
                    newly_loaded.append(tool_name)
                    self.load_history.append(tool_name)

        self.exposed_tools = to_load
        return newly_loaded

    def load_tools_for_micro_step(self, step_type: MicroStepType) -> List[str]:
        """
        Load minimal tools for a specific micro-step.

        This is the most granular loading - only what's absolutely needed.
        """
        core = self.manifest.get_minimal_tools()
        step_tools = set(get_tools_for_step(step_type))

        to_load = (core | step_tools) - self.disabled_tools

        newly_loaded = []
        for tool_name in to_load:
            if tool_name in self.registry:
                tool = self.registry[tool_name]
                if tool.state == ToolLoadState.UNLOADED:
                    tool.state = ToolLoadState.LOADED
                    tool.load_count += 1
                    newly_loaded.append(tool_name)

        self.exposed_tools = to_load
        return newly_loaded

    def unload_all(self):
        """Unload all tools (reset to minimal state)."""
        for tool in self.registry.values():
            if tool.state == ToolLoadState.LOADED:
                tool.state = ToolLoadState.UNLOADED
        self.exposed_tools = set()

    def get_exposed_tools(self) -> List[str]:
        """Get list of currently exposed tool names."""
        return list(self.exposed_tools)

    def is_tool_loaded(self, name: str) -> bool:
        """Check if a tool is currently loaded."""
        return name in self.exposed_tools

    def get_tool_handler(self, name: str) -> Optional[Callable]:
        """Get handler for a tool if loaded."""
        if name not in self.exposed_tools:
            return None
        tool = self.registry.get(name)
        return tool.handler if tool else None

    def get_context_cost(self) -> int:
        """Get total context cost of currently loaded tools."""
        return sum(
            self.registry[name].definition.context_cost
            for name in self.exposed_tools
            if name in self.registry
        )

    def get_usage_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return {
            name: tool.load_count
            for name, tool in self.registry.items()
        }


# =============================================================================
# SKILL SESSION
# =============================================================================

@dataclass
class SkillState:
    """Current state of skill execution."""
    current_stage: Optional[StageType] = None
    current_skill: Optional[str] = None
    current_micro_step: Optional[MicroStepType] = None
    loaded_tools: List[str] = field(default_factory=list)
    completed_steps: List[MicroStepType] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)


class SkillSession:
    """
    Manages a skill session with dynamic tool loading.

    Provides high-level API for:
    1. Starting/ending skill execution
    2. Automatic tool loading based on context
    3. Tracking skill progress and results
    """

    def __init__(
        self,
        manifest: SkillManifest,
        tool_registry: DynamicToolRegistry,
    ):
        self.manifest = manifest
        self.tool_registry = tool_registry
        self.state = SkillState()
        self._session_id: Optional[str] = None

    def start_session(self, session_id: str):
        """Start a new skill session."""
        self._session_id = session_id
        self.state = SkillState()

    def enter_stage(self, stage: StageType) -> List[str]:
        """
        Enter a workflow stage.

        Loads all tools needed for the stage.

        Returns:
            List of loaded tool names
        """
        self.state.current_stage = stage
        self.state.current_skill = None
        self.state.current_micro_step = None

        loaded = self.tool_registry.load_tools_for_stage(stage)
        self.state.loaded_tools = self.tool_registry.get_exposed_tools()

        return loaded

    def enter_skill(self, skill_name: str) -> Tuple[bool, List[str]]:
        """
        Enter a specific skill within the current stage.

        Loads only tools needed for this skill.

        Returns:
            (success, list of loaded tool names)
        """
        skill = self.manifest.get_skill(skill_name)
        if not skill:
            return False, []

        # Validate skill belongs to current stage
        if self.state.current_stage and skill.stage != self.state.current_stage:
            return False, []

        self.state.current_skill = skill_name
        self.state.current_stage = skill.stage

        loaded = self.tool_registry.load_tools_for_skill(skill_name)
        self.state.loaded_tools = self.tool_registry.get_exposed_tools()

        return True, loaded

    def enter_micro_step(self, step_type: MicroStepType) -> Tuple[bool, List[str]]:
        """
        Enter a specific micro-step (most granular loading).

        Loads minimal tools for just this step.

        Returns:
            (success, list of loaded tool names)
        """
        spec = MICRO_STEP_SPECS.get(step_type)
        if not spec:
            return False, []

        # Validate step belongs to current stage
        if self.state.current_stage and spec.stage != self.state.current_stage:
            return False, []

        self.state.current_micro_step = step_type

        loaded = self.tool_registry.load_tools_for_micro_step(step_type)
        self.state.loaded_tools = self.tool_registry.get_exposed_tools()

        return True, loaded

    def complete_micro_step(self, step_type: MicroStepType, result: Any):
        """Mark a micro-step as completed with its result."""
        self.state.completed_steps.append(step_type)
        self.state.step_results[step_type.value] = result

    def exit_skill(self):
        """Exit current skill."""
        self.state.current_skill = None
        self.state.current_micro_step = None

    def exit_stage(self):
        """Exit current stage and unload non-core tools."""
        self.state.current_stage = None
        self.state.current_skill = None
        self.state.current_micro_step = None

        # Keep only core tools
        self.tool_registry.unload_all()
        core = self.manifest.get_minimal_tools()
        for tool_name in core:
            if tool_name in self.tool_registry.registry:
                self.tool_registry.registry[tool_name].state = ToolLoadState.LOADED
        self.tool_registry.exposed_tools = core
        self.state.loaded_tools = list(core)

    def end_session(self):
        """End the session and clean up."""
        self.tool_registry.unload_all()
        self.state = SkillState()
        self._session_id = None

    def get_state(self) -> Dict[str, Any]:
        """Get current session state."""
        return {
            "session_id": self._session_id,
            "current_stage": self.state.current_stage.value if self.state.current_stage else None,
            "current_skill": self.state.current_skill,
            "current_micro_step": self.state.current_micro_step.value if self.state.current_micro_step else None,
            "loaded_tools": self.state.loaded_tools,
            "completed_steps": [s.value for s in self.state.completed_steps],
            "context_cost": self.tool_registry.get_context_cost(),
        }

    def get_available_skills(self) -> List[Dict[str, Any]]:
        """Get skills available for current stage."""
        if not self.state.current_stage:
            return []

        skills = self.manifest.get_skills_for_stage(self.state.current_stage)
        return [
            {
                "name": s.name,
                "description": s.description,
                "required_tools": s.required_tools,
                "micro_steps": [ms.value for ms in s.micro_steps],
            }
            for s in skills
        ]

    def get_available_micro_steps(self) -> List[Dict[str, Any]]:
        """Get micro-steps available for current skill."""
        if not self.state.current_skill:
            return []

        skill = self.manifest.get_skill(self.state.current_skill)
        if not skill:
            return []

        return [
            {
                "type": ms.value,
                "description": MICRO_STEP_SPECS[ms].description,
                "has_oracle": MICRO_STEP_SPECS[ms].has_oracle,
                "default_k": MICRO_STEP_SPECS[ms].default_k,
                "required_tools": MICRO_STEP_SPECS[ms].required_tools,
            }
            for ms in skill.micro_steps
        ]


# =============================================================================
# SKILL LOADER (File-based skill loading)
# =============================================================================

class SkillLoader:
    """
    Loads skill definitions from files.

    Skills are defined in:
    - skills/*.md - Skill prompts
    - schemas/*.json - Output schemas
    - roles/*.md - Role/persona definitions
    """

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: Base directory containing skills/, schemas/, roles/
        """
        self.base_dir = Path(base_dir)
        self.skills_dir = self.base_dir / "skills"
        self.schemas_dir = self.base_dir / "schemas"
        self.roles_dir = self.base_dir / "roles"
        self._cache: Dict[str, str] = {}

    def load_skill_prompt(self, skill_name: str) -> Optional[str]:
        """Load skill prompt from file."""
        cache_key = f"skill:{skill_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try different naming patterns
        patterns = [
            f"{skill_name}.md",
            f"stage*_{skill_name}.md",
        ]

        for pattern in patterns:
            for path in self.skills_dir.glob(pattern):
                content = path.read_text()
                self._cache[cache_key] = content
                return content

        return None

    def load_schema(self, schema_name: str) -> Optional[Dict]:
        """Load output schema from file."""
        cache_key = f"schema:{schema_name}"
        if cache_key in self._cache:
            return json.loads(self._cache[cache_key])

        # Try different naming patterns
        patterns = [
            f"{schema_name}.json",
            f"stage*_{schema_name}.json",
        ]

        for pattern in patterns:
            for path in self.schemas_dir.glob(pattern):
                content = path.read_text()
                self._cache[cache_key] = content
                return json.loads(content)

        return None

    def load_role(self, role_name: str) -> Optional[str]:
        """Load role/persona from file."""
        cache_key = f"role:{role_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.roles_dir / f"{role_name}.md"
        if path.exists():
            content = path.read_text()
            self._cache[cache_key] = content
            return content

        return None

    def get_skill_for_stage(self, stage: StageType) -> Optional[str]:
        """Get skill prompt for a stage."""
        stage_map = {
            StageType.ANALYZE: "stage1_example_analysis",
            StageType.HYPOTHESIZE: "stage2_hypothesis",
            StageType.IMPLEMENT: "stage3_implementation",
            StageType.DEBUG: "stage4_debug_loop",
            StageType.IMPROVE: "stage5_recursive_improve",
        }
        skill_name = stage_map.get(stage)
        return self.load_skill_prompt(skill_name) if skill_name else None

    def get_schema_for_stage(self, stage: StageType) -> Optional[Dict]:
        """Get output schema for a stage."""
        stage_num = {
            StageType.ANALYZE: 1,
            StageType.HYPOTHESIZE: 2,
            StageType.IMPLEMENT: 3,
            StageType.DEBUG: 4,
            StageType.IMPROVE: 5,
        }
        num = stage_num.get(stage)
        return self.load_schema(f"stage{num}_output") if num else None

    def clear_cache(self):
        """Clear the file cache."""
        self._cache.clear()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_skill_session(
    base_dir: str,
    manifest_path: Optional[str] = None,
    max_tools: int = 10,
    disabled_tools: Optional[Set[str]] = None,
) -> Tuple[SkillSession, SkillLoader]:
    """
    Create a skill session with all dependencies.

    Args:
        base_dir: Base directory for skills/schemas/roles
        manifest_path: Optional path to skill_manifest.yaml
        max_tools: Maximum tools to expose at once
        disabled_tools: Set of tool names to disable

    Returns:
        (SkillSession, SkillLoader)
    """
    manifest = SkillManifest(manifest_path)
    registry = DynamicToolRegistry(
        manifest=manifest,
        max_tools=max_tools,
        disabled_tools=disabled_tools,
    )
    session = SkillSession(manifest, registry)
    loader = SkillLoader(base_dir)

    return session, loader


def get_recommended_tools_for_task(
    task_description: str,
    manifest: SkillManifest,
) -> List[str]:
    """
    Recommend tools based on task description.

    Simple keyword-based recommendation for bootstrap.
    """
    task_lower = task_description.lower()

    # Keywords -> stages mapping
    keywords = {
        StageType.ANALYZE: ["analyze", "understand", "read", "examine", "investigate"],
        StageType.HYPOTHESIZE: ["why", "cause", "reason", "hypothesis", "theory"],
        StageType.IMPLEMENT: ["fix", "implement", "add", "create", "write", "patch"],
        StageType.DEBUG: ["debug", "error", "fail", "broken", "not working"],
        StageType.IMPROVE: ["refactor", "improve", "optimize", "clean", "performance"],
    }

    # Find matching stages
    matching_stages = []
    for stage, words in keywords.items():
        if any(word in task_lower for word in words):
            matching_stages.append(stage)

    # Default to analyze if no match
    if not matching_stages:
        matching_stages = [StageType.ANALYZE]

    # Collect tools from matching stages
    tools = set(manifest.get_minimal_tools())
    for stage in matching_stages:
        tools.update(manifest.get_tools_for_stage(stage))

    return list(tools)
