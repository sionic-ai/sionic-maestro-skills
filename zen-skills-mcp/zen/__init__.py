"""
Zen Skills MCP - Multi-LLM Orchestration with Measured Coordination

Based on two papers:
1. "Towards a Science of Scaling Agent Systems" (Kim et al., 2025):
   - Tool-Coordination Trade-off: Tools are expensive in MAS
   - Capability Saturation: ~45% baseline = diminishing MAS returns
   - Error Amplification: Independent agents amplify errors 17.2x

2. "Solving a Million-Step LLM Task With Zero Errors" (MAKER, 2025):
   - MAD: Maximal Agentic Decomposition into micro-steps
   - First-to-ahead-by-k voting for error correction
   - Red-flagging: Format errors signal reasoning errors

Architecture: Centralized Consult Pattern
- Claude Code = Orchestrator (tool execution)
- Codex/Gemini/Claude CLI = Consultants (text advice only)
"""

__version__ = "0.4.0"

from .config import ZenConfig
from .providers import CodexProvider, GeminiProvider, ClaudeProvider, ProviderRegistry
from .context import ContextPacker
from .workflow import WorkflowEngine, Stage
from .selection import SelectionEngine, SelectionMode
from .tracing import TraceStore, Metrics
from .verify import VerificationEngine, VerificationType
from .workspace import WorkspaceManager
from .consensus import ConsensusEngine, ConsensusConfig

# MAKER-style modules
from .maker import (
    StageType, MicroStepType, MICRO_STEP_SPECS,
    RedFlagger, VoteStep, Calibrator,
    get_tools_for_step, get_tools_for_stage,
)
from .skills import (
    SkillManifest, DynamicToolRegistry, SkillSession, SkillLoader,
    create_skill_session,
)
from .coordination import (
    CoordinationTopology, TaskStructureFeatures, CoordinationDecision,
    CoordinationMetrics, MetricsTracker, ArchitectureSelectionEngine,
    TaskStructureClassifier, DegradationStrategy, CoordinationPolicy,
)

__all__ = [
    # Config
    "ZenConfig",
    # Providers
    "CodexProvider",
    "GeminiProvider",
    "ClaudeProvider",
    "ProviderRegistry",
    # Context
    "ContextPacker",
    # Workflow
    "WorkflowEngine",
    "Stage",
    # Selection
    "SelectionEngine",
    "SelectionMode",
    # Tracing
    "TraceStore",
    "Metrics",
    # Verification
    "VerificationEngine",
    "VerificationType",
    # Workspace
    "WorkspaceManager",
    # Consensus
    "ConsensusEngine",
    "ConsensusConfig",
    # MAKER
    "StageType",
    "MicroStepType",
    "MICRO_STEP_SPECS",
    "RedFlagger",
    "VoteStep",
    "Calibrator",
    "get_tools_for_step",
    "get_tools_for_stage",
    # Skills
    "SkillManifest",
    "DynamicToolRegistry",
    "SkillSession",
    "SkillLoader",
    "create_skill_session",
    # Coordination (Architecture Selection Engine)
    "CoordinationTopology",
    "TaskStructureFeatures",
    "CoordinationDecision",
    "CoordinationMetrics",
    "MetricsTracker",
    "ArchitectureSelectionEngine",
    "TaskStructureClassifier",
    "DegradationStrategy",
    "CoordinationPolicy",
]
