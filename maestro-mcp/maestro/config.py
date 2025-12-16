"""
Configuration management for Maestro Skills MCP.
Loads from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ProviderConfig:
    """Configuration for a single CLI provider."""
    cmd: str
    default_model: Optional[str] = None
    timeout_sec: int = 1800  # 30 minutes default (generous timeout)
    max_output_chars: int = 60000
    enabled: bool = True
    max_no_output_sec: int = 300  # Kill if no output for 5 minutes


@dataclass
class CoordinationPolicy:
    """
    Measured Coordination Policy based on the paper findings.

    Key thresholds from "Towards a Science of Scaling Agent Systems":
    - capability_threshold: ~45% baseline = MAS returns diminish
    - max_consult_per_stage: Limit overhead
    - max_consult_total: Total budget per workflow
    """
    capability_threshold: float = 0.45  # Paper: ~45% baseline saturation
    max_consult_per_stage: int = 2
    max_consult_total: int = 6
    prefer_tests_first: bool = True  # Poetiq pattern: auto-eval > LLM judge
    enable_ensemble_for_hypotheses: bool = True
    enable_ensemble_for_implementation: bool = False  # Paper: tool-heavy = bad for MAS
    debug_loop_single_agent: bool = True  # Paper: sequential tasks degrade with MAS


@dataclass
class ContextConfig:
    """Configuration for context packing."""
    max_files: int = 7
    max_lines_per_file: int = 200
    max_error_lines: int = 50
    max_total_chars: int = 40000
    truncate_strategy: str = "tail"  # "head", "tail", "middle"


@dataclass
class TracingConfig:
    """Configuration for tracing and metrics."""
    enabled: bool = True
    trace_dir: str = ".maestro-traces"
    log_level: str = "INFO"
    save_prompts: bool = True
    save_responses: bool = True


@dataclass
class MaestroConfig:
    """Master configuration for Maestro Skills MCP."""

    # Provider configurations
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    # Coordination policy
    policy: CoordinationPolicy = field(default_factory=CoordinationPolicy)

    # Context engineering
    context: ContextConfig = field(default_factory=ContextConfig)

    # Tracing
    tracing: TracingConfig = field(default_factory=TracingConfig)

    # Disabled tools (to reduce context overhead as PAL MCP suggests)
    disabled_tools: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "MaestroConfig":
        """Load configuration from environment variables."""

        # Generous timeouts by default (30 min), with liveness checking for stalls
        providers = {
            "codex": ProviderConfig(
                cmd=os.getenv("MAESTRO_CODEX_CMD", "codex"),
                default_model=os.getenv("MAESTRO_CODEX_MODEL", "gpt-5.1-codex-max"),
                timeout_sec=int(os.getenv("MAESTRO_CODEX_TIMEOUT", "1800")),  # 30 min default
                enabled=os.getenv("MAESTRO_CODEX_ENABLED", "true").lower() == "true",
                max_no_output_sec=int(os.getenv("MAESTRO_CODEX_STALL_TIMEOUT", "300")),  # 5 min stall
            ),
            "gemini": ProviderConfig(
                cmd=os.getenv("MAESTRO_GEMINI_CMD", "gemini"),
                default_model=os.getenv("MAESTRO_GEMINI_MODEL", "gemini-3-pro-preview"),
                timeout_sec=int(os.getenv("MAESTRO_GEMINI_TIMEOUT", "1800")),  # 30 min default
                enabled=os.getenv("MAESTRO_GEMINI_ENABLED", "true").lower() == "true",
                max_no_output_sec=int(os.getenv("MAESTRO_GEMINI_STALL_TIMEOUT", "300")),  # 5 min stall
            ),
            "claude": ProviderConfig(
                cmd=os.getenv("MAESTRO_CLAUDE_CMD", "claude"),
                default_model=os.getenv("MAESTRO_CLAUDE_MODEL", "opus"),
                timeout_sec=int(os.getenv("MAESTRO_CLAUDE_TIMEOUT", "1800")),  # 30 min default
                enabled=os.getenv("MAESTRO_CLAUDE_ENABLED", "true").lower() == "true",
                max_no_output_sec=int(os.getenv("MAESTRO_CLAUDE_STALL_TIMEOUT", "300")),  # 5 min stall
            ),
        }

        policy = CoordinationPolicy(
            capability_threshold=float(os.getenv("MAESTRO_CAPABILITY_THRESHOLD", "0.45")),
            max_consult_per_stage=int(os.getenv("MAESTRO_MAX_CONSULT_PER_STAGE", "2")),
            max_consult_total=int(os.getenv("MAESTRO_MAX_CONSULT_TOTAL", "6")),
            prefer_tests_first=os.getenv("MAESTRO_PREFER_TESTS_FIRST", "true").lower() == "true",
            enable_ensemble_for_hypotheses=os.getenv("MAESTRO_ENSEMBLE_HYPOTHESES", "true").lower() == "true",
            enable_ensemble_for_implementation=os.getenv("MAESTRO_ENSEMBLE_IMPL", "false").lower() == "true",
            debug_loop_single_agent=os.getenv("MAESTRO_DEBUG_SINGLE_AGENT", "true").lower() == "true",
        )

        context = ContextConfig(
            max_files=int(os.getenv("MAESTRO_CONTEXT_MAX_FILES", "7")),
            max_lines_per_file=int(os.getenv("MAESTRO_CONTEXT_MAX_LINES", "200")),
            max_error_lines=int(os.getenv("MAESTRO_CONTEXT_MAX_ERROR_LINES", "50")),
            max_total_chars=int(os.getenv("MAESTRO_CONTEXT_MAX_CHARS", "40000")),
        )

        tracing = TracingConfig(
            enabled=os.getenv("MAESTRO_TRACING_ENABLED", "true").lower() == "true",
            trace_dir=os.getenv("MAESTRO_TRACE_DIR", ".maestro-traces"),
            log_level=os.getenv("MAESTRO_LOG_LEVEL", "INFO"),
        )

        disabled_tools = [
            t.strip()
            for t in os.getenv("MAESTRO_DISABLED_TOOLS", "").split(",")
            if t.strip()
        ]

        return cls(
            providers=providers,
            policy=policy,
            context=context,
            tracing=tracing,
            disabled_tools=disabled_tools,
        )

    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        return self.providers.get(name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return tool_name not in self.disabled_tools
