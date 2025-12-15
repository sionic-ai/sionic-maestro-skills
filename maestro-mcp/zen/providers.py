"""
CLI Provider implementations for external LLM access.

Each provider wraps a CLI tool (codex, gemini, claude) with:
- Safe subprocess execution (shell=False)
- Timeout handling
- Structured output support where available
- Consistent response format

Security: All commands use shell=False to prevent injection attacks.
"""

import subprocess
import os
import json
import tempfile
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("zen.providers")

# Shared executor for async operations (avoids creating new executor per call)
_SHARED_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_shared_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None:
        _SHARED_EXECUTOR = ThreadPoolExecutor(max_workers=4)
    return _SHARED_EXECUTOR


def _extract_json_from_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from CLI output that may contain prose/logs around it.

    CLIs often print logs or messages before/after JSON output.
    This function tries multiple strategies to find valid JSON.
    """
    text = text.strip()

    # Strategy 1: Try the whole thing (fastest path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object boundaries { ... }
    # Look for the outermost JSON object
    brace_start = text.find('{')
    if brace_start != -1:
        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        for i, char in enumerate(text[brace_start:], brace_start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Strategy 3: Find JSON array boundaries [ ... ]
    bracket_start = text.find('[')
    if bracket_start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i, char in enumerate(text[bracket_start:], bracket_start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    candidate = text[bracket_start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Strategy 4: Try each line (some CLIs output JSON on a single line)
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith(('{', '[')):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return None


@dataclass
class ProviderResponse:
    """Standardized response from any CLI provider."""
    ok: bool
    stdout: str
    stderr: str
    returncode: Optional[int] = None
    elapsed_ms: float = 0.0
    provider: str = ""
    model: str = ""
    structured: Optional[Dict[str, Any]] = None
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "elapsed_ms": self.elapsed_ms,
            "provider": self.provider,
            "model": self.model,
            "structured": self.structured,
            "truncated": self.truncated,
        }


class CLIProvider(ABC):
    """Base class for CLI providers."""

    def __init__(
        self,
        cmd: str,
        default_model: Optional[str] = None,
        timeout_sec: int = 300,
        max_output_chars: int = 60000,
        env: Optional[Dict[str, str]] = None,
    ):
        self.cmd = cmd
        self.default_model = default_model
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
        self.env = {**os.environ, **(env or {})}
        self.name = self.__class__.__name__.replace("Provider", "").lower()

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        model: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Build the CLI command for this provider."""
        pass

    def run(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout_sec: Optional[int] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        cwd: Optional[str] = None,
    ) -> ProviderResponse:
        """
        Execute the CLI command synchronously.
        Returns a standardized ProviderResponse.
        """
        import time

        effective_model = model or self.default_model
        effective_timeout = timeout_sec or self.timeout_sec

        # Handle output schema (temp file if needed)
        schema_path = None
        if output_schema:
            fd, schema_path = tempfile.mkstemp(prefix=f"{self.name}_schema_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(output_schema, f)

        try:
            cmd = self.build_command(prompt, effective_model, output_schema if schema_path else None)
            if schema_path:
                cmd = self._inject_schema_arg(cmd, schema_path)

            logger.debug(f"Executing: {' '.join(cmd[:3])}... (prompt length: {len(prompt)})")

            start_time = time.time()
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                cwd=cwd,
                env=self.env,
                shell=False,  # CRITICAL: Security
            )
            elapsed_ms = (time.time() - start_time) * 1000

            # Truncate if needed
            stdout = result.stdout
            truncated = False
            if len(stdout) > self.max_output_chars:
                stdout = stdout[: self.max_output_chars] + "\n... [OUTPUT TRUNCATED]"
                truncated = True

            # Try to parse structured output (using smart extraction)
            structured = None
            if output_schema:
                structured = _extract_json_from_output(stdout)

            return ProviderResponse(
                ok=result.returncode == 0,
                stdout=stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                elapsed_ms=elapsed_ms,
                provider=self.name,
                model=effective_model or "",
                structured=structured,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired as e:
            return ProviderResponse(
                ok=False,
                stdout=e.stdout or "" if hasattr(e, 'stdout') else "",
                stderr=f"TIMEOUT after {effective_timeout}s",
                provider=self.name,
                model=effective_model or "",
            )
        except FileNotFoundError:
            return ProviderResponse(
                ok=False,
                stdout="",
                stderr=f"CLI not found: {self.cmd}. Please ensure it's installed and in PATH.",
                provider=self.name,
                model=effective_model or "",
            )
        except Exception as e:
            logger.exception(f"Provider {self.name} execution failed")
            return ProviderResponse(
                ok=False,
                stdout="",
                stderr=str(e),
                provider=self.name,
                model=effective_model or "",
            )
        finally:
            if schema_path:
                try:
                    os.remove(schema_path)
                except OSError:
                    pass

    async def run_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout_sec: Optional[int] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        cwd: Optional[str] = None,
    ) -> ProviderResponse:
        """Execute the CLI command asynchronously using shared thread pool."""
        loop = asyncio.get_event_loop()
        # Use shared executor instead of creating new one per call
        return await loop.run_in_executor(
            _get_shared_executor(),
            lambda: self.run(prompt, model, timeout_sec, output_schema, cwd),
        )

    def _inject_schema_arg(self, cmd: List[str], schema_path: str) -> List[str]:
        """Override in subclasses to inject schema argument."""
        return cmd


class CodexProvider(CLIProvider):
    """
    OpenAI Codex CLI provider.
    Command: codex exec --model <model> <prompt>

    Features:
    - --output-schema for structured JSON output
    - Defaults to read-only, final message to stdout
    """

    def build_command(
        self,
        prompt: str,
        model: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        cmd = [self.cmd, "exec"]
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)
        return cmd

    def _inject_schema_arg(self, cmd: List[str], schema_path: str) -> List[str]:
        # Insert before the prompt (last element)
        return cmd[:-1] + ["--output-schema", schema_path] + [cmd[-1]]


class GeminiProvider(CLIProvider):
    """
    Google Gemini CLI provider.
    Command: gemini --model <model> <prompt>

    Features:
    - --output-format json for JSON output
    - Supports sandbox mode
    - Default model: gemini-3-pro-preview
    """

    def build_command(
        self,
        prompt: str,
        model: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        cmd = [self.cmd]
        if model:
            cmd.extend(["--model", model])
        if output_schema:
            cmd.extend(["--output-format", "json"])
        cmd.append(prompt)
        return cmd


class ClaudeProvider(CLIProvider):
    """
    Anthropic Claude CLI provider.
    Command: claude -p <prompt> --model <model>

    Features:
    - --output-format json for structured output
    - --json-schema for schema-constrained output
    - --model for model selection (opus, sonnet, haiku)
    - Default model: opus
    """

    def build_command(
        self,
        prompt: str,
        model: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        cmd = [self.cmd, "-p", prompt]
        if model:
            cmd.extend(["--model", model])
        if output_schema:
            cmd.extend(["--output-format", "json"])
        return cmd

    def _inject_schema_arg(self, cmd: List[str], schema_path: str) -> List[str]:
        return cmd + ["--json-schema", schema_path]


class ProviderRegistry:
    """
    Central registry for all CLI providers.
    Supports concurrent execution for ensemble operations.
    """

    def __init__(self):
        self.providers: Dict[str, CLIProvider] = {}
        # Note: Providers use the module-level shared executor (_get_shared_executor)

    def register(self, name: str, provider: CLIProvider) -> None:
        """Register a provider."""
        self.providers[name] = provider
        logger.info(f"Registered provider: {name}")

    def get(self, name: str) -> Optional[CLIProvider]:
        """Get a provider by name."""
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self.providers.keys())

    def run(
        self,
        provider_name: str,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ProviderResponse:
        """Run a single provider synchronously."""
        provider = self.get(provider_name)
        if not provider:
            return ProviderResponse(
                ok=False,
                stdout="",
                stderr=f"Unknown provider: {provider_name}",
                provider=provider_name,
            )
        return provider.run(prompt, model, **kwargs)

    async def run_parallel(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[ProviderResponse]:
        """
        Run multiple providers in parallel.

        Args:
            requests: List of dicts with keys:
                - provider: provider name
                - prompt: the prompt
                - model: optional model override
                - (other kwargs)

        Returns:
            List of ProviderResponse in same order as requests.
        """
        tasks = []
        for req in requests:
            provider = self.get(req["provider"])
            if not provider:
                # Create a coroutine that returns an error response
                async def error_response(name=req["provider"]):
                    return ProviderResponse(
                        ok=False,
                        stdout="",
                        stderr=f"Unknown provider: {name}",
                        provider=name,
                    )
                tasks.append(error_response())
            else:
                tasks.append(
                    provider.run_async(
                        prompt=req["prompt"],
                        model=req.get("model"),
                        timeout_sec=req.get("timeout_sec"),
                        output_schema=req.get("output_schema"),
                        cwd=req.get("cwd"),
                    )
                )
        return await asyncio.gather(*tasks)

    @classmethod
    def from_config(cls, config: "ZenConfig") -> "ProviderRegistry":
        """Create a registry from configuration."""
        from .config import ZenConfig

        registry = cls()

        provider_classes = {
            "codex": CodexProvider,
            "gemini": GeminiProvider,
            "claude": ClaudeProvider,
        }

        for name, pconfig in config.providers.items():
            if not pconfig.enabled:
                logger.info(f"Provider {name} is disabled")
                continue

            provider_cls = provider_classes.get(name)
            if provider_cls:
                provider = provider_cls(
                    cmd=pconfig.cmd,
                    default_model=pconfig.default_model,
                    timeout_sec=pconfig.timeout_sec,
                    max_output_chars=pconfig.max_output_chars,
                )
                registry.register(name, provider)

        return registry
