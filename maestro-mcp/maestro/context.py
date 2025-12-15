"""
Context Packing for Multi-LLM Coordination.

Paper Insight: "Information fragmentation increases coordination tax."

Strategy:
- Pack only relevant excerpts and facts to minimize context window usage
- Stage-specific packing strategies (analyze=broad, implement=narrow)
- Smart truncation that preserves important information
- Error log summarization that keeps essential stack traces

SECURITY:
- All file reads are sandboxed to REPO_ROOTS if configured
- Paths outside allowed roots are blocked to prevent data exfiltration
"""

import os
import re
import glob as glob_module
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from pathlib import Path

logger = logging.getLogger("maestro.context")


# =============================================================================
# SECURITY: Repo-root sandboxing
# =============================================================================

# Configurable repo roots (can be set via environment)
_REPO_ROOTS: Set[Path] = set()


def configure_repo_roots(roots: List[str]) -> None:
    """Configure allowed repository roots for file access."""
    global _REPO_ROOTS
    _REPO_ROOTS = {Path(r).resolve() for r in roots if r}
    if _REPO_ROOTS:
        logger.info(f"Context sandboxing enabled. Allowed roots: {_REPO_ROOTS}")


def _init_repo_roots_from_env() -> None:
    """Initialize repo roots from environment on module load."""
    roots_str = os.getenv("MAESTRO_REPO_ROOTS", "")
    if roots_str:
        roots = [r.strip() for r in roots_str.split(":") if r.strip()]
        configure_repo_roots(roots)


def is_path_allowed(path: str) -> bool:
    """
    Check if a path is within allowed repo roots.

    Security: Prevents reading arbitrary files like ~/.ssh/, ~/.config/, etc.

    Args:
        path: File path to check

    Returns:
        True if allowed (no sandbox, or path is within sandbox)
    """
    if not _REPO_ROOTS:
        # No sandbox configured - allow all (for backwards compatibility)
        # WARNING: This should be avoided in production
        return True

    try:
        resolved = Path(path).resolve()
        for root in _REPO_ROOTS:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False
    except Exception:
        return False


def _block_sensitive_paths(path: str) -> bool:
    """
    Block obviously sensitive paths even without sandbox.

    This is a defense-in-depth measure.
    """
    sensitive_patterns = [
        "/.ssh/",
        "/.gnupg/",
        "/.config/",
        "/.aws/",
        "/.kube/",
        "/etc/passwd",
        "/etc/shadow",
        "/.env",
        "/credentials",
        "/secrets",
        "/.git/config",  # May contain tokens
    ]
    path_lower = path.lower()
    return any(p in path_lower for p in sensitive_patterns)


# Initialize on module load
_init_repo_roots_from_env()


class TruncateStrategy(Enum):
    """Truncation strategies for long content."""
    HEAD = "head"       # Keep beginning
    TAIL = "tail"       # Keep end (good for errors)
    MIDDLE = "middle"   # Keep beginning and end
    SMART = "smart"     # Heuristic-based


@dataclass
class PackingConfig:
    """Configuration for context packing."""
    max_files: int = 7
    max_lines_per_file: int = 200
    max_error_lines: int = 50
    max_total_chars: int = 40000
    truncate_strategy: TruncateStrategy = TruncateStrategy.SMART
    include_line_numbers: bool = True
    collapse_whitespace: bool = True


@dataclass
class ContextPack:
    """A packed context ready to send to a sub-agent."""
    task: str
    files: List[Dict[str, str]]  # {path, content}
    facts: List[str]
    errors: List[str]
    constraints: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_chars: int = 0
    truncated: bool = False

    def to_prompt_section(self) -> str:
        """Convert to a prompt-ready string."""
        return ContextPacker.format_pack(self)


class ContextPacker:
    """
    Smart context packing for sub-agent calls.

    Key strategies:
    1. Priority ordering: constraints > errors > facts > files
    2. Smart truncation preserving boundaries
    3. Error summarization keeping stack traces
    4. File excerpt with important regions highlighted
    """

    @staticmethod
    def read_file_excerpt(
        path: str,
        max_lines: int = 200,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        include_line_numbers: bool = True,
    ) -> str:
        """
        Read file with smart excerpting.

        SECURITY: Paths are validated against repo roots and sensitive patterns.

        Args:
            path: File path
            max_lines: Maximum lines to include
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed)
            include_line_numbers: Whether to prefix with line numbers
        """
        # SECURITY: Block sensitive paths
        if _block_sensitive_paths(path):
            logger.warning(f"BLOCKED: Attempt to read sensitive path: {path}")
            return f"[BLOCKED: Access to sensitive path denied: {path}]"

        # SECURITY: Sandbox check
        if not is_path_allowed(path):
            logger.warning(f"BLOCKED: Path outside allowed roots: {path}")
            return f"[BLOCKED: Path outside allowed repository roots: {path}]"

        if not os.path.exists(path):
            return f"[File not found: {path}]"

        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Apply range if specified
            if start_line is not None or end_line is not None:
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else total_lines
                lines = all_lines[start_idx:end_idx]
                line_offset = start_idx
            else:
                lines = all_lines
                line_offset = 0

            # Truncate if needed
            truncated = False
            if len(lines) > max_lines:
                # Keep first half and last half
                half = max_lines // 2
                lines = lines[:half] + [f"... [{len(lines) - max_lines} lines omitted] ...\n"] + lines[-half:]
                truncated = True

            # Add line numbers
            if include_line_numbers:
                result_lines = []
                actual_line = line_offset + 1
                for line in lines:
                    if line.startswith("..."):
                        result_lines.append(line)
                    else:
                        result_lines.append(f"{actual_line:4d} | {line}")
                        actual_line += 1
                content = "".join(result_lines)
            else:
                content = "".join(lines)

            if truncated:
                content += f"\n[Total: {total_lines} lines]"

            return content

        except Exception as e:
            return f"[Error reading file: {str(e)}]"

    @staticmethod
    def summarize_error(
        error: str,
        max_lines: int = 50,
        keep_traceback: bool = True,
    ) -> str:
        """
        Summarize error output, preserving important information.

        Keeps:
        - First few lines (error message)
        - Traceback frames
        - Last few lines (actual error)
        """
        lines = error.strip().split('\n')

        if len(lines) <= max_lines:
            return error

        # Always keep first 5 and last 10 lines
        head = lines[:5]
        tail = lines[-10:]

        # Find traceback frames in the middle
        middle = lines[5:-10]
        traceback_lines = []

        if keep_traceback:
            # Keep lines that look like traceback frames
            for line in middle:
                if re.match(r'\s*File "', line) or re.match(r'\s+\w+Error:', line):
                    traceback_lines.append(line)
                    # Also keep the next line (usually the code)
                    idx = middle.index(line)
                    if idx + 1 < len(middle):
                        traceback_lines.append(middle[idx + 1])

            # Limit traceback lines
            if len(traceback_lines) > max_lines - 15:
                traceback_lines = traceback_lines[:max_lines - 15]

        omitted = len(lines) - len(head) - len(tail) - len(traceback_lines)

        result = head
        if omitted > 0:
            result.append(f"... [{omitted} lines omitted] ...")
        result.extend(traceback_lines)
        result.extend(tail)

        return '\n'.join(result)

    @staticmethod
    def pack(
        files: List[str],
        facts: List[str],
        errors: List[str],
        constraints: List[str],
        config: Optional[PackingConfig] = None,
        task: str = "",
    ) -> str:
        """
        Create a packed context string.

        Priority order (what gets included first):
        1. Constraints (always included)
        2. Errors (summarized if needed)
        3. Facts (always included)
        4. Files (truncated to fit)
        """
        config = config or PackingConfig()
        packed = []
        current_chars = 0

        # 1. Constraints (always included)
        if constraints:
            packed.append("## Constraints (MUST follow)")
            for c in constraints:
                packed.append(f"- {c}")
            packed.append("")
            current_chars = sum(len(line) + 1 for line in packed)

        # 2. Errors (high priority for debugging)
        if errors:
            packed.append("## Errors / Logs")
            for e in errors:
                summarized = ContextPacker.summarize_error(e, config.max_error_lines)
                packed.append(f"```\n{summarized}\n```")
            packed.append("")
            current_chars = sum(len(line) + 1 for line in packed)

        # 3. Facts
        if facts:
            packed.append("## Known Facts")
            for f in facts:
                packed.append(f"- {f}")
            packed.append("")
            current_chars = sum(len(line) + 1 for line in packed)

        # 4. Files (fill remaining budget)
        remaining_chars = config.max_total_chars - current_chars
        if files and remaining_chars > 500:  # Only if we have meaningful space
            packed.append("## Relevant Files")

            # Expand globs
            expanded_files = []
            for file_pattern in files[:config.max_files]:
                matches = glob_module.glob(file_pattern)
                if matches:
                    expanded_files.extend(matches)
                elif '*' not in file_pattern:
                    expanded_files.append(file_pattern)

            # Deduplicate and limit
            expanded_files = list(dict.fromkeys(expanded_files))[:config.max_files]

            # Calculate per-file budget
            per_file_budget = remaining_chars // max(len(expanded_files), 1)
            per_file_lines = min(config.max_lines_per_file, per_file_budget // 50)

            for path in expanded_files:
                content = ContextPacker.read_file_excerpt(
                    path,
                    max_lines=per_file_lines,
                    include_line_numbers=config.include_line_numbers,
                )

                file_section = f"### {path}\n```\n{content}\n```\n"

                if current_chars + len(file_section) > config.max_total_chars:
                    packed.append(f"### {path}\n[Omitted due to size limit]\n")
                    break

                packed.append(file_section)
                current_chars += len(file_section)

        # Final assembly
        full_text = "\n".join(packed)

        # Hard truncation as safety net
        if len(full_text) > config.max_total_chars:
            full_text = full_text[:config.max_total_chars] + "\n... [CONTEXT TRUNCATED]"

        return full_text

    @staticmethod
    def pack_structured(
        files: List[str],
        facts: List[str],
        errors: List[str],
        constraints: List[str],
        task: str = "",
        config: Optional[PackingConfig] = None,
    ) -> ContextPack:
        """
        Create a structured ContextPack object.

        Returns object that can be serialized or formatted differently.
        """
        config = config or PackingConfig()

        # Expand files
        expanded_files = []
        for file_pattern in files[:config.max_files]:
            matches = glob_module.glob(file_pattern)
            if matches:
                expanded_files.extend(matches)
            elif '*' not in file_pattern:
                expanded_files.append(file_pattern)
        expanded_files = list(dict.fromkeys(expanded_files))[:config.max_files]

        # Read files
        file_contents = []
        for path in expanded_files:
            content = ContextPacker.read_file_excerpt(
                path,
                max_lines=config.max_lines_per_file,
            )
            file_contents.append({"path": path, "content": content})

        # Summarize errors
        summarized_errors = [
            ContextPacker.summarize_error(e, config.max_error_lines)
            for e in errors
        ]

        pack = ContextPack(
            task=task,
            files=file_contents,
            facts=facts,
            errors=summarized_errors,
            constraints=constraints,
        )

        # Calculate total chars
        pack.total_chars = (
            sum(len(f["content"]) for f in pack.files) +
            sum(len(f) for f in pack.facts) +
            sum(len(e) for e in pack.errors) +
            sum(len(c) for c in pack.constraints)
        )

        pack.truncated = pack.total_chars > config.max_total_chars

        return pack

    @staticmethod
    def format_pack(pack: ContextPack) -> str:
        """Format a ContextPack as a prompt string."""
        sections = []

        if pack.task:
            sections.append(f"## Task\n{pack.task}\n")

        if pack.constraints:
            sections.append("## Constraints (MUST follow)")
            sections.extend(f"- {c}" for c in pack.constraints)
            sections.append("")

        if pack.errors:
            sections.append("## Errors")
            for e in pack.errors:
                sections.append(f"```\n{e}\n```")
            sections.append("")

        if pack.facts:
            sections.append("## Known Facts")
            sections.extend(f"- {f}" for f in pack.facts)
            sections.append("")

        if pack.files:
            sections.append("## Files")
            for f in pack.files:
                sections.append(f"### {f['path']}")
                sections.append(f"```\n{f['content']}\n```")
            sections.append("")

        return "\n".join(sections)

    @staticmethod
    def for_stage(
        stage: str,
        files: List[str],
        facts: List[str],
        errors: List[str],
        constraints: List[str],
        task: str = "",
    ) -> str:
        """
        Stage-specific context packing.

        Different stages need different context strategies:
        - analyze: Broad context, more files
        - hypothesize: Focused on errors and facts
        - implement: Narrow, specific files only
        - debug: Error-heavy, recent changes
        - improve: Full context for review
        """
        stage_configs = {
            "analyze": PackingConfig(
                max_files=10,
                max_lines_per_file=300,
                max_error_lines=100,
                max_total_chars=50000,
            ),
            "hypothesize": PackingConfig(
                max_files=5,
                max_lines_per_file=150,
                max_error_lines=80,
                max_total_chars=30000,
            ),
            "implement": PackingConfig(
                max_files=3,
                max_lines_per_file=200,
                max_error_lines=30,
                max_total_chars=20000,
            ),
            "debug": PackingConfig(
                max_files=3,
                max_lines_per_file=100,
                max_error_lines=100,  # Errors are key for debug
                max_total_chars=25000,
            ),
            "improve": PackingConfig(
                max_files=5,
                max_lines_per_file=250,
                max_error_lines=50,
                max_total_chars=40000,
            ),
        }

        config = stage_configs.get(stage, PackingConfig())
        return ContextPacker.pack(files, facts, errors, constraints, config, task)


# Convenience function for backward compatibility
def pack(
    files: List[str],
    facts: List[str],
    errors: List[str],
    constraints: List[str],
    max_chars: int = 30000,
) -> str:
    """Pack context (backward compatible interface)."""
    config = PackingConfig(max_total_chars=max_chars)
    return ContextPacker.pack(files, facts, errors, constraints, config)
