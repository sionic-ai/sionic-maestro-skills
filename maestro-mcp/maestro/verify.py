"""
Verification Engine for running tests, lint, and type-checks.

Implements the "Test, don't vote" philosophy from:
- "Towards a Science of Scaling Agent Systems": Centralized verification prevents error amplification
- Poetiq ARC Solver: Automated tests > LLM judge for selection

Key principle: Deterministic signals (test pass/fail) are more reliable than
LLM consensus, especially for code correctness.
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("maestro.verify")


class VerificationType(Enum):
    """Types of verification that can be performed."""
    UNIT_TEST = "unit_test"          # pytest, jest, go test, etc.
    INTEGRATION_TEST = "integration"  # Integration tests
    LINT = "lint"                     # ruff, eslint, golangci-lint
    TYPE_CHECK = "type_check"         # mypy, tsc, etc.
    FORMAT = "format"                 # black --check, prettier --check
    CUSTOM = "custom"                 # User-defined command
    BUILD = "build"                   # cargo build, npm run build


@dataclass
class VerificationResult:
    """Result from a single verification command."""
    type: VerificationType
    command: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "command": self.command,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:5000] if len(self.stdout) > 5000 else self.stdout,
            "stderr": self.stderr[:2000] if len(self.stderr) > 2000 else self.stderr,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class VerificationReport:
    """Aggregated report from multiple verifications."""
    results: List[VerificationResult]
    all_passed: bool
    summary: str
    total_duration_ms: float
    critical_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "summary": self.summary,
            "total_duration_ms": self.total_duration_ms,
            "critical_failures": self.critical_failures,
            "results": [r.to_dict() for r in self.results],
            "pass_count": sum(1 for r in self.results if r.passed),
            "fail_count": sum(1 for r in self.results if not r.passed),
        }


# Command templates for common verification tools
VERIFICATION_COMMANDS: Dict[str, Dict[str, str]] = {
    "python": {
        "unit_test": "python -m pytest {args} -v --tb=short",
        "lint": "python -m ruff check {args}",
        "type_check": "python -m mypy {args}",
        "format": "python -m black --check {args}",
    },
    "javascript": {
        "unit_test": "npm test -- {args}",
        "lint": "npm run lint -- {args}",
        "type_check": "npx tsc --noEmit {args}",
        "format": "npx prettier --check {args}",
    },
    "typescript": {
        "unit_test": "npm test -- {args}",
        "lint": "npm run lint -- {args}",
        "type_check": "npx tsc --noEmit {args}",
        "format": "npx prettier --check {args}",
    },
    "go": {
        "unit_test": "go test {args} -v",
        "lint": "golangci-lint run {args}",
        "type_check": "go build {args}",
        "format": "gofmt -l {args}",
    },
    "rust": {
        "unit_test": "cargo test {args}",
        "lint": "cargo clippy {args}",
        "type_check": "cargo check {args}",
        "format": "cargo fmt --check {args}",
        "build": "cargo build {args}",
    },
}

# Security: Allowlist of safe commands
ALLOWED_COMMAND_PREFIXES = [
    "python", "python3", "pytest", "mypy", "ruff", "black", "isort",
    "npm", "npx", "node", "jest", "eslint", "prettier", "tsc",
    "go", "golangci-lint", "gofmt",
    "cargo", "rustfmt", "clippy",
    "make", "cmake",
    "git",  # For git-based checks
]

# Security: Allowlist of safe Python modules for `python -m`
# IMPORTANT: Do NOT add modules that can execute arbitrary code or network access
ALLOWED_PYTHON_MODULES = [
    "pytest",
    "mypy",
    "ruff",
    "black",
    "isort",
    "flake8",
    "pylint",
    "coverage",
    "unittest",
    "doctest",
    "py_compile",
    "compileall",
    "json.tool",  # JSON validation
]

# Explicitly blocked modules (dangerous even if someone adds them accidentally)
BLOCKED_PYTHON_MODULES = [
    "http.server",  # Can expose files
    "pip",  # Can install arbitrary code
    "ensurepip",
    "venv",
    "site",
    "code",  # Interactive interpreter
    "codeop",
    "pdb",  # Debugger (can execute arbitrary code)
    "idlelib",
    "webbrowser",
    "smtplib",
    "ftplib",
    "telnetlib",
    "socketserver",
    "SimpleHTTPServer",  # Python 2
    "BaseHTTPServer",  # Python 2
]


# Shared executor for async verification operations
_SHARED_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_shared_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor for verification."""
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None:
        _SHARED_EXECUTOR = ThreadPoolExecutor(max_workers=4)
    return _SHARED_EXECUTOR


class VerificationEngine:
    """
    Executes verification commands safely and returns structured results.

    Security features:
    - Command allowlist to prevent arbitrary execution
    - Timeout enforcement
    - Output truncation
    - Working directory isolation
    """

    def __init__(
        self,
        timeout_sec: int = 300,
        max_output_chars: int = 50000,
        allowed_commands: Optional[List[str]] = None,
    ):
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
        self.allowed_commands = allowed_commands or ALLOWED_COMMAND_PREFIXES
        # Note: Uses module-level shared executor (_get_shared_executor) for async ops

    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate that a command is safe to execute."""
        # Split command to get the base command
        parts = command.strip().split()
        if not parts:
            return False, "Empty command"

        base_cmd = parts[0]

        # Check against allowlist
        for allowed in self.allowed_commands:
            if base_cmd == allowed or base_cmd.endswith(f"/{allowed}"):
                # Special case: validate `python -m` module is allowed
                if base_cmd in ["python", "python3"] and len(parts) > 2 and parts[1] == "-m":
                    module = parts[2].split(".")[0]  # Get base module name

                    # Check blocklist first
                    if module in BLOCKED_PYTHON_MODULES or parts[2] in BLOCKED_PYTHON_MODULES:
                        return False, f"Module '{parts[2]}' is blocked for security reasons"

                    # Check allowlist
                    if module not in ALLOWED_PYTHON_MODULES and parts[2] not in ALLOWED_PYTHON_MODULES:
                        return False, (
                            f"Module '{parts[2]}' not in allowed modules. "
                            f"Allowed: {', '.join(ALLOWED_PYTHON_MODULES[:5])}..."
                        )

                return True, ""

        return False, f"Command '{base_cmd}' not in allowlist. Allowed: {', '.join(self.allowed_commands[:10])}..."

    def _run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> Tuple[int, str, str, float]:
        """
        Execute a command and return (exit_code, stdout, stderr, duration_ms).

        Uses shell=False for security.
        """
        effective_timeout = timeout_sec or self.timeout_sec
        start_time = time.time()

        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        try:
            # Parse command for shell=False execution
            import shlex
            cmd_parts = shlex.split(command)

            result = subprocess.run(
                cmd_parts,
                cwd=cwd,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                shell=False,  # SECURITY: Never use shell=True
            )

            duration_ms = (time.time() - start_time) * 1000

            # Truncate output if needed
            stdout = result.stdout
            stderr = result.stderr
            if len(stdout) > self.max_output_chars:
                stdout = stdout[:self.max_output_chars] + "\n... [OUTPUT TRUNCATED]"
            if len(stderr) > self.max_output_chars // 2:
                stderr = stderr[:self.max_output_chars // 2] + "\n... [STDERR TRUNCATED]"

            return result.returncode, stdout, stderr, duration_ms

        except subprocess.TimeoutExpired as e:
            duration_ms = (time.time() - start_time) * 1000
            return -1, "", f"TIMEOUT after {effective_timeout}s", duration_ms

        except FileNotFoundError as e:
            duration_ms = (time.time() - start_time) * 1000
            return -1, "", f"Command not found: {e.filename}", duration_ms

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return -1, "", str(e), duration_ms

    def run(
        self,
        command: str,
        verification_type: VerificationType = VerificationType.CUSTOM,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> VerificationResult:
        """
        Run a single verification command.

        Args:
            command: The command to execute
            verification_type: Type of verification (for categorization)
            cwd: Working directory
            env: Additional environment variables
            timeout_sec: Timeout override

        Returns:
            VerificationResult with pass/fail status and output
        """
        # Validate command
        is_valid, error = self._validate_command(command)
        if not is_valid:
            return VerificationResult(
                type=verification_type,
                command=command,
                passed=False,
                exit_code=-1,
                stdout="",
                stderr=f"SECURITY: {error}",
                duration_ms=0,
                metadata={"blocked": True, "reason": error},
            )

        # Execute
        exit_code, stdout, stderr, duration_ms = self._run_command(
            command, cwd, env, timeout_sec
        )

        # Determine pass/fail
        # Most tools: exit code 0 = pass
        passed = exit_code == 0

        return VerificationResult(
            type=verification_type,
            command=command,
            passed=passed,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
        )

    def run_multiple(
        self,
        commands: List[Dict[str, Any]],
        cwd: Optional[str] = None,
        stop_on_failure: bool = False,
    ) -> VerificationReport:
        """
        Run multiple verification commands.

        Args:
            commands: List of command specs:
                      [{"command": "...", "type": "unit_test", "timeout_sec": 60}, ...]
            cwd: Default working directory
            stop_on_failure: Stop after first failure

        Returns:
            VerificationReport aggregating all results
        """
        results = []
        total_start = time.time()
        critical_failures = []

        for cmd_spec in commands:
            command = cmd_spec.get("command", "")
            v_type = VerificationType(cmd_spec.get("type", "custom"))
            timeout = cmd_spec.get("timeout_sec")
            cmd_cwd = cmd_spec.get("cwd", cwd)

            result = self.run(
                command=command,
                verification_type=v_type,
                cwd=cmd_cwd,
                timeout_sec=timeout,
            )
            results.append(result)

            # Track critical failures (tests and type checks)
            if not result.passed and v_type in [
                VerificationType.UNIT_TEST,
                VerificationType.TYPE_CHECK,
            ]:
                critical_failures.append(f"{v_type.value}: {command}")

            if stop_on_failure and not result.passed:
                break

        total_duration = (time.time() - total_start) * 1000
        all_passed = all(r.passed for r in results)

        # Build summary
        pass_count = sum(1 for r in results if r.passed)
        fail_count = len(results) - pass_count
        summary = f"{pass_count}/{len(results)} checks passed"
        if fail_count > 0:
            failed_types = [r.type.value for r in results if not r.passed]
            summary += f" (failed: {', '.join(failed_types)})"

        return VerificationReport(
            results=results,
            all_passed=all_passed,
            summary=summary,
            total_duration_ms=total_duration,
            critical_failures=critical_failures,
        )

    async def run_parallel(
        self,
        commands: List[Dict[str, Any]],
        cwd: Optional[str] = None,
    ) -> VerificationReport:
        """
        Run multiple verification commands in parallel.

        Useful for independent checks like lint + type-check.
        """
        loop = asyncio.get_event_loop()
        total_start = time.time()

        async def run_one(cmd_spec: Dict[str, Any]) -> VerificationResult:
            command = cmd_spec.get("command", "")
            v_type = VerificationType(cmd_spec.get("type", "custom"))
            timeout = cmd_spec.get("timeout_sec")
            cmd_cwd = cmd_spec.get("cwd", cwd)

            return await loop.run_in_executor(
                _get_shared_executor(),
                lambda: self.run(command, v_type, cmd_cwd, timeout_sec=timeout),
            )

        results = await asyncio.gather(*[run_one(cmd) for cmd in commands])
        total_duration = (time.time() - total_start) * 1000

        all_passed = all(r.passed for r in results)
        critical_failures = [
            f"{r.type.value}: {r.command}"
            for r in results
            if not r.passed and r.type in [VerificationType.UNIT_TEST, VerificationType.TYPE_CHECK]
        ]

        pass_count = sum(1 for r in results if r.passed)
        summary = f"{pass_count}/{len(results)} checks passed"

        return VerificationReport(
            results=list(results),
            all_passed=all_passed,
            summary=summary,
            total_duration_ms=total_duration,
            critical_failures=critical_failures,
        )

    def detect_project_type(self, cwd: str) -> Optional[str]:
        """Detect project type from common config files."""
        path = Path(cwd)

        if (path / "pyproject.toml").exists() or (path / "setup.py").exists():
            return "python"
        if (path / "package.json").exists():
            # Check for TypeScript
            if (path / "tsconfig.json").exists():
                return "typescript"
            return "javascript"
        if (path / "go.mod").exists():
            return "go"
        if (path / "Cargo.toml").exists():
            return "rust"

        return None

    def get_default_commands(
        self,
        project_type: str,
        include_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get default verification commands for a project type.

        Args:
            project_type: 'python', 'javascript', 'typescript', 'go', 'rust'
            include_types: Which verification types to include
                           (default: unit_test, lint)
        """
        if project_type not in VERIFICATION_COMMANDS:
            return []

        commands = VERIFICATION_COMMANDS[project_type]
        include = include_types or ["unit_test", "lint"]

        result = []
        for v_type in include:
            if v_type in commands:
                result.append({
                    "command": commands[v_type].format(args="."),
                    "type": v_type,
                })

        return result


def parse_test_output(output: str, test_framework: str = "pytest") -> Dict[str, Any]:
    """
    Parse test output to extract structured results.

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "skipped": int,
            "errors": int,
            "failed_tests": ["test_name", ...],
            "duration_sec": float
        }
    """
    result = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "failed_tests": [],
        "duration_sec": 0.0,
    }

    if test_framework == "pytest":
        # Parse pytest output
        # Example: "5 passed, 2 failed, 1 skipped in 1.23s"
        summary_match = re.search(
            r'(\d+) passed.*?(\d+) failed.*?(\d+) skipped.*?in ([\d.]+)s',
            output
        )
        if summary_match:
            result["passed"] = int(summary_match.group(1))
            result["failed"] = int(summary_match.group(2))
            result["skipped"] = int(summary_match.group(3))
            result["duration_sec"] = float(summary_match.group(4))
            result["total"] = result["passed"] + result["failed"] + result["skipped"]

        # Alternative pattern: "5 passed in 1.23s"
        if result["total"] == 0:
            simple_match = re.search(r'(\d+) passed.*?in ([\d.]+)s', output)
            if simple_match:
                result["passed"] = int(simple_match.group(1))
                result["total"] = result["passed"]
                result["duration_sec"] = float(simple_match.group(2))

        # Extract failed test names
        failed_pattern = re.findall(r'FAILED ([\w_/:.]+)', output)
        result["failed_tests"] = failed_pattern

    elif test_framework == "jest":
        # Parse Jest output
        # Example: "Tests: 2 failed, 5 passed, 7 total"
        match = re.search(r'Tests:\s*(\d+) failed,\s*(\d+) passed,\s*(\d+) total', output)
        if match:
            result["failed"] = int(match.group(1))
            result["passed"] = int(match.group(2))
            result["total"] = int(match.group(3))

    elif test_framework == "go":
        # Parse go test output
        # Count PASS/FAIL lines
        result["passed"] = len(re.findall(r'^--- PASS:', output, re.MULTILINE))
        result["failed"] = len(re.findall(r'^--- FAIL:', output, re.MULTILINE))
        result["total"] = result["passed"] + result["failed"]

    return result


def parse_lint_output(output: str, linter: str = "ruff") -> Dict[str, Any]:
    """
    Parse lint output to extract issue counts.

    Returns:
        {
            "total_issues": int,
            "errors": int,
            "warnings": int,
            "issues": [{"file": "...", "line": int, "message": "..."}]
        }
    """
    result = {
        "total_issues": 0,
        "errors": 0,
        "warnings": 0,
        "issues": [],
    }

    if linter == "ruff":
        # Parse ruff output: "file.py:10:5: E501 Line too long"
        issues = re.findall(r'([^:]+):(\d+):(\d+):\s*(\w+)\s+(.+)', output)
        for file, line, col, code, message in issues:
            result["issues"].append({
                "file": file,
                "line": int(line),
                "column": int(col),
                "code": code,
                "message": message,
            })
            if code.startswith("E"):
                result["errors"] += 1
            else:
                result["warnings"] += 1
        result["total_issues"] = len(issues)

    elif linter == "eslint":
        # Parse eslint JSON output or standard output
        issues = re.findall(r'(\d+):(\d+)\s+(error|warning)\s+(.+)', output)
        for line, col, severity, message in issues:
            result["issues"].append({
                "line": int(line),
                "column": int(col),
                "severity": severity,
                "message": message,
            })
            if severity == "error":
                result["errors"] += 1
            else:
                result["warnings"] += 1
        result["total_issues"] = len(issues)

    return result
