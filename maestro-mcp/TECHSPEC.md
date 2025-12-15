# Maestro MCP - Technical Specification for Improvements

## Overview

This document specifies implementation details for priority improvements identified in the critical analysis comparing maestro-mcp with sionic-mcp.

**Reference**: `CRITICAL_ANALYSIS.md`

---

## Priority Matrix

| Priority | Feature | Effort | Risk | Dependency |
|----------|---------|--------|------|------------|
| **P0** | Secret Masking | Low | High (security) | None |
| **P0** | JSON Schema Validation | Low | Medium | `jsonschema` lib |
| **P0** | Test Suite | Medium | Low | `pytest` |
| **P1** | Git State Helper | Low | Low | None |
| **P1** | Artifact System | Medium | Low | None |
| **P2** | Module Consolidation | High | Medium | All above |

---

## P0-1: Secret Masking (`MASK_REGEXES`)

### Problem

Prompts sent to external CLIs (Codex/Gemini/Claude) may contain sensitive data:
- API keys, tokens
- Passwords, credentials
- Connection strings
- Private paths

Currently, maestro-mcp sends prompts **without sanitization** — a security risk.

### Solution

Add regex-based masking before any external CLI call.

### Implementation

#### 1. Add to `maestro/config.py`

Following the existing pattern of nested dataclasses (`ProviderConfig`, `CoordinationPolicy`, etc.):

```python
import re
from typing import List, Pattern

@dataclass
class MaskingConfig:
    """Configuration for secret masking in prompts."""
    mask_regexes: List[Pattern] = field(default_factory=list)
    mask_replacement: str = "***MASKED***"


@dataclass
class MaestroConfig:
    # ... existing nested configs ...
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    policy: CoordinationPolicy = field(default_factory=CoordinationPolicy)
    context: ContextConfig = field(default_factory=ContextConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)

    # NEW: Secret masking (follows same nested pattern)
    masking: MaskingConfig = field(default_factory=MaskingConfig)

    @classmethod
    def from_env(cls) -> "MaestroConfig":
        # ... existing code ...

        # Parse MASK_REGEXES from env (comma-separated patterns)
        mask_patterns_str = os.getenv("MAESTRO_MASK_REGEXES", "")
        mask_regexes = []
        if mask_patterns_str:
            for pattern in mask_patterns_str.split(","):
                pattern = pattern.strip()
                if pattern:
                    try:
                        mask_regexes.append(re.compile(pattern, re.IGNORECASE))
                    except re.error:
                        logging.warning(f"Invalid mask regex: {pattern}")

        # Add default patterns (no inline (?i) - using IGNORECASE flag instead)
        default_patterns = [
            r"(api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w-]+['\"]?",
            r"(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?",
            r"(secret|token)\s*[:=]\s*['\"]?[\w-]+['\"]?",
            r"(bearer)\s+[\w-]+",
            r"(authorization)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?",
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub tokens
            r"gho_[a-zA-Z0-9]{36}",  # GitHub OAuth tokens
            r"glpat-[a-zA-Z0-9-]{20}",  # GitLab tokens
        ]
        for pattern in default_patterns:
            mask_regexes.append(re.compile(pattern, re.IGNORECASE))

        masking = MaskingConfig(
            mask_regexes=mask_regexes,
            mask_replacement=os.getenv("MAESTRO_MASK_REPLACEMENT", "***MASKED***"),
        )

        return cls(
            # ... existing fields ...
            masking=masking,
        )
```

#### 2. Add masking utility to `maestro/providers.py`

```python
def mask_secrets(text: str, config: MaestroConfig) -> str:
    """Mask sensitive data in text before sending to external CLIs."""
    result = text
    for pattern in config.masking.mask_regexes:  # Use nested config
        result = pattern.sub(config.masking.mask_replacement, result)
    return result
```

#### 3. Apply in all provider `consult()` methods

```python
class CodexProvider(BaseProvider):
    async def consult(self, prompt: str, ...) -> ProviderResponse:
        # Mask secrets before sending
        safe_prompt = mask_secrets(prompt, self.config)

        # ... existing CLI call with safe_prompt ...
```

### Environment Variables

```bash
# Custom patterns (comma-separated regexes)
MAESTRO_MASK_REGEXES="my-secret-\d+,internal-token-[a-z]+"

# Custom replacement text
MAESTRO_MASK_REPLACEMENT="[REDACTED]"
```

### Testing

```python
def test_mask_secrets():
    config = MaestroConfig(mask_regexes=[re.compile(r"sk-\w+")])
    text = "Use API key sk-abc123xyz"
    assert mask_secrets(text, config) == "Use API key ***MASKED***"

def test_mask_multiple_patterns():
    # Test default patterns mask API keys, passwords, tokens
    ...
```

---

## P0-2: JSON Schema Validation

### Problem

Current implementation has basic field checking but no proper JSON Schema validation:

```python
# Current (weak)
def _check_required_fields(content, schema):
    required = schema.get("required", [])
    return [f for f in required if f not in data]
```

sionic-mcp uses `jsonschema.Draft202012Validator` for proper validation.

### Solution

Add `jsonschema` dependency and use it for all schema validation.

### Implementation

#### 1. Update `requirements.txt`

```
jsonschema>=4.17.0  # Required for Draft202012Validator
```

#### 2. Add validation utility to `maestro/selection.py`

```python
from jsonschema import Draft202012Validator, ValidationError

def validate_json_schema(data: dict, schema: dict) -> tuple[bool, str]:
    """
    Validate data against JSON schema.

    Returns:
        (is_valid, error_message)
    """
    try:
        validator = Draft202012Validator(schema)
        errors = list(validator.iter_errors(data))
        if errors:
            # Return first error for simplicity
            return False, errors[0].message
        return True, ""
    except Exception as e:
        return False, f"Schema validation error: {str(e)}"
```

#### 3. Update `RedFlagger` in `maestro/maker.py`

```python
from jsonschema import Draft202012Validator, ValidationError

class RedFlagger:
    def _check_required_fields(self, content: str, schema: Dict) -> List[str]:
        """Check if JSON content validates against schema."""
        try:
            # Extract JSON from content (non-greedy to avoid over-matching)
            # Note: r'\{[\s\S]*\}' is too greedy - matches first { to LAST }
            # Using non-greedy or iterative approach instead
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
            if not json_match:
                # Fallback: try to find valid JSON by parsing incrementally
                start = content.find('{')
                if start == -1:
                    return ["No JSON object found"]
                for end in range(len(content) - 1, start, -1):
                    if content[end] == '}':
                        try:
                            data = json.loads(content[start:end+1])
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    return ["No valid JSON object found"]
            else:
                data = json.loads(json_match.group())

            # Use jsonschema for validation
            validator = Draft202012Validator(schema)
            errors = list(validator.iter_errors(data))

            return [e.message for e in errors[:3]]  # Return first 3 errors

        except json.JSONDecodeError as e:
            return [f"Invalid JSON: {str(e)}"]
        except Exception as e:
            return [f"Validation error: {str(e)}"]
```

#### 4. Add schema validation to stage output checking

```python
# In maestro/workflow.py
def validate_stage_output(output: dict, stage: str) -> tuple[bool, str]:
    """Validate stage output against its schema."""
    schema = load_stage_schema(stage)
    if not schema:
        return True, ""  # No schema = no validation

    return validate_json_schema(output, schema)
```

### Stage Schemas

Ensure all stage schemas in `schemas/` are proper JSON Schema:

```json
// schemas/stage2_output.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["hypotheses"],
  "properties": {
    "hypotheses": {
      "type": "array",
      "minItems": 1,
      "maxItems": 5,
      "items": {
        "type": "object",
        "required": ["id", "claim", "confidence", "fast_test"],
        "properties": {
          "id": { "type": "string", "pattern": "^H\\d+$" },
          "claim": { "type": "string", "minLength": 10 },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
          "fast_test": { "type": "string", "minLength": 5 }
        }
      }
    }
  }
}
```

### Testing

```python
def test_valid_hypothesis_output():
    output = {
        "hypotheses": [
            {"id": "H1", "claim": "Null pointer in auth", "confidence": 0.8, "fast_test": "Check line 42"}
        ]
    }
    schema = load_stage_schema("hypothesize")
    is_valid, error = validate_json_schema(output, schema)
    assert is_valid

def test_invalid_hypothesis_output():
    output = {"hypotheses": [{"id": "H1"}]}  # Missing required fields
    schema = load_stage_schema("hypothesize")
    is_valid, error = validate_json_schema(output, schema)
    assert not is_valid
    assert "claim" in error or "confidence" in error
```

---

## P0-3: Test Suite

### Problem

Zero test files. Unacceptable for production code.

### Solution

Create comprehensive pytest test suite.

### Directory Structure

```
maestro-mcp/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_config.py           # MaestroConfig tests
│   ├── test_providers.py        # Provider tests (mocked CLIs)
│   ├── test_selection.py        # Selection engine tests
│   ├── test_coordination.py     # Architecture selection tests
│   ├── test_skills.py           # Dynamic tool loading tests
│   ├── test_workspace.py        # Patch application tests
│   ├── test_integration.py      # End-to-end tests
│   └── maker/                   # MAKER module tests (subpackage)
│       ├── __init__.py
│       ├── test_red_flagger.py
│       ├── test_vote_step.py
│       └── test_calibrator.py
├── pytest.ini
└── requirements-dev.txt
```

> **Note**: Python doesn't allow `.py` files to contain other files. Use either:
> - Flat structure: All `test_*.py` in `tests/`
> - Package structure: `tests/maker/` directory with `__init__.py`

### Implementation

#### 1. `requirements-dev.txt`

```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
```

#### 2. `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
asyncio_mode = auto
addopts = -v --cov=maestro --cov-report=term-missing
```

#### 3. `tests/conftest.py`

```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from maestro.config import MaestroConfig
from maestro.providers import ProviderRegistry


@pytest.fixture
def config():
    """Default test configuration."""
    return MaestroConfig(
        codex_cmd="echo",  # Mock with echo
        gemini_cmd="echo",
        claude_cmd="echo",
        timeout=5,
    )


@pytest.fixture
def mock_provider_registry(config):
    """Registry with mocked providers."""
    registry = ProviderRegistry(config)

    # Mock all providers to return predictable responses
    for provider in registry.providers.values():
        provider.consult = AsyncMock(return_value=ProviderResponse(
            content="Mock response",
            provider="mock",
            model="mock-model",
            elapsed_ms=100,
        ))

    return registry


@pytest.fixture
def sample_context():
    """Sample context pack for testing."""
    return {
        "task": "Fix the authentication bug",
        "files": [{"path": "auth.py", "content": "def login(): pass"}],
        "errors": ["NullPointerException at line 42"],
    }


@pytest.fixture
def temp_workspace(tmp_path):
    """Temporary workspace for file operations."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create sample files
    (workspace / "main.py").write_text("print('hello')")
    (workspace / "test.py").write_text("def test_main(): pass")

    return workspace
```

#### 4. `tests/test_maker.py` (Example)

```python
import pytest
from maestro.maker import (
    RedFlagger, RedFlaggerConfig, RedFlagReason,
    VoteStep, VoteResult,
    MicroStepType, MICRO_STEP_SPECS,
)


class TestRedFlagger:
    @pytest.fixture
    def flagger(self):
        return RedFlagger()

    def test_too_long_content(self, flagger):
        content = "x" * 15000
        result = flagger.validate(content)
        assert result.is_flagged
        assert RedFlagReason.TOO_LONG in result.reasons

    def test_too_short_content(self, flagger):
        content = "hi"
        result = flagger.validate(content)
        assert result.is_flagged
        assert RedFlagReason.TOO_SHORT in result.reasons

    def test_hedging_detection(self, flagger):
        content = "I'm not sure, but maybe the bug is in auth.py"
        result = flagger.validate(content, rules=["hedging"])
        assert result.is_flagged
        assert RedFlagReason.HEDGING in result.reasons

    def test_dangerous_code_detection(self, flagger):
        content = "eval(user_input)"
        result = flagger.validate(content, rules=["dangerous_code"])
        assert result.is_flagged
        assert RedFlagReason.DANGEROUS_CODE in result.reasons

    def test_valid_content_passes(self, flagger):
        content = "The root cause is a null check missing at line 42 in UserService.java"
        result = flagger.validate(content)
        assert not result.is_flagged


class TestVoteStep:
    def test_single_unanimous_vote(self):
        voter = VoteStep(k=3, max_rounds=10)

        # All samples return same answer
        call_count = [0]
        def sample_fn():
            call_count[0] += 1
            return ("answer_a", "provider1", "raw")

        result = voter.vote(sample_fn)

        assert result.winner is not None
        assert result.winner.content == "answer_a"
        assert result.converged
        assert call_count[0] >= 3  # At least k samples

    def test_competing_answers(self):
        voter = VoteStep(k=2, max_rounds=20)

        # Alternating answers, but A slightly more common
        call_count = [0]
        def sample_fn():
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return ("answer_b", "provider1", "raw")
            return ("answer_a", "provider1", "raw")

        result = voter.vote(sample_fn)

        assert result.winner is not None
        assert result.winner.content == "answer_a"

    def test_red_flagged_samples_discarded(self):
        voter = VoteStep(k=2, max_rounds=10)

        call_count = [0]
        def sample_fn():
            call_count[0] += 1
            if call_count[0] <= 3:
                return ("x", "p", "r")  # Too short, will be flagged
            return ("valid answer that is long enough", "p", "r")

        result = voter.vote(sample_fn)

        assert result.red_flagged_count >= 3


class TestMicroStepSpecs:
    def test_all_stages_have_steps(self):
        from maestro.maker import StageType

        for stage in StageType:
            steps = [s for s in MICRO_STEP_SPECS.values() if s.stage == stage]
            assert len(steps) >= 1, f"Stage {stage} has no micro-steps"

    def test_all_steps_have_schema(self):
        for step_type, spec in MICRO_STEP_SPECS.items():
            assert spec.output_schema is not None
            assert "type" in spec.output_schema
            assert "required" in spec.output_schema or "properties" in spec.output_schema
```

#### 5. `tests/test_coordination.py`

```python
import pytest
from maestro.coordination import (
    ArchitectureSelectionEngine,
    TaskStructureFeatures,
    CoordinationTopology,
    TaskStructureClassifier,
    MetricsTracker,
)


class TestArchitectureSelection:
    @pytest.fixture
    def engine(self):
        return ArchitectureSelectionEngine()

    def test_high_sequential_returns_sas(self, engine):
        features = TaskStructureFeatures(
            sequential_dependency_score=0.9,
            decomposability_score=0.3,
        )
        decision = engine.select_architecture(features)
        assert decision.topology == CoordinationTopology.SAS

    def test_high_decomposability_returns_mas(self, engine):
        features = TaskStructureFeatures(
            sequential_dependency_score=0.2,
            decomposability_score=0.8,
            tool_complexity=0.3,
        )
        decision = engine.select_architecture(features)
        assert decision.topology == CoordinationTopology.MAS_INDEPENDENT

    def test_high_tool_complexity_returns_centralized(self, engine):
        features = TaskStructureFeatures(
            sequential_dependency_score=0.2,
            decomposability_score=0.8,
            tool_complexity=0.8,
        )
        decision = engine.select_architecture(features)
        assert decision.topology == CoordinationTopology.MAS_CENTRALIZED

    def test_debug_stage_always_sas(self, engine):
        features = TaskStructureFeatures(decomposability_score=0.9)
        decision = engine.select_architecture(features, stage="debug")
        # Debug stage recommendation overrides
        rec = engine.get_stage_recommendation("debug")
        assert rec["topology"] == "sas"


class TestTaskClassifier:
    @pytest.fixture
    def classifier(self):
        return TaskStructureClassifier()

    def test_debug_task_high_sequential(self, classifier):
        features = classifier.classify(
            "Debug the failing login test and fix the bug step by step"
        )
        assert features.sequential_dependency_score >= 0.5
        assert features.domain == "debugging"

    def test_refactor_task_decomposable(self, classifier):
        features = classifier.classify(
            "Refactor multiple files to use the new API pattern"
        )
        assert features.decomposability_score >= 0.4
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=maestro --cov-report=html

# Run specific test file
pytest tests/test_maker.py

# Run specific test
pytest tests/test_maker.py::TestRedFlagger::test_hedging_detection
```

---

## P1-1: Git State Helper

### Problem

No helper to capture git repository state for context/audit.

### Solution

Add `git_state()` function similar to sionic-mcp.

### Implementation

#### Add to `maestro/workspace.py`

```python
import subprocess
from typing import Optional, Dict, Any


def git_state(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """
    Capture current git repository state.

    Returns dict with:
    - root: Repository root path
    - branch: Current branch name
    - commit: Current commit SHA
    - short_commit: Short commit SHA
    - status: Porcelain status output
    - is_dirty: Whether working tree has changes
    - diff_stat: Diff statistics
    - remote_url: Origin remote URL (if exists)

    Returns empty dict if not in a git repository.
    """
    def run_git(args: list[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    # Check if in a git repo
    root = run_git(["rev-parse", "--show-toplevel"])
    if not root:
        return {}

    state = {"root": root}

    # Branch (handle detached HEAD state)
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        state["branch"] = branch
        # Detect detached HEAD state
        if branch == "HEAD":
            state["is_detached_head"] = True
            # Try to get more context about detached state
            describe = run_git(["describe", "--tags", "--always"])
            if describe:
                state["detached_at"] = describe

    # Commit
    commit = run_git(["rev-parse", "HEAD"])
    if commit:
        state["commit"] = commit
        state["short_commit"] = commit[:8]

    # Status
    status = run_git(["status", "--porcelain"])
    if status is not None:
        state["status"] = status
        state["is_dirty"] = len(status) > 0

    # Diff stat
    diff_stat = run_git(["diff", "--stat"])
    if diff_stat:
        state["diff_stat"] = diff_stat

    # Remote URL
    remote_url = run_git(["remote", "get-url", "origin"])
    if remote_url:
        state["remote_url"] = remote_url

    # Recent commits (last 3)
    log = run_git(["log", "--oneline", "-3"])
    if log:
        state["recent_commits"] = log.split("\n")

    return state
```

#### Add MCP tool in `server.py`

```python
@mcp.tool()
def maestro_git_state(
    path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get current git repository state.

    Returns branch, commit, status, and other git information.
    Useful for context packing and audit trails.

    Args:
        path: Optional path within the repository

    Returns:
        Git state dict or empty dict if not in a repo
    """
    from maestro.workspace import git_state
    return git_state(Path(path) if path else None)
```

#### Use in context packing

```python
# In maestro/context.py
def pack_context(self, ...) -> Dict[str, Any]:
    context = {
        "task": task,
        "files": files,
        # ... existing fields ...
    }

    # Add git state if available
    git = git_state()
    if git:
        context["git"] = {
            "branch": git.get("branch"),
            "commit": git.get("short_commit"),
            "is_dirty": git.get("is_dirty", False),
        }

    return context
```

---

## P1-2: Artifact System

### Problem

No persistence of run artifacts (stage results, policy decisions, recommendations).

### Solution

Add artifact directory with JSONL logging per run.

### Implementation

#### 1. Add to `maestro/config.py`

```python
@dataclass
class MaestroConfig:
    # ... existing fields ...

    # Artifacts
    artifact_dir: Path = Path(".maestro-artifacts")
    artifact_enabled: bool = True
```

#### 2. Create `maestro/artifacts.py`

```python
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class ArtifactMetadata:
    run_id: str
    created_at: str
    task: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None


@dataclass
class RunContext:
    """Thread-safe context for a single run (avoids race conditions)."""
    run_id: str
    run_dir: Path

    def save_stage_result(self, stage: str, result: Dict[str, Any]):
        """Save stage output."""
        self._write_json(f"stage_{stage}.json", result)

    def save_policy_decision(self, decision: Dict[str, Any]):
        """Append policy decision to JSONL."""
        self._append_jsonl("policy_decisions.jsonl", decision)

    def _write_json(self, filename: str, data: Dict):
        path = self.run_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _append_jsonl(self, filename: str, data: Dict):
        path = self.run_dir / filename
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")


class ArtifactStore:
    """
    Persists run artifacts for audit and replay.

    Directory structure:
    .maestro-artifacts/
    ├── {run_id}/
    │   ├── metadata.json
    │   ├── stage_analyze.json
    │   ├── stage_hypothesize.json
    │   ├── stage_implement.json
    │   ├── stage_debug.json
    │   ├── stage_improve.json
    │   ├── policy_decisions.jsonl
    │   ├── coordination_metrics.json
    │   └── final_recommendation.json

    Note: Returns RunContext objects to avoid race conditions with concurrent runs.
    Each RunContext is independent and thread-safe.
    """

    def __init__(self, base_dir: Path, enabled: bool = True):
        self.base_dir = base_dir
        self.enabled = enabled
        # Note: Removed instance-level current_run_id/_run_dir to prevent race conditions
        # Instead, start_run() returns a RunContext that callers should use

    def start_run(self, task: str, git_state: Optional[Dict] = None) -> RunContext:
        """
        Start a new run and return a RunContext for thread-safe artifact writing.

        Returns:
            RunContext object that should be used for all artifact operations.
            This avoids race conditions when multiple runs happen concurrently.
        """
        if not self.enabled:
            return None

        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata = ArtifactMetadata(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            task=task,
            git_commit=git_state.get("short_commit") if git_state else None,
            git_branch=git_state.get("branch") if git_state else None,
        )
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

        return RunContext(run_id=run_id, run_dir=run_dir)

    def list_runs(self, limit: int = 10) -> list[Dict[str, Any]]:
        """List recent runs."""
        if not self.base_dir.exists():
            return []

        runs = []
        for run_dir in sorted(self.base_dir.iterdir(), reverse=True)[:limit]:
            if run_dir.is_dir():
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        runs.append(json.load(f))

        return runs

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load all artifacts for a run."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            return {}

        artifacts = {}
        for path in run_dir.iterdir():
            if path.suffix == ".json":
                with open(path) as f:
                    artifacts[path.stem] = json.load(f)
            elif path.suffix == ".jsonl":
                with open(path) as f:
                    artifacts[path.stem] = [json.loads(line) for line in f]

        return artifacts
```

#### 3. Add MCP tools

```python
# Global store for active run contexts (keyed by run_id)
_active_runs: Dict[str, RunContext] = {}

@mcp.tool()
def maestro_start_artifact_run(task: str) -> Dict[str, str]:
    """Start a new artifact run for the given task.

    Returns a run_id that should be passed to subsequent artifact operations.
    This design avoids race conditions when multiple runs happen concurrently.
    """
    from maestro.workspace import git_state
    ctx = artifact_store.start_run(task, git_state())
    if ctx:
        _active_runs[ctx.run_id] = ctx
        return {"run_id": ctx.run_id, "enabled": True}
    return {"run_id": "", "enabled": False}

@mcp.tool()
def maestro_save_stage_artifact(run_id: str, stage: str, result: Dict) -> Dict[str, bool]:
    """Save stage result to an active artifact run."""
    ctx = _active_runs.get(run_id)
    if ctx:
        ctx.save_stage_result(stage, result)
        return {"success": True}
    return {"success": False, "error": f"No active run: {run_id}"}

@mcp.tool()
def maestro_end_artifact_run(run_id: str) -> Dict[str, str]:
    """End an artifact run and return the artifact directory."""
    ctx = _active_runs.pop(run_id, None)
    if ctx:
        return {"run_id": run_id, "path": str(ctx.run_dir)}
    return {"run_id": run_id, "error": "Run not found"}

@mcp.tool()
def maestro_list_artifact_runs(limit: int = 10) -> List[Dict]:
    """List recent artifact runs."""
    return artifact_store.list_runs(limit)

@mcp.tool()
def maestro_load_artifact_run(run_id: str) -> Dict[str, Any]:
    """Load all artifacts from a previous run."""
    return artifact_store.load_run(run_id)
```

### Environment Variables

```bash
MAESTRO_ARTIFACT_DIR=.maestro-artifacts
MAESTRO_ARTIFACT_ENABLED=true
```

---

## P2-1: Module Consolidation

### Problem

Over-engineered with too many abstractions:
- `CodexProvider`, `GeminiProvider`, `ClaudeProvider` → could be one function
- `ContextPacker`, `WorkflowEngine`, `SelectionEngine` → add complexity
- 13 separate modules vs sionic-mcp's 1 file

### Solution

Consolidate into 3-4 core modules while preserving functionality.

### Target Structure

```
maestro-mcp/
├── server.py              # MCP tools (unchanged)
└── maestro/
    ├── __init__.py
    ├── core.py            # Config + Providers + Context (merged)
    ├── workflow.py        # Workflow + Selection + Verification (merged)
    ├── maker.py           # MAKER module (unchanged - well-structured)
    └── coordination.py    # Coordination (unchanged - well-structured)
```

### Consolidation Plan

#### Phase 1: Merge `config.py` + `providers.py` → `core.py`

```python
# maestro/core.py

# --- Config section ---
@dataclass
class MaestroConfig:
    # All config fields
    ...

# --- Provider section ---
def call_provider(
    provider: str,
    prompt: str,
    model: Optional[str] = None,
    config: Optional[MaestroConfig] = None,
    timeout: Optional[int] = None,
) -> ProviderResponse:
    """Universal provider caller (replaces 3 provider classes)."""
    config = config or MaestroConfig.from_env()

    if provider == "codex":
        cmd = [config.codex_cmd, "exec", "--model", model or config.codex_model]
        ...
    elif provider == "gemini":
        cmd = [config.gemini_cmd, "-p", prompt, "--model", model or config.gemini_model]
        ...
    elif provider == "claude":
        cmd = [config.claude_cmd, "-p", prompt, "--model", model or config.claude_model]
        ...

    # Run subprocess, parse output, return ProviderResponse
    ...

# --- Context section ---
def pack_context(
    task: str,
    files: List[Dict] = None,
    errors: List[str] = None,
    constraints: List[str] = None,
    git: bool = True,
) -> Dict[str, Any]:
    """Pack context for LLM consultation."""
    ...
```

#### Phase 2: Merge `workflow.py` + `selection.py` + `verify.py` → `workflow.py`

```python
# maestro/workflow.py

# --- Stage definitions ---
STAGES = ["analyze", "hypothesize", "implement", "debug", "improve"]

# --- Selection ---
def select_best(
    candidates: List[Dict],
    mode: str = "tests_first",
    test_results: Optional[List[Dict]] = None,
) -> Dict:
    """Select best candidate."""
    ...

# --- Verification ---
def verify(commands: List[Dict]) -> Dict:
    """Run verification commands."""
    ...

# --- Workflow runner ---
def run_stage(
    stage: str,
    task: str,
    context: Dict,
    use_ensemble: bool = False,
) -> Dict:
    """Run a workflow stage."""
    ...
```

#### Phase 3: Keep `maker.py` and `coordination.py` as-is

These modules are well-structured and implement specific paper concepts.

### Migration Strategy

1. Create new consolidated modules alongside existing
2. Update `server.py` imports one at a time
3. Add deprecation warnings to old modules
4. Remove old modules after verification

### Deprecation Phase (Required)

To avoid breaking external imports, add re-exports with warnings:

```python
# maestro/providers.py (DEPRECATED - keep for backwards compatibility)
import warnings
from .core import call_provider, ProviderResponse, MaestroConfig

# Re-export for backwards compatibility
__all__ = ["CodexProvider", "GeminiProvider", "ClaudeProvider", "ProviderRegistry"]

class CodexProvider:
    """DEPRECATED: Use maestro.core.call_provider() instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CodexProvider is deprecated. Use maestro.core.call_provider('codex', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._args = args
        self._kwargs = kwargs

    def run(self, prompt: str, model: Optional[str] = None, **kwargs) -> ProviderResponse:
        return call_provider("codex", prompt, model=model, **kwargs)


class GeminiProvider:
    """DEPRECATED: Use maestro.core.call_provider() instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GeminiProvider is deprecated. Use maestro.core.call_provider('gemini', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def run(self, prompt: str, model: Optional[str] = None, **kwargs) -> ProviderResponse:
        return call_provider("gemini", prompt, model=model, **kwargs)


# Similar for ClaudeProvider and ProviderRegistry...
```

This allows existing code like `from maestro.providers import CodexProvider` to continue
working while emitting deprecation warnings, giving users time to migrate.

### Risk Mitigation

- All tests must pass after each consolidation step
- Keep git history clean with atomic commits
- Document breaking changes in CHANGELOG.md
- Keep deprecated modules for at least 2 minor versions before removal

---

## Implementation Order

Tasks should be completed in this order (no time estimates - let users schedule):

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| 1 | P0-1 (Secret Masking), P0-2 (JSON Schema) | None |
| 2 | P0-3 (Test Suite - core tests) | Phase 1 |
| 3 | P0-3 (Test Suite - remaining), P1-1 (Git State) | Phase 2 |
| 4 | P1-2 (Artifact System) | Phase 1 |
| 5 | P2-1 (Module Consolidation) | All P0 and P1 complete |

---

## Success Criteria

### P0 Complete
- [ ] All prompts sanitized before external CLI calls
- [ ] All stage outputs validated against JSON Schema
- [ ] Test coverage > 70%
- [ ] All tests pass in CI

### P1 Complete
- [ ] `maestro_git_state` tool available and working
- [ ] Artifacts saved for each run
- [ ] Can replay/audit previous runs

### P2 Complete
- [ ] Module count reduced from 13 to 4-5
- [ ] No functionality lost
- [ ] All tests still pass
- [ ] Documentation updated

---

## Appendix: Existing Code Fixes

### Fix Deprecated `asyncio.get_event_loop()` in `maestro/providers.py`

Current code at line 193-198 uses deprecated `asyncio.get_event_loop()`:

```python
# CURRENT (deprecated in Python 3.10+, removed in 3.12)
async def run_async(self, prompt, ...):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, ...)
```

**Fix** - Use `asyncio.to_thread()` (Python 3.9+):

```python
# FIXED
async def run_async(
    self,
    prompt: str,
    model: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    cwd: Optional[str] = None,
) -> ProviderResponse:
    """Execute the CLI command asynchronously."""
    # asyncio.to_thread() is the modern replacement for run_in_executor
    # Available since Python 3.9, handles thread pool internally
    return await asyncio.to_thread(
        self.run, prompt, model, timeout_sec, output_schema, cwd
    )
```

This should be fixed as part of P2-1 (Module Consolidation) when merging into `core.py`.

---

## References

- `CRITICAL_ANALYSIS.md` - Comparison with sionic-mcp
- sionic-mcp `app.py` - Reference implementation
- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-core.html)
- [pytest documentation](https://docs.pytest.org/)
- [asyncio.to_thread() documentation](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)
