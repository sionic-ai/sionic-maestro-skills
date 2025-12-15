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

#### 1. Add to `zen/config.py`

```python
import re
from typing import List, Pattern

@dataclass
class ZenConfig:
    # ... existing fields ...

    # Secret masking
    mask_regexes: List[Pattern] = field(default_factory=list)
    mask_replacement: str = "***MASKED***"

    @classmethod
    def from_env(cls) -> "ZenConfig":
        # ... existing code ...

        # Parse MASK_REGEXES from env (comma-separated patterns)
        mask_patterns_str = os.getenv("ZEN_MASK_REGEXES", "")
        mask_regexes = []
        if mask_patterns_str:
            for pattern in mask_patterns_str.split(","):
                pattern = pattern.strip()
                if pattern:
                    try:
                        mask_regexes.append(re.compile(pattern, re.IGNORECASE))
                    except re.error:
                        logging.warning(f"Invalid mask regex: {pattern}")

        # Add default patterns
        default_patterns = [
            r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w-]+['\"]?",
            r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?",
            r"(?i)(secret|token)\s*[:=]\s*['\"]?[\w-]+['\"]?",
            r"(?i)(bearer)\s+[\w-]+",
            r"(?i)(authorization)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?",
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub tokens
            r"gho_[a-zA-Z0-9]{36}",  # GitHub OAuth tokens
            r"glpat-[a-zA-Z0-9-]{20}",  # GitLab tokens
        ]
        for pattern in default_patterns:
            mask_regexes.append(re.compile(pattern))

        return cls(
            # ... existing fields ...
            mask_regexes=mask_regexes,
            mask_replacement=os.getenv("ZEN_MASK_REPLACEMENT", "***MASKED***"),
        )
```

#### 2. Add masking utility to `zen/providers.py`

```python
def mask_secrets(text: str, config: ZenConfig) -> str:
    """Mask sensitive data in text before sending to external CLIs."""
    result = text
    for pattern in config.mask_regexes:
        result = pattern.sub(config.mask_replacement, result)
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
ZEN_MASK_REGEXES="my-secret-\d+,internal-token-[a-z]+"

# Custom replacement text
ZEN_MASK_REPLACEMENT="[REDACTED]"
```

### Testing

```python
def test_mask_secrets():
    config = ZenConfig(mask_regexes=[re.compile(r"sk-\w+")])
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
jsonschema>=4.0.0
```

#### 2. Add validation utility to `zen/selection.py`

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

#### 3. Update `RedFlagger` in `zen/maker.py`

```python
from jsonschema import Draft202012Validator, ValidationError

class RedFlagger:
    def _check_required_fields(self, content: str, schema: Dict) -> List[str]:
        """Check if JSON content validates against schema."""
        try:
            # Extract JSON from content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return ["No JSON object found"]

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
# In zen/workflow.py
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
│   ├── conftest.py              # Fixtures
│   ├── test_config.py           # ZenConfig tests
│   ├── test_providers.py        # Provider tests (mocked CLIs)
│   ├── test_selection.py        # Selection engine tests
│   ├── test_maker.py            # MAKER module tests
│   │   ├── test_red_flagger.py
│   │   ├── test_vote_step.py
│   │   └── test_calibrator.py
│   ├── test_coordination.py     # Architecture selection tests
│   ├── test_skills.py           # Dynamic tool loading tests
│   ├── test_workspace.py        # Patch application tests
│   └── test_integration.py      # End-to-end tests
├── pytest.ini
└── requirements-dev.txt
```

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
addopts = -v --cov=zen --cov-report=term-missing
```

#### 3. `tests/conftest.py`

```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from zen.config import ZenConfig
from zen.providers import ProviderRegistry


@pytest.fixture
def config():
    """Default test configuration."""
    return ZenConfig(
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
from zen.maker import (
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
        from zen.maker import StageType

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
from zen.coordination import (
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
pytest --cov=zen --cov-report=html

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

#### Add to `zen/workspace.py`

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

    # Branch
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        state["branch"] = branch

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
def zen_git_state(
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
    from zen.workspace import git_state
    return git_state(Path(path) if path else None)
```

#### Use in context packing

```python
# In zen/context.py
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

#### 1. Add to `zen/config.py`

```python
@dataclass
class ZenConfig:
    # ... existing fields ...

    # Artifacts
    artifact_dir: Path = Path(".maestro-artifacts")
    artifact_enabled: bool = True
```

#### 2. Create `zen/artifacts.py`

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
    """

    def __init__(self, base_dir: Path, enabled: bool = True):
        self.base_dir = base_dir
        self.enabled = enabled
        self.current_run_id: Optional[str] = None
        self._run_dir: Optional[Path] = None

    def start_run(self, task: str, git_state: Optional[Dict] = None) -> str:
        """Start a new run and return run_id."""
        if not self.enabled:
            return ""

        self.current_run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._run_dir = self.base_dir / self.current_run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata = ArtifactMetadata(
            run_id=self.current_run_id,
            created_at=datetime.now().isoformat(),
            task=task,
            git_commit=git_state.get("short_commit") if git_state else None,
            git_branch=git_state.get("branch") if git_state else None,
        )
        self._write_json("metadata.json", asdict(metadata))

        return self.current_run_id

    def save_stage_result(self, stage: str, result: Dict[str, Any]):
        """Save stage output."""
        if not self.enabled or not self._run_dir:
            return
        self._write_json(f"stage_{stage}.json", result)

    def save_policy_decision(self, decision: Dict[str, Any]):
        """Append policy decision to JSONL."""
        if not self.enabled or not self._run_dir:
            return
        self._append_jsonl("policy_decisions.jsonl", decision)

    def save_coordination_metrics(self, metrics: Dict[str, Any]):
        """Save coordination metrics."""
        if not self.enabled or not self._run_dir:
            return
        self._write_json("coordination_metrics.json", metrics)

    def save_recommendation(self, recommendation: Dict[str, Any]):
        """Save final recommendation."""
        if not self.enabled or not self._run_dir:
            return
        self._write_json("final_recommendation.json", recommendation)

    def end_run(self) -> Optional[Path]:
        """End current run and return artifact directory."""
        if not self.enabled or not self._run_dir:
            return None

        run_dir = self._run_dir
        self.current_run_id = None
        self._run_dir = None
        return run_dir

    def _write_json(self, filename: str, data: Dict):
        path = self._run_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _append_jsonl(self, filename: str, data: Dict):
        path = self._run_dir / filename
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

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
@mcp.tool()
def zen_start_artifact_run(task: str) -> Dict[str, str]:
    """Start a new artifact run for the given task."""
    from zen.workspace import git_state
    run_id = artifact_store.start_run(task, git_state())
    return {"run_id": run_id, "enabled": artifact_store.enabled}

@mcp.tool()
def zen_list_artifact_runs(limit: int = 10) -> List[Dict]:
    """List recent artifact runs."""
    return artifact_store.list_runs(limit)

@mcp.tool()
def zen_load_artifact_run(run_id: str) -> Dict[str, Any]:
    """Load all artifacts from a previous run."""
    return artifact_store.load_run(run_id)
```

### Environment Variables

```bash
ZEN_ARTIFACT_DIR=.maestro-artifacts
ZEN_ARTIFACT_ENABLED=true
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
└── zen/
    ├── __init__.py
    ├── core.py            # Config + Providers + Context (merged)
    ├── workflow.py        # Workflow + Selection + Verification (merged)
    ├── maker.py           # MAKER module (unchanged - well-structured)
    └── coordination.py    # Coordination (unchanged - well-structured)
```

### Consolidation Plan

#### Phase 1: Merge `config.py` + `providers.py` → `core.py`

```python
# zen/core.py

# --- Config section ---
@dataclass
class ZenConfig:
    # All config fields
    ...

# --- Provider section ---
def call_provider(
    provider: str,
    prompt: str,
    model: Optional[str] = None,
    config: Optional[ZenConfig] = None,
    timeout: Optional[int] = None,
) -> ProviderResponse:
    """Universal provider caller (replaces 3 provider classes)."""
    config = config or ZenConfig.from_env()

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
# zen/workflow.py

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

### Risk Mitigation

- All tests must pass after each consolidation step
- Keep git history clean with atomic commits
- Document breaking changes in CHANGELOG.md

---

## Implementation Timeline

| Week | Tasks |
|------|-------|
| 1 | P0-1 (Secret Masking), P0-2 (JSON Schema) |
| 2 | P0-3 (Test Suite - core tests) |
| 3 | P0-3 (Test Suite - remaining), P1-1 (Git State) |
| 4 | P1-2 (Artifact System) |
| 5-6 | P2-1 (Module Consolidation) |

---

## Success Criteria

### P0 Complete
- [ ] All prompts sanitized before external CLI calls
- [ ] All stage outputs validated against JSON Schema
- [ ] Test coverage > 70%
- [ ] All tests pass in CI

### P1 Complete
- [ ] `zen_git_state` tool available and working
- [ ] Artifacts saved for each run
- [ ] Can replay/audit previous runs

### P2 Complete
- [ ] Module count reduced from 13 to 4-5
- [ ] No functionality lost
- [ ] All tests still pass
- [ ] Documentation updated

---

## References

- `CRITICAL_ANALYSIS.md` - Comparison with sionic-mcp
- sionic-mcp `app.py` - Reference implementation
- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-core.html)
- [pytest documentation](https://docs.pytest.org/)
