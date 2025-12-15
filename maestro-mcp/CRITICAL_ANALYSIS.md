# Critical Analysis: sionic-mcp vs maestro-mcp (maestro-mcp)

## Executive Summary

This document provides a rigorous comparative analysis of two implementations of multi-LLM orchestration systems based on:
1. **"Towards a Science of Scaling Agent Systems"** (Kim et al., 2025)
2. **"Solving a Million-Step LLM Task With Zero Errors"** (MAKER, 2025)

Both implementations target the same goal: **centralized consult architecture** where Claude Code orchestrates and external LLM CLIs (Codex/Gemini/Claude) act as consultants.

---

## Architecture Comparison

### sionic-mcp
```
src/sionic_mcp/
├── __init__.py
├── server.py          (entrypoint)
└── app.py             (2985 lines - MONOLITHIC)

External configs:
├── skills/*.md        (stage docs)
├── schemas/*.json     (JSON schemas)
├── roles/*.md         (persona prompts)
├── checklists/*.md    (completion criteria)
└── policies/*.md      (collaboration rules)
```

### maestro-mcp (maestro-mcp)
```
maestro-mcp/
├── server.py          (tool registration)
└── maestro/
    ├── __init__.py
    ├── config.py      (MaestroConfig)
    ├── providers.py   (CodexProvider, GeminiProvider, ClaudeProvider)
    ├── context.py     (ContextPacker)
    ├── workflow.py    (WorkflowEngine)
    ├── selection.py   (SelectionEngine)
    ├── tracing.py     (TraceStore, Metrics)
    ├── verify.py      (VerificationEngine)
    ├── workspace.py   (WorkspaceManager)
    ├── consensus.py   (ConsensusEngine)
    ├── maker.py       (MAKER implementation)
    ├── skills.py      (Dynamic tool loading)
    └── coordination.py (Architecture Selection)

External configs:
└── conf/skill_manifest.yaml
```

---

## Where sionic-mcp is Better

### 1. Pragmatic Monolithic Design ✅

sionic-mcp's single `app.py` is actually **more production-ready**:

```python
# sionic-mcp: Everything in one place
def _call_provider(*, provider, model, prompt, ...):
    if provider == "codex":
        cmd = [cfg.codex_cmd, "exec", cfg.codex_model_flag, model, ...]
    elif provider == "gemini":
        cmd = [cfg.gemini_cmd, ...]
    # Clean, readable, debuggable
```

vs maestro-mcp:
```python
# Spread across multiple classes
class CodexProvider(BaseProvider):
    async def consult(self, prompt, ...):
        # Abstraction for abstraction's sake
```

**Verdict**: sionic-mcp's approach is easier to understand, deploy, and debug.

### 2. Better CLI Flag Handling ✅

sionic-mcp handles per-provider CLI flags comprehensively:

```python
codex_model_flag: str               # --model
codex_output_schema_flag: str | None # --output-schema
gemini_prompt_flag: str | None      # -p
gemini_model_flag: str | None       # --model
gemini_output_format_flag: str | None # --output-format
claude_prompt_flag: str | None      # -p
claude_model_flag: str | None       # --models
claude_output_format_flag: str | None # --output-format
claude_json_schema_flag: str | None  # --json-schema
```

maestro-mcp lacks this granularity.

### 3. Secret Masking ✅

sionic-mcp has prompt sanitization:

```python
mask_regexes: list[re.Pattern[str]]  # MASK_REGEXES env var
mask_replacement: str                 # ***MASKED***

def _mask_text(cfg, text):
    for pattern in cfg.mask_regexes:
        out = pattern.sub(cfg.mask_replacement, out)
```

**maestro-mcp is missing this entirely** — a security oversight.

### 4. JSON Schema Validation ✅

sionic-mcp uses proper JSON Schema validation:

```python
from jsonschema import Draft202012Validator

def _schema_validate(schema, value):
    Draft202012Validator(schema).validate(value)
```

maestro-mcp has basic field checking but no proper JSON Schema validation.

### 5. Complete Stage Schemas ✅

sionic-mcp has complete JSON schemas for all 5 stages hardcoded:

```python
_STAGE_SCHEMAS = {
    "analyze": {
        "type": "object",
        "required": ["observations", "repro_steps", "affected_modules", "invariants"],
        # ...
    },
    "hypothesize": {...},
    "implement": {...},
    "debug": {...},
    "improve": {...},
}
```

These are well-thought-out and match the paper's concepts.

### 6. Git State Helper ✅

sionic-mcp has comprehensive git state tracking:

```python
def _git_state(cwd):
    state["root"] = _run_git(["rev-parse", "--show-toplevel"])
    state["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    state["commit"] = _run_git(["rev-parse", "HEAD"])
    state["status"] = _run_git(["status", "--porcelain"])
    state["diff_stat"] = _run_git(["diff", "--stat"])
```

maestro-mcp lacks this — important for debugging/audit.

### 7. Artifact System ✅

sionic-mcp saves run artifacts:

```python
artifact_dir: Path  # .maestroloop/artifacts
artifact_enabled: bool

# Saves stage results, policy decisions, recommendations
```

maestro-mcp only has trace logging, no structured artifacts.

### 8. Tool Profile Implementation ✅

sionic-mcp's tool profiles are cleaner:

```python
# Explicit profile definitions
profiles = {
    "bootstrap": {"skills.get", "clink.run", "maestro.stage.run", ...},
    "analysis": {"repo.read_file", "candidates.generate", ...},
    "implementation": {"repo.apply_patch", "verify.run", ...},
}
```

### 9. Test Suite ✅

sionic-mcp has `tests/test_app_helpers.py` with actual tests.

maestro-mcp has **no tests** — unacceptable for production code.

---

## Where maestro-mcp (maestro-mcp) is Better

### 1. Faithful MAKER Implementation ✅

maestro-mcp has explicit MAKER paper concepts:

```python
class MicroStepType(Enum):
    # Analyze stage (S)
    S1_SPEC_EXTRACT = "s1_spec_extract"
    S2_EDGE_CASE = "s2_edge_case"
    S3_MRE = "s3_mre"
    # Hypothesize stage (H)
    H1_ROOT_CAUSE = "h1_root_cause"
    H2_VERIFICATION = "h2_verification"
    # ... 10 total micro-steps

class MicroStepSpec:
    step_type: MicroStepType
    output_schema: Dict[str, Any]
    default_k: int  # Voting margin
    has_oracle: bool  # Tool-based verification available?
    red_flag_rules: List[str]
    required_tools: List[str]
```

sionic-mcp only has 5 stages, not the finer MAD (Maximal Agentic Decomposition).

### 2. Calibration System ✅

maestro-mcp implements p/k calibration from MAKER:

```python
class Calibrator:
    def estimate_step_accuracy(self, sample_fn, oracle_fn, ...):
        """Estimate p (accuracy) and v (valid rate)."""

    def calculate_k(self, p, total_steps, target_success_rate=0.99):
        """Calculate required k for target overall success rate."""
        # Binary search for optimal k
        for k in range(1, 20):
            ratio = (1 - p) / p
            step_accuracy = 1 - (ratio ** k)
            overall = step_accuracy ** total_steps
            if overall >= target_success_rate:
                return k
```

sionic-mcp **hardcodes k** without calibration.

### 3. Comprehensive RedFlagger ✅

maestro-mcp has richer red-flag validation:

```python
class RedFlagger:
    hedging_patterns = [
        r"i'm not sure", r"maybe", r"perhaps", r"it might be"
    ]
    dangerous_code_patterns = [
        r"eval\s*\(", r"exec\s*\(", r"os\.system"
    ]
    dangerous_command_patterns = [
        r"rm\s+-rf", r"curl.*\|\s*bash"
    ]
    forbidden_file_patterns = [
        r"\.env$", r"secrets?\.json", r"\.pem$"
    ]
```

sionic-mcp's red-flagging is simpler (truncation, regex match/missing).

### 4. Architecture Selection Engine ✅

maestro-mcp has a complete implementation of Rules A-D:

```python
class ArchitectureSelectionEngine:
    SEQUENTIAL_THRESHOLD = 0.7  # Rule B
    DECOMPOSABLE_THRESHOLD = 0.6  # Rule B
    TOOL_COMPLEXITY_HIGH = 0.6

    def select_architecture(self, features, stage, force_topology):
        # Rule D: Check calibration data
        # Rule B: Sequential → SAS
        # Rule B: Decomposable + tool complexity
        # Rule A: Stage-specific defaults
        # Rule C: Overhead as first-class cost

    def should_degrade(self, current_topology, metrics, decision):
        # Automatic degradation when MAS isn't working
```

sionic-mcp's `policy_recommend_topology` is simpler (single function).

### 5. Metrics Tracking ✅

maestro-mcp has rolling metrics:

```python
class MetricsTracker:
    def get_topology_stats(self, topology) -> Dict[str, float]:
        return {
            "success_rate": 0.7,
            "avg_overhead": 1.5,
            "avg_tokens": 2000,
            "count": 50,
        }
```

### 6. Degradation Strategy ✅

maestro-mcp has automatic fallback:

```python
class DegradationStrategy:
    def should_degrade(self) -> Tuple[bool, str]:
        if self._format_error_count >= self.max_format_errors:
            return True, "Degrading to SAS with simplified prompts"

    def get_degraded_config(self, current_topology):
        return {
            "topology": CoordinationTopology.SAS,
            "simplify_prompts": True,
            "reduce_context": True,
            "single_model": True,
        }
```

sionic-mcp lacks this.

### 7. VoteStep with Equivalence ✅

maestro-mcp's voting supports custom equivalence:

```python
class VoteStep:
    def __init__(self,
                 equivalence_fn: Optional[Callable[[str, str], bool]],
                 normalize_fn: Optional[Callable[[str], str]]):
        # Custom equivalence for semantic matching
```

sionic-mcp only does exact hash matching.

---

## Issues in sionic-mcp

### 1. Missing Calibration ❌
No way to empirically estimate p and auto-set k. Hardcoded k=3.

### 2. No MicroStep Granularity ❌
Only 5 stages, not 10+ micro-steps for finer voting.

### 3. Missing Degradation ❌
No automatic fallback when MAS coordination fails.

### 4. No Rolling Metrics ❌
Can't learn from historical runs.

### 5. Monolithic = Harder to Test ❌
2985 lines in one file makes unit testing difficult.

---

## Issues in maestro-mcp (Self-Critique)

### 1. Over-Engineered ❌
Too many abstractions:
- `CodexProvider`, `GeminiProvider`, `ClaudeProvider` could be one function
- `ContextPacker`, `WorkflowEngine`, `SelectionEngine` add complexity
- 7+ separate modules vs 1 file that works

**Fix**: Consolidate into 2-3 files max.

### 2. Missing Secret Masking ❌
No `MASK_REGEXES` equivalent — **security risk**.

**Fix**: Add prompt sanitization immediately.

### 3. No JSON Schema Validation ❌
Basic field checking != proper jsonschema validation.

**Fix**: Add `jsonschema` dependency.

### 4. Missing Git State ❌
No `_git_state()` for repo context.

**Fix**: Add git helper.

### 5. No Artifact System ❌
Only trace logging, no structured run artifacts.

**Fix**: Add artifact directory support.

### 6. No Tests ❌
Zero test files.

**Fix**: Add test suite.

### 7. Role Prompts Incomplete ❌
sionic-mcp has 6 well-crafted persona prompts.

**Fix**: Match or exceed sionic-mcp's role system.

### 8. YAML Config Less Intuitive ❌
`conf/skill_manifest.yaml` is harder to understand than sionic-mcp's inline configs.

**Fix**: Consider inline defaults with YAML override.

---

## Recommended Synthesis

The ideal implementation would combine:

| Feature | Source | Priority |
|---------|--------|----------|
| Monolithic simplicity | sionic-mcp | HIGH |
| CLI flag handling | sionic-mcp | HIGH |
| Secret masking | sionic-mcp | HIGH |
| JSON Schema validation | sionic-mcp | HIGH |
| Complete stage schemas | sionic-mcp | MEDIUM |
| Git state helper | sionic-mcp | MEDIUM |
| Artifact system | sionic-mcp | MEDIUM |
| Tests | sionic-mcp | HIGH |
| MicroStep granularity | maestro-mcp | MEDIUM |
| Calibration system | maestro-mcp | MEDIUM |
| Comprehensive RedFlagger | maestro-mcp | MEDIUM |
| Architecture Selection Engine | maestro-mcp | MEDIUM |
| Metrics Tracking | maestro-mcp | LOW |
| Degradation Strategy | maestro-mcp | LOW |
| Equivalence in voting | maestro-mcp | LOW |

---

## Conclusion

**sionic-mcp is more production-ready** despite having less theoretical fidelity to the papers.

**maestro-mcp (maestro-mcp) is more academically faithful** but over-engineered.

### Recommended Action:
1. Keep maestro-mcp's MAKER/coordination concepts
2. Adopt sionic-mcp's pragmatic patterns (single file, CLI handling, masking)
3. Add missing critical features (tests, secret masking, JSON schema)
4. Simplify abstractions

---

## Version History
- 2025-01-15: Initial critical analysis
