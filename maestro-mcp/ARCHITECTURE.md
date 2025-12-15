# Maestro MCP - Architecture Document

## Overview

Maestro MCP is a Model Context Protocol server implementing **measured multi-LLM coordination** for coding workflows. Like a conductor orchestrating an orchestra, it coordinates Claude, Codex, and Gemini CLIs through a 5-stage problem-solving pipeline with paper-backed collaboration rules.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Claude Code (Orchestrator)                           │
│                 File editing, test running, tool execution                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ MCP Protocol
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Maestro MCP Server                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Architecture Selection Engine                      │  │
│  │  Rule A: Domain-dependent | Rule B: Decompose→MAS, Sequential→SAS   │  │
│  │  Rule C: Overhead as cost | Rule D: Calibration required             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │   Codex     │  │   Gemini    │  │   Claude    │  CLI Providers         │
│  │   (Code)    │  │  (Context)  │  │  (Review)   │  (Text only)           │
│  └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  5-Stage Workflow Engine (FSM)                                        │  │
│  │  Analyze → Hypothesize → Implement → Debug → Improve                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  MAKER Error Correction                                               │  │
│  │  MAD (Micro-steps) | First-to-ahead-by-k | Red-flagging               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Selection Engine (Poetiq-style)                                      │  │
│  │  Tests First > Lint > LLM Judge > Voting                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Philosophy

### Goals

1. **Measured Coordination**: Use multi-LLM strategically, not universally
2. **Paper-Backed Decisions**: Every design choice traces to research findings
3. **Context Efficiency**: Minimize token overhead through dynamic tool loading
4. **Graceful Degradation**: Auto-fallback when MAS underperforms SAS

### Non-Goals

- "More agents = better" is explicitly rejected
- Long-term memory/personalization (optional, off by default)
- Fully autonomous operation (human-in-the-loop for critical decisions)

---

## Foundational Papers

### 1. "Towards a Science of Scaling Agent Systems" (Kim et al., 2025)

**Key Findings:**
- Multi-agent systems (MAS) can **degrade** performance by 39-70% in sequential tasks
- Error amplification: Independent agents amplify errors 17.2x without verification
- Coordination overhead is a measurable, first-class cost
- Architecture choice must depend on task structure, not fixed patterns

**Our Implementation:** Architecture Selection Engine (Rules A-D)

### 2. "Solving a Million-Step LLM Task With Zero Errors" (MAKER, 2025)

**Key Findings:**
- Step-level voting dramatically reduces compound errors
- Red-flagging (format errors → reasoning errors) improves accuracy
- Maximal decomposition enables efficient error correction
- System-level correction scales better than model intelligence

**Our Implementation:** MAKER module (vote_step, red_flagger, calibrator)

---

## System Components

### 1. CLI Providers (`zen/providers.py`)

Adapters for external LLM CLIs. Each provider:
- Wraps a CLI command (codex, gemini, claude)
- Handles timeout, output truncation, error recovery
- Returns structured `ProviderResponse`

```python
class ProviderResponse:
    ok: bool
    stdout: str
    stderr: str
    provider: str
    model: str
    elapsed_ms: float
    truncated: bool
```

**Provider Roles:**
| Provider | Strength | Primary Use |
|----------|----------|-------------|
| Codex | Code generation | Implementation, patches |
| Gemini | Large context | Analysis, long files |
| Claude | Reasoning | Review, synthesis, debugging |

### 2. Architecture Selection Engine (`zen/coordination.py`)

Implements the 4 collaboration rules from the Scaling Agent Systems paper.

#### Rule A: Domain/Task Structure Dependency

```python
class TaskStructureClassifier:
    """Analyzes task to extract structure features."""

    def classify(self, task_description, code_context, error_logs):
        return TaskStructureFeatures(
            decomposability_score=...,      # 0-1: parallelizable?
            sequential_dependency_score=..., # 0-1: state accumulation?
            tool_complexity=...,             # 0-1: tool usage
        )
```

#### Rule B: Decomposable → MAS, Sequential → SAS

```python
def select_architecture(self, features, stage):
    # High sequential dependency → SAS
    if features.sequential_dependency_score > 0.7:
        return CoordinationDecision(topology=SAS)

    # Decomposable + tool-heavy → Centralized
    if features.decomposability_score > 0.6:
        if features.tool_complexity > 0.6:
            return CoordinationDecision(topology=MAS_CENTRALIZED)
        else:
            return CoordinationDecision(topology=MAS_INDEPENDENT)
```

#### Rule C: Coordination Overhead as Cost

```python
class CoordinationMetrics:
    total_messages: int
    total_rounds: int
    redundancy_rate: float        # Similarity of outputs
    error_amplification: float    # MAS vs SAS error ratio
    coordination_overhead: float  # Extra work vs SAS
```

Automatic degradation when overhead exceeds threshold:

```python
def should_degrade(self, metrics, decision):
    if metrics.error_amplification > 1.2:
        return True, SAS, "MAS making more errors than SAS"
    if metrics.redundancy_rate > 0.85:
        return True, SAS, "Agents producing identical outputs"
```

#### Rule D: Calibration Required

```python
class MetricsTracker:
    """Tracks per-topology statistics for calibration."""

    def get_best_topology_for_features(self, features):
        # Based on historical success rates
        return best_topology if enough_samples else None
```

### 3. MAKER Error Correction (`zen/maker.py`)

#### Maximal Agentic Decomposition (MAD)

Tasks are broken into atomic micro-steps:

```
Stage        | Micro-steps
-------------|------------------------------------------
Analyze      | s1_spec_extract, s2_edge_case, s3_mre
Hypothesize  | h1_root_cause, h2_verification
Implement    | c1_minimal_patch, c2_compile_check
Debug        | d1_failure_label, d2_next_experiment
Improve      | r1_refactor, r2_perf
```

Each micro-step has:
- Output schema (JSON)
- Default voting k
- Oracle availability (test-based verification)
- Red-flag rules

#### First-to-ahead-by-k Voting

```python
class VoteStep:
    """MAKER-style voting for error correction."""

    def vote(self, sample_fn, step_type, oracle_fn=None):
        while round < max_rounds:
            content, provider, _ = sample_fn()

            # Red-flag check (DISCARD, don't repair)
            if red_flagger.validate(content).is_flagged:
                continue

            # Oracle check (if available)
            if oracle_fn and not oracle_fn(content):
                continue

            # Count votes
            normalized = normalize(content)
            vote_counts[normalized] += 1

            # First-to-ahead-by-k: winner needs k-vote margin
            if margin >= k:
                return winner
```

#### Red-flagging

Format errors signal reasoning errors. **Discard, don't repair.**

```python
class RedFlagger:
    def validate(self, content, step_type):
        reasons = []

        if len(content) > max_chars:
            reasons.append(TOO_LONG)
        if hedging_pattern.search(content):
            reasons.append(HEDGING)
        if not valid_json(content):
            reasons.append(INVALID_JSON)
        if forbidden_file_pattern.search(content):
            reasons.append(FORBIDDEN_FILE)

        return RedFlagResult(is_flagged=len(reasons) > 0)
```

### 4. Selection Engine (`zen/selection.py`)

Poetiq-style candidate selection with priority:

```
1. Test Results (pytest, npm test)     ← HIGHEST PRIORITY
2. Static Analysis (lint, type-check)
3. LLM Judge (when tests inconclusive)
4. Voting (last resort)
```

```python
class SelectionEngine:
    def select(self, candidates, mode, test_signals, lint_signals):
        # Filter red-flagged candidates FIRST
        valid = [c for c in candidates if not c.red_flagged]

        if mode == TESTS_FIRST:
            # Tests are the judge for code
            passed = [c for c in valid if tests_pass(c)]
            if len(passed) == 1:
                return passed[0]
            # Multiple passed → use lint scores
            return max(passed, key=lambda c: lint_score(c))
```

### 5. Dynamic Skill Loading (`zen/skills.py`)

Minimizes context overhead by loading only needed tools.

```
Entry Point              │ Tools Loaded
─────────────────────────┼────────────────────────────────────
zen_enter_stage(analyze) │ zen_pack_context, zen_log_evidence,
                         │ zen_consult
                         │
zen_enter_stage(debug)   │ zen_verify, zen_restore_from_backup,
                         │ zen_pack_context
                         │
zen_exit_stage()         │ Core only: zen_list_providers,
                         │ zen_get_skill, zen_workflow_state
```

Configuration:
```bash
ZEN_MAX_TOOLS=10                    # Maximum exposed tools
ZEN_DISABLED_TOOLS=zen_run_stage    # Always disable these
```

### 6. Verification Engine (`zen/verify.py`)

Executes tests, lint, type-checks in sandbox:

```python
class VerificationEngine:
    ALLOWED_COMMANDS = ["pytest", "npm test", "ruff", "mypy", ...]

    def run(self, command, type, cwd):
        if not is_allowed(command):
            return BLOCKED

        result = subprocess.run(command, timeout=timeout)
        return VerificationResult(
            passed=(result.returncode == 0),
            exit_code=result.returncode,
            stdout=truncate(result.stdout),
        )
```

### 7. Workspace Manager (`zen/workspace.py`)

Safe file operations with automatic backup:

```python
class WorkspaceManager:
    def apply_patch(self, patch, dry_run=False):
        # 1. Validate paths (no escape from workspace)
        # 2. Create backup
        # 3. Parse unified diff
        # 4. Apply changes
        # 5. Return backup_session for rollback
```

---

## 5-Stage Workflow

### Stage 1: Analyze

**Goal:** Freeze facts before guessing

**Inputs:** Code files, error logs, test failures
**Outputs:** Observations, Constraints, Failure Signature, Task Structure

**Coordination:** `mas_independent` (parallel info gathering)
**Voting:** Score-based (not test-based)

```python
# Key SKILL outputs
{
    "observations": [...],      # Facts with evidence
    "constraints": [...],       # Input/output/invariants
    "failure_signature": {...}, # Reproduction steps
    "decomposability_score": 0.7,
    "sequential_dependency_score": 0.3
}
```

### Stage 2: Hypothesize

**Goal:** Generate testable hypotheses with experiments

**Inputs:** Observations from Stage 1
**Outputs:** Ranked hypotheses with verification plans

**Coordination:** `mas_independent` (parallel hypothesis generation)
**Selection:** Falsifiability > Low experiment cost > Explanation coverage

```python
# Hypothesis scoring criteria
{
    "falsifiable": True,           # Can be tested
    "experiment_cost": "low",      # Quick to verify
    "observations_explained": 3,   # Coverage
    "risk_level": "medium"
}
```

### Stage 3: Implement

**Goal:** Minimal, testable code changes

**Inputs:** Selected hypothesis, codebase
**Outputs:** Patch candidates with test results

**Coordination:** `mas_centralized` (tool-heavy, needs supervision)
**Selection:** TEST-FIRST (tests > lint > voting)

```python
# Patch selection priority
1. test_pass      # Must pass all tests
2. lint_score     # Higher is better
3. diff_size      # Smaller is better
4. risk_score     # Lower is better
```

### Stage 4: Debug

**Goal:** Fix without divergence

**Inputs:** Failing tests, implementation
**Outputs:** Fixed code or escalation

**Coordination:** `sas` (SINGLE AGENT ONLY)
**Max Iterations:** 5

⚠️ **WARNING:** Paper shows MAS degrades 39-70% in sequential debugging tasks.

```python
# Debug loop structure
for iteration in range(5):
    failure = triage_failures()
    fix = generate_fix_candidate()  # Single agent
    result = run_tests(fix)

    if result.passed:
        return SUCCESS

    if is_flaky(result):
        retry_with_isolation()

    if no_progress():
        escalate_to_human()
```

### Stage 5: Improve

**Goal:** Extract reusable patterns

**Inputs:** Successful fix trace
**Outputs:** Skill templates, policy updates

**Coordination:** `mas_independent` (parallel review OK)

```python
# Recursive improvement outputs
{
    "new_skill": {...},        # Reusable prompt/checker
    "policy_update": {...},    # Architecture calibration
    "playbook_entry": {...}    # Workflow improvement
}
```

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           User Request                                    │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     1. Task Classification                                │
│  zen_classify_task() → TaskStructureFeatures                             │
│  (decomposability, sequential_dependency, tool_complexity)               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   2. Architecture Selection                               │
│  zen_select_architecture() → CoordinationTopology                        │
│  (sas | mas_independent | mas_centralized)                               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    3. Dynamic Tool Loading                                │
│  zen_enter_stage() → Load stage-specific tools                           │
│  (minimize context overhead)                                              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    4. Stage Execution                                     │
│                                                                           │
│  ┌─────────────┐                                                         │
│  │   Analyze   │──► Observations, Task Structure                         │
│  └──────┬──────┘                                                         │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │ Hypothesize │──► Ranked Hypotheses                                    │
│  └──────┬──────┘     (parallel generation if MAS)                        │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │  Implement  │──► Patch Candidates                                     │
│  └──────┬──────┘     (parallel gen → test selection)                     │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │    Debug    │──► Fixed Code (SAS ONLY)                                │
│  └──────┬──────┘     (sequential, max 5 iterations)                      │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │   Improve   │──► Skill Templates, Policy Updates                      │
│  └─────────────┘                                                         │
│                                                                           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   5. Degradation Check                                    │
│  zen_check_degradation() → Fallback if needed                            │
│  (overhead, error_amplification, redundancy)                             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   6. Calibration Recording                                │
│  zen_record_coordination_result() → Update topology stats                │
│  (for future architecture selection)                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## MCP Tools Summary

### Core (4)
| Tool | Purpose |
|------|---------|
| `zen_consult` | Single model consultation |
| `zen_consult_with_role` | Consultation with persona |
| `zen_ensemble_generate` | Multi-model candidates |
| `zen_select_best` | Pick best candidate |

### Workflow (6)
| Tool | Purpose |
|------|---------|
| `zen_run_stage` | Execute workflow stage |
| `zen_workflow_state` | Check progress |
| `zen_get_skill` | Get skill definition |
| `zen_get_role` | Get persona prompt |
| `zen_get_schema` | Get output schema |
| `zen_get_coordination_policy` | Get paper-aligned rules |

### Verification (2)
| Tool | Purpose |
|------|---------|
| `zen_verify` | Run tests/lint/type-check |
| `zen_validate_content` | Red-flag validation |

### Workspace (2)
| Tool | Purpose |
|------|---------|
| `zen_apply_patch` | Apply unified diff safely |
| `zen_restore_from_backup` | Rollback changes |

### Consensus (1)
| Tool | Purpose |
|------|---------|
| `zen_consensus_vote` | First-to-ahead-by-k voting |

### Evidence (4)
| Tool | Purpose |
|------|---------|
| `zen_log_evidence` | Log to reasoning chain |
| `zen_get_evidence_chain` | Query evidence |
| `zen_get_metrics` | Paper-aligned metrics |
| `zen_pack_context` | Smart context packing |

### Dynamic Loading (5)
| Tool | Purpose |
|------|---------|
| `zen_enter_stage` | Enter stage + load tools |
| `zen_enter_skill` | Enter skill + load tools |
| `zen_exit_stage` | Exit stage + unload |
| `zen_get_loaded_tools` | Show loaded tools |
| `zen_recommend_tools` | Recommend tools for task |

### MAKER (4)
| Tool | Purpose |
|------|---------|
| `zen_get_micro_steps` | Get atomic step definitions |
| `zen_vote_micro_step` | Vote on micro-step |
| `zen_calibrate` | Calibrate voting k |
| `zen_red_flag_check` | Check for format errors |

### Coordination (6)
| Tool | Purpose |
|------|---------|
| `zen_classify_task` | Analyze task structure |
| `zen_select_architecture` | Choose topology |
| `zen_check_degradation` | Check for fallback |
| `zen_record_coordination_result` | Record for calibration |
| `zen_get_coordination_stats` | View statistics |
| `zen_get_stage_strategy` | Get stage strategy |

**Total: 34 tools**

---

## Configuration

### Environment Variables

```bash
# Provider configuration
ZEN_CODEX_CMD=codex
ZEN_CODEX_MODEL=gpt-5.2-xhigh
ZEN_CODEX_TIMEOUT=900

ZEN_GEMINI_CMD=gemini
ZEN_GEMINI_MODEL=gemini-3-pro-preview
ZEN_GEMINI_TIMEOUT=600

ZEN_CLAUDE_CMD=claude
ZEN_CLAUDE_MODEL=opus
ZEN_CLAUDE_TIMEOUT=600

# Coordination policy
ZEN_CAPABILITY_THRESHOLD=0.45    # Skip ensemble above this
ZEN_MAX_CONSULT_PER_STAGE=2
ZEN_MAX_CONSULT_TOTAL=6

# Dynamic tool loading
ZEN_MAX_TOOLS=10                 # Maximum tools at once
ZEN_DISABLED_TOOLS=zen_run_stage # Comma-separated

# Context packing
ZEN_CONTEXT_MAX_FILES=7
ZEN_CONTEXT_MAX_CHARS=40000

# Tracing
ZEN_TRACE_DIR=.zen-traces
ZEN_LOG_LEVEL=INFO
```

### Skill Manifest (`conf/skill_manifest.yaml`)

```yaml
# Tool definitions with context costs
tools:
  - name: zen_consult
    context_cost: 200
    category: consultation

# Skill definitions with required tools
skills:
  - name: root_cause_analysis
    stage: hypothesize
    required_tools:
      - zen_consult
      - zen_log_evidence
    micro_steps:
      - h1_root_cause

# Coordination policies
coordination:
  capability_threshold: 0.45
  stage_rules:
    debug:
      single_agent_only: true
      max_iterations: 5
```

---

## Safety & Operations

### File Operations
- Path validation (no workspace escape)
- Automatic backup before modification
- Allowlist-based file patterns
- Dry-run mode for patches

### Command Execution
- Allowlist of safe commands
- Timeout enforcement
- Output truncation
- No shell injection

### Degradation Triggers
- Format errors > 3 consecutive → Simplify prompts
- Overhead violations > 2 → Switch to SAS
- Success rate < 30% with high overhead → Degrade

### Escalation Points
- Debug loop > 5 iterations → Human review
- All candidates red-flagged → Human input
- Security-sensitive files → Explicit approval

---

## References

1. [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296) (Kim et al., 2025)
2. [Solving a Million-Step LLM Task With Zero Errors](https://arxiv.org/abs/2511.09030) (MAKER, 2025)
3. [Poetiq ARC Solver](https://github.com/poetiq-ai/poetiq-arc-agi-solver) - Ensemble patterns
4. [PAL MCP Server](https://github.com/BeehiveInnovations/pal-mcp-server) - CLI bridging
5. [Model Context Protocol](https://modelcontextprotocol.io/)
