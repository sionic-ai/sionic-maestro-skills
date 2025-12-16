# Maestro Skills - Architecture Document

## Overview

Maestro Skills is a Claude Code Skill system implementing **measured multi-LLM coordination** for coding workflows. Like a conductor orchestrating an orchestra, it coordinates Claude, Codex, and Gemini CLIs through a 5-stage problem-solving pipeline with paper-backed collaboration rules and **Human-in-the-Loop (HITL) approval gates**.

> **Delivery Mechanism**: While delivered via MCP (Model Context Protocol), this is conceptually a **Skills system** - providing structured workflows, approval gates, and multi-LLM coordination as reusable capabilities for Claude Code.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Claude Code (Orchestrator)                           │
│                 File editing, test running, tool execution                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ MCP Protocol
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Maestro Skills Server                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Architecture Selection Engine                      │  │
│  │  Rule A: Domain-dependent | Rule B: Decompose→MAS, Sequential→SAS   │  │
│  │  Rule C: Overhead as cost | Rule D: Calibration required             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                 Human-in-the-Loop (HITL) System                       │  │
│  │  Stage Reports | Review Questions | Approval Gates | Feedback Loop   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │   Codex     │  │   Gemini    │  │   Claude    │  CLI Providers         │
│  │   (Code)    │  │  (Context)  │  │  (Review)   │  (Text only)           │
│  └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  5-Stage Workflow Engine (FSM) + Approval Gates                       │  │
│  │  Analyze ──► Hypothesize ──► Implement ──► Debug ──► Improve          │  │
│  │      ↓           ↓              ↓           ↓          ↓              │  │
│  │   [HITL]      [HITL]         [HITL]      [HITL]     [HITL]            │  │
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
- Fully autonomous operation without oversight

### Human-in-the-Loop Principle

Based on: **"매 stage마다 사용자의 의견을 매번 자세하게 꼼꼼히 물어보기"**
(Ask for detailed user feedback at every stage)

Every workflow stage requires explicit human approval before proceeding:
- Prevents automated mistakes from propagating
- Ensures human oversight at critical decision points
- Collects feedback for continuous improvement

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

### 1. CLI Providers (`maestro/providers.py`)

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

### 2. Architecture Selection Engine (`maestro/coordination.py`)

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

### 3. MAKER Error Correction (`maestro/maker.py`)

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

### 4. Selection Engine (`maestro/selection.py`)

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

### 5. Dynamic Skill Loading (`maestro/skills.py`)

Minimizes context overhead by loading only needed tools.

```
Entry Point              │ Tools Loaded
─────────────────────────┼────────────────────────────────────
maestro_enter_stage(analyze) │ maestro_pack_context, maestro_log_evidence,
                         │ maestro_consult
                         │
maestro_enter_stage(debug)   │ maestro_verify, maestro_restore_from_backup,
                         │ maestro_pack_context
                         │
maestro_exit_stage()         │ Core only: maestro_list_providers,
                         │ maestro_get_skill, maestro_workflow_state
```

Configuration:
```bash
MAESTRO_MAX_TOOLS=10                    # Maximum exposed tools
MAESTRO_DISABLED_TOOLS=maestro_run_stage    # Always disable these
```

### 6. Verification Engine (`maestro/verify.py`)

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

### 7. Workspace Manager (`maestro/workspace.py`)

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

### 8. Human-in-the-Loop System (`maestro/human_loop.py`)

Approval gate system requiring human review at each stage.

#### Core Components

```python
class HumanLoopManager:
    """Thread-safe approval manager with memory-safe history."""

    pending_requests: Dict[str, ApprovalRequest]  # Awaiting approval
    completed_requests: Deque[ApprovalRequest]    # maxlen=100 (no memory leak)
    _lock: threading.Lock                          # Thread safety

    def request_approval(stage, outputs, duration_ms) -> ApprovalRequest
    def submit_approval(request_id, approved, feedback) -> Dict
    def get_pending_requests() -> List[Dict]
    def get_approval_history() -> List[Dict]
```

#### Stage Reports

Each stage completion generates a comprehensive bilingual report:

```python
@dataclass
class StageReport:
    stage: str
    stage_display_name: str
    stage_display_name_ko: str      # Korean translation
    summary: str
    summary_ko: str
    outputs: Dict[str, Any]
    key_findings: List[str]
    key_findings_ko: List[str]
    risks: List[str]
    risks_ko: List[str]
    questions: List[ReviewQuestion]  # Priority-based
    next_stage: Optional[str]
    next_stage_preview: str
    next_stage_preview_ko: str
```

#### Review Questions

Stage-specific questions with priority levels:

```python
class ReviewPriority(Enum):
    CRITICAL = "critical"  # 🔴 Must review before proceeding
    HIGH = "high"          # 🟠 Strongly recommended
    MEDIUM = "medium"      # 🟡 Review if time permits
    LOW = "low"            # 🟢 Optional

# Questions per stage
STAGE_QUESTIONS = {
    "analyze": [
        {"id": "analyze_completeness", "priority": CRITICAL, ...},
        {"id": "analyze_accuracy", "priority": CRITICAL, ...},
        # ... 5 questions total
    ],
    "hypothesize": [...],  # 5 questions
    "implement": [...],    # 6 questions (most critical stage)
    "debug": [...],        # 5 questions
    "improve": [...],      # 5 questions
}
```

#### Approval Flow

```
Stage Execution
      │
      ▼
┌─────────────────┐
│ Generate Report │ ← Outputs, findings, risks, questions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Request Approval│ ← maestro_request_approval()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Human Review   │ ← Review report, answer questions
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────┐
│Approve│  │  Reject  │
└───┬───┘  └────┬─────┘
    │           │
    ▼           ▼
 Next Stage   Stop/Revise
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
│  maestro_classify_task() → TaskStructureFeatures                          │
│  (decomposability, sequential_dependency, tool_complexity)               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   2. Architecture Selection                               │
│  maestro_select_architecture() → CoordinationTopology                     │
│  (sas | mas_independent | mas_centralized)                               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    3. Dynamic Tool Loading                                │
│  maestro_enter_stage() → Load stage-specific tools                        │
│  (minimize context overhead)                                              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              4. Stage Execution with HITL Approval Gates                  │
│                                                                           │
│  ┌─────────────┐     ┌──────────┐                                        │
│  │   Analyze   │──►  │  [HITL]  │──► Approval Required                   │
│  └─────────────┘     └────┬─────┘                                        │
│                           │ ✓ Approved                                    │
│                           ▼                                               │
│  ┌─────────────┐     ┌──────────┐                                        │
│  │ Hypothesize │──►  │  [HITL]  │──► Approval Required                   │
│  └─────────────┘     └────┬─────┘                                        │
│                           │ ✓ Approved                                    │
│                           ▼                                               │
│  ┌─────────────┐     ┌──────────┐                                        │
│  │  Implement  │──►  │  [HITL]  │──► Approval Required                   │
│  └─────────────┘     └────┬─────┘                                        │
│                           │ ✓ Approved                                    │
│                           ▼                                               │
│  ┌─────────────┐     ┌──────────┐                                        │
│  │    Debug    │──►  │  [HITL]  │──► Approval Required (SAS ONLY)        │
│  └─────────────┘     └────┬─────┘                                        │
│                           │ ✓ Approved                                    │
│                           ▼                                               │
│  ┌─────────────┐     ┌──────────┐                                        │
│  │   Improve   │──►  │  [HITL]  │──► Final Approval                      │
│  └─────────────┘     └──────────┘                                        │
│                                                                           │
│  HITL Components:                                                         │
│  • Stage Report (bilingual EN/KO)                                         │
│  • Review Questions (Critical/High/Medium/Low priority)                   │
│  • Approval/Rejection with feedback                                       │
│  • Revision instructions if rejected                                      │
│                                                                           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   5. Degradation Check                                    │
│  maestro_check_degradation() → Fallback if needed                         │
│  (overhead, error_amplification, redundancy)                             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   6. Calibration Recording                                │
│  maestro_record_coordination_result() → Update topology stats             │
│  (for future architecture selection)                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## MCP Tools Summary

### Core (4)
| Tool | Purpose |
|------|---------|
| `maestro_consult` | Single model consultation |
| `maestro_consult_with_role` | Consultation with persona |
| `maestro_ensemble_generate` | Multi-model candidates |
| `maestro_select_best` | Pick best candidate |

### Workflow (6)
| Tool | Purpose |
|------|---------|
| `maestro_run_stage` | Execute workflow stage |
| `maestro_workflow_state` | Check progress |
| `maestro_get_skill` | Get skill definition |
| `maestro_get_role` | Get persona prompt |
| `maestro_get_schema` | Get output schema |
| `maestro_get_coordination_policy` | Get paper-aligned rules |

### Verification (2)
| Tool | Purpose |
|------|---------|
| `maestro_verify` | Run tests/lint/type-check |
| `maestro_validate_content` | Red-flag validation |

### Workspace (2)
| Tool | Purpose |
|------|---------|
| `maestro_apply_patch` | Apply unified diff safely |
| `maestro_restore_from_backup` | Rollback changes |

### Consensus (1)
| Tool | Purpose |
|------|---------|
| `maestro_consensus_vote` | First-to-ahead-by-k voting |

### Evidence (4)
| Tool | Purpose |
|------|---------|
| `maestro_log_evidence` | Log to reasoning chain |
| `maestro_get_evidence_chain` | Query evidence |
| `maestro_get_metrics` | Paper-aligned metrics |
| `maestro_pack_context` | Smart context packing |

### Dynamic Loading (5)
| Tool | Purpose |
|------|---------|
| `maestro_enter_stage` | Enter stage + load tools |
| `maestro_enter_skill` | Enter skill + load tools |
| `maestro_exit_stage` | Exit stage + unload |
| `maestro_get_loaded_tools` | Show loaded tools |
| `maestro_recommend_tools` | Recommend tools for task |

### MAKER (4)
| Tool | Purpose |
|------|---------|
| `maestro_get_micro_steps` | Get atomic step definitions |
| `maestro_vote_micro_step` | Vote on micro-step |
| `maestro_calibrate` | Calibrate voting k |
| `maestro_red_flag_check` | Check for format errors |

### Coordination (6)
| Tool | Purpose |
|------|---------|
| `maestro_classify_task` | Analyze task structure |
| `maestro_select_architecture` | Choose topology |
| `maestro_check_degradation` | Check for fallback |
| `maestro_record_coordination_result` | Record for calibration |
| `maestro_get_coordination_stats` | View statistics |
| `maestro_get_stage_strategy` | Get stage strategy |

### Human-in-the-Loop (7)
| Tool | Purpose |
|------|---------|
| `maestro_workflow_with_hitl` | Start HITL workflow |
| `maestro_run_stage_with_approval` | Run stage + request approval |
| `maestro_request_approval` | Request human approval |
| `maestro_submit_approval` | Submit approval decision |
| `maestro_get_pending_approvals` | View pending approvals |
| `maestro_get_approval_history` | Review past decisions |
| `maestro_get_stage_questions` | Preview review questions |

**Total: 41 tools**

---

## Configuration

### Environment Variables

```bash
# Provider configuration
MAESTRO_CODEX_CMD=codex
MAESTRO_CODEX_MODEL=gpt-5.1-codex-max
MAESTRO_CODEX_TIMEOUT=900

MAESTRO_GEMINI_CMD=gemini
MAESTRO_GEMINI_MODEL=gemini-3-pro-preview
MAESTRO_GEMINI_TIMEOUT=600

MAESTRO_CLAUDE_CMD=claude
MAESTRO_CLAUDE_MODEL=opus
MAESTRO_CLAUDE_TIMEOUT=600

# Coordination policy
MAESTRO_CAPABILITY_THRESHOLD=0.45    # Skip ensemble above this
MAESTRO_MAX_CONSULT_PER_STAGE=2
MAESTRO_MAX_CONSULT_TOTAL=6

# Dynamic tool loading
MAESTRO_MAX_TOOLS=10                 # Maximum tools at once
MAESTRO_DISABLED_TOOLS=maestro_run_stage # Comma-separated

# Context packing
MAESTRO_CONTEXT_MAX_FILES=7
MAESTRO_CONTEXT_MAX_CHARS=40000

# Tracing
MAESTRO_TRACE_DIR=.maestro-traces
MAESTRO_LOG_LEVEL=INFO
```

### Skill Manifest (`conf/skill_manifest.yaml`)

```yaml
# Tool definitions with context costs
tools:
  - name: maestro_consult
    context_cost: 200
    category: consultation

# Skill definitions with required tools
skills:
  - name: root_cause_analysis
    stage: hypothesize
    required_tools:
      - maestro_consult
      - maestro_log_evidence
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

### Human-in-the-Loop Gates
- Every stage completion → Approval required before proceeding
- Rejection → Workflow stops or requests revision
- Feedback collected → Incorporated into next iteration
- History tracked → For audit and calibration

---

## References

1. [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296) (Kim et al., 2025)
2. [Solving a Million-Step LLM Task With Zero Errors](https://arxiv.org/abs/2511.09030) (MAKER, 2025)
3. [Poetiq ARC Solver](https://github.com/poetiq-ai/poetiq-arc-agi-solver) - Ensemble patterns
4. [PAL MCP Server](https://github.com/BeehiveInnovations/pal-mcp-server) - CLI bridging
5. [Model Context Protocol](https://modelcontextprotocol.io/)
