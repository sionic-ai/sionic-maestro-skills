# Maestro Skills

A Claude Code Skill implementing **Centralized Consult** architecture for Multi-LLM coding workflows. Like a maestro conducting an orchestra, Claude Code orchestrates multiple LLM models to produce harmonious, accurate output.

> **Note**: While delivered via MCP (Model Context Protocol), this is conceptually a **Skills system** for Claude Code - providing structured workflows, human-in-the-loop approval, and multi-LLM coordination capabilities.

Based on:
- **"Towards a Science of Scaling Agent Systems"** (Kim et al., 2025)
- **"Solving a Million-Step LLM Task With Zero Errors"** (MAKER)

## Philosophy

This project implements "measured coordination" - using multiple LLMs strategically rather than assuming "more agents = better."

### Key Insights from the Papers

| Finding | Implication | Our Implementation |
|---------|-------------|-------------------|
| **Tool-Coordination Trade-off** | Tool-heavy tasks suffer from MAS overhead | Only orchestrator (Claude Code) runs tools |
| **Capability Saturation (~45%)** | High baseline → MAS returns diminish | Skip ensemble when confident |
| **Error Amplification (17.2x)** | Independent agents amplify errors | Tests-first selection, not voting |
| **Sequential Task Degradation** | Debug loops degrade with MAS | Single-agent for implement/debug |
| **Red-flagging (MAKER)** | Format errors signal reasoning errors | Reject malformed responses before selection |
| **First-to-ahead-by-k (MAKER)** | Efficient micro-decision voting | Consensus engine for small decisions |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code (Orchestrator)                   │
│  - File editing, test running, tool execution                    │
│  - Final decision maker                                          │
└─────────────────┬───────────────────────────────────────────────┘
                  │ MCP Protocol
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Maestro Skills Server                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Codex     │  │   Gemini    │  │   Claude    │  Consultants │
│  │   (Code)    │  │  (Context)  │  │  (Review)   │  (Text only) │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  5-Stage Workflow Engine (FSM)                             │ │
│  │  Analyze → Hypothesize → Implement → Debug → Improve       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Selection Engine (Poetiq-style + MAKER Red-flagging)      │ │
│  │  Tests First > Lint > LLM Judge > Voting                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Evidence Chain (JSONL logging for auditability)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Quick Install (Recommended)

```bash
cd maestro-mcp
make install
```

This single command will:
1. Create a virtual environment with Python 3.11
2. Install all dependencies using `uv`
3. **Configure globally** in `~/.claude.json`
4. Make Maestro available in **all** Claude Code sessions

After installation, restart Claude Code and verify:
```bash
# In Claude Code
/mcp
# Should show: ✓ maestro-mcp (running)
```

### Other Commands

```bash
make help        # Show all available commands
make check       # Verify installation
make verify      # Check if CLI tools are available
make run         # Start the MCP server manually
make mcp-config  # Show MCP config (for manual setup)
make clean       # Remove venv and cache
make reinstall   # Clean and reinstall
```

### Prerequisites

Before installing Maestro Skills, ensure you have:

1. **Python 3.11** - Check with `python3 --version`
2. **uv** - Fast Python package manager
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Claude Code CLI** - Anthropic's official CLI (`claude`)
4. **At least one LLM CLI** (optional but recommended):
   - OpenAI Codex CLI (`codex`)
   - Google Gemini CLI (`gemini`)

### Manual Installation

If you prefer manual setup:

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd sionic-maestro-skills/maestro-mcp

# Create virtual environment with Python 3.11
uv venv --python 3.11 .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Step 3: Install LLM CLI Tools

#### OpenAI Codex CLI (Recommended)
```bash
# Install via npm
npm install -g @openai/codex

# Or install via pip
pip install openai-codex

# Verify installation
codex --version
```

#### Google Gemini CLI (Optional)
```bash
# Install Gemini CLI
pip install google-generativeai

# Verify installation
gemini --version
```

#### Anthropic Claude CLI (Installed with Claude Code)
```bash
# Usually comes with Claude Code
claude --version
```

### Step 4: Verify Installation

> **Note**: `make install` automatically configures the global MCP server.
> For manual configuration, run `make mcp-config` to see the JSON config.

```bash
# Start Claude Code
claude

# Check MCP servers (inside Claude Code)
/mcp

# Should show:
# ✓ maestro-mcp (running)
```

Test a simple tool:
```bash
# In Claude Code conversation
"List available maestro providers"
# Claude should use maestro_list_providers and show available CLIs
```

### Environment Variables (Optional)

Create a `.env` file or export these variables:

```bash
# Provider Models
export MAESTRO_CODEX_MODEL="gpt-5.1-codex-max"  # Default Codex model
export MAESTRO_GEMINI_MODEL="gemini-3-pro-preview"
export MAESTRO_CLAUDE_MODEL="opus"

# Timeouts (seconds)
export MAESTRO_CODEX_TIMEOUT=900
export MAESTRO_GEMINI_TIMEOUT=600
export MAESTRO_CLAUDE_TIMEOUT=600

# Coordination Policy
export MAESTRO_CAPABILITY_THRESHOLD=0.45
export MAESTRO_MAX_CONSULT_PER_STAGE=2
export MAESTRO_MAX_CONSULT_TOTAL=6

# Tracing
export MAESTRO_TRACE_DIR=".maestro-traces"
export MAESTRO_LOG_LEVEL="INFO"
```

### Troubleshooting

#### "Provider not found" Error
```bash
# Check if CLI is in PATH
which codex
which gemini

# If not found, add to PATH or install
export PATH="$PATH:/path/to/cli"
```

#### "MCP server not starting" Error
```bash
# Check Python path
which python3

# Run server directly to see errors
python3 maestro-mcp/server.py

# Check for import errors
cd maestro-mcp && python -c "import server; print('OK')"
```

#### Timeout Issues
```bash
# Increase timeout in environment
export MAESTRO_CODEX_TIMEOUT=1800  # 30 minutes
```

## How to Use Maestro Skills

Maestro Skills is **not automatically invoked**. You must explicitly request it.

### Method 1: Keyword Triggers

Include these keywords in your request:

| Keyword | Example |
|---------|---------|
| `maestro` | "Use **maestro** to debug this bug" |
| `multi-llm` | "Get **multi-llm** opinions on this code" |
| `HITL` / `approval` | "Debug with **HITL** approval at each step" |
| `workflow` | "Run the full **workflow** on this task" |

### Method 2: Slash Commands

```
/maestro-debug <description>     Debug with full HITL workflow
/maestro-analyze <description>   Analyze code/issue only
/maestro-consult <question>      Ask another LLM
/maestro-workflow <description>  Run complete 5-stage workflow
```

**Examples**:
```
/maestro-debug Fix the login bug in auth.py
/maestro-analyze Review the payment processing code
/maestro-consult What's the best way to handle this error?
/maestro-workflow Implement user session management
```

### Method 3: Direct Tool Calls

```
"Call maestro_workflow_with_hitl to start debugging"
"Use maestro_consult with codex to get a second opinion"
```

## Available Tools (41 Total)

### Core Consultation Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_consult` | Single model consultation | Analysis, review, specific questions |
| `maestro_consult_with_role` | Consultation with persona | Stage-specific prompting |
| `maestro_ensemble_generate` | Multi-model candidates | Hypothesis generation, exploration |
| `maestro_select_best` | Pick best candidate | After ensemble, with test results |

### Workflow Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_run_stage` | Execute workflow stage | Structured 5-stage workflow |
| `maestro_workflow_state` | Check progress | Monitor budget, see metrics |
| `maestro_get_skill` | Get stage skill definition | Before starting a stage |
| `maestro_get_role` | Get persona prompt | Role-based consultation |
| `maestro_get_schema` | Get output schema | Validate stage outputs |
| `maestro_get_coordination_policy` | Get paper-aligned rules | Understand when to use MAS |

### Verification Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_verify` | Run tests/lint/type-check | Before accepting any change |
| `maestro_validate_content` | Red-flag validation | Validate LLM responses |

### Workspace Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_apply_patch` | Apply unified diff safely | Implementing code changes |
| `maestro_restore_from_backup` | Rollback changes | When a patch introduces bugs |

### Consensus Tools (MAKER-style)

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_consensus_vote` | First-to-ahead-by-k voting | Micro-decisions (file selection, yes/no) |

### Evidence & Metrics Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_log_evidence` | Log to reasoning chain | Track observations, decisions |
| `maestro_get_evidence_chain` | Query evidence | Audit decision trail |
| `maestro_get_metrics` | Paper-aligned metrics | Performance analysis |
| `maestro_list_providers` | Show available CLIs | Configuration check |
| `maestro_pack_context` | Smart context packing | Before any consultation |

### Dynamic Skill Loading Tools (NEW - Context Optimization)

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_enter_stage` | Enter stage + load tools | Starting a workflow stage |
| `maestro_enter_skill` | Enter skill + load tools | Fine-grained tool loading |
| `maestro_exit_stage` | Exit stage + unload tools | Transitioning between stages |
| `maestro_get_loaded_tools` | Show loaded tools | Check available tools |
| `maestro_recommend_tools` | Recommend tools for task | Before starting any work |

### MAKER-style Micro-step Tools (Error Correction)

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_get_micro_steps` | Get atomic step definitions | Planning decomposition |
| `maestro_vote_micro_step` | Vote on micro-step result | Error-corrected execution |
| `maestro_calibrate` | Calibrate voting k parameter | Before long workflows |
| `maestro_red_flag_check` | Check for format errors | Before accepting any output |

### Coordination Tools (NEW - Architecture Selection)

| Tool | Purpose | Use When |
|------|---------|----------|
| `maestro_classify_task` | Analyze task structure | Start of Stage 1 (Analyze) |
| `maestro_select_architecture` | Choose SAS vs MAS topology | Before each stage |
| `maestro_check_degradation` | Check if should fall back | During coordination |
| `maestro_record_coordination_result` | Record for calibration | After each coordination |
| `maestro_get_coordination_stats` | Get calibration data | Monitor effectiveness |
| `maestro_get_stage_strategy` | Get stage-specific strategy | Before executing stage |

## The 5-Stage Workflow

### Stage 1: Analyze
**Goal**: Freeze facts before guessing.
- Read files, gather error logs
- Document observations and constraints
- Low overhead (2 consults max)

### Stage 2: Hypothesize
**Goal**: Generate competing explanations.
- Use ensemble generation (best stage for MAS!)
- Each hypothesis needs a verification test
- Select with `maestro_select_best`

### Stage 3: Implement
**Goal**: Apply minimal, testable changes.
- Single agent PREFERRED (tool-heavy)
- Test after EVERY change
- Use `maestro_apply_patch` for safe modifications

### Stage 4: Debug
**Goal**: Fix without divergence.
- Single agent ONLY (paper shows MAS degrades here)
- 5 iteration limit
- Use `maestro_restore_from_backup` if needed

### Stage 5: Improve
**Goal**: Refactor and stabilize.
- ONLY after tests pass
- Ensemble OK for review suggestions
- Don't over-engineer

## New Features

### Dynamic Skill Loading (Context Optimization)

Minimize context overhead by loading only the tools needed for the current task:

```python
# Enter a stage - loads stage-specific tools
result = maestro_enter_stage("hypothesize")
# Returns: {"loaded_tools": ["maestro_ensemble_generate", "maestro_consensus_vote", ...]}

# Enter a specific skill - even finer-grained loading
result = maestro_enter_skill("root_cause_analysis")
# Returns: {"loaded_tools": ["maestro_consult", "maestro_log_evidence"]}

# Check what's loaded
result = maestro_get_loaded_tools()
# Returns: {"loaded_tools": [...], "context_cost": 600}

# Exit stage - unload to core tools only
result = maestro_exit_stage()
# Returns: {"loaded_tools": ["maestro_list_providers", "maestro_get_skill", "maestro_workflow_state"]}
```

Configure via environment variables:
```bash
MAESTRO_MAX_TOOLS=10           # Maximum tools to expose at once
MAESTRO_DISABLED_TOOLS=maestro_run_stage,maestro_get_metrics  # Always disable these
```

### MAKER-style Micro-step Decomposition (MAD)

Break tasks into atomic steps for error-corrected execution:

```python
# Get available micro-steps for a stage
result = maestro_get_micro_steps("hypothesize")
# Returns micro-steps: h1_root_cause, h2_verification

# Vote on a micro-step with error correction
result = maestro_vote_micro_step(
    step_type="h1_root_cause",
    prompt="What is the most likely root cause of the NullPointerException?",
    context="Error at line 42 in UserService.java...",
    k=3,  # Need 3-vote margin to win
    providers=["codex", "gemini"]
)
# Returns: {"winner": "Uninitialized user object", "converged": True, "confidence": 0.85}
```

Micro-step types by stage:
- **Analyze**: `s1_spec_extract`, `s2_edge_case`, `s3_mre`
- **Hypothesize**: `h1_root_cause`, `h2_verification`
- **Implement**: `c1_minimal_patch`, `c2_compile_check`
- **Debug**: `d1_failure_label`, `d2_next_experiment`
- **Improve**: `r1_refactor`, `r2_perf`

### Architecture Selection Engine (Coordination Rules A-D)

Automatically select optimal coordination topology based on task structure:

```python
# 1. Classify task structure at start
result = maestro_classify_task(
    task_description="Fix the authentication bug in login flow",
    error_logs="NullPointerException at AuthService.java:42..."
)
# Returns: {
#   "features": {"decomposability_score": 0.3, "sequential_dependency_score": 0.7},
#   "recommended_topology": "sas",
#   "paper_rule_applied": "Rule B: Sequential → SAS"
# }

# 2. Select architecture for each stage
result = maestro_select_architecture(
    stage="hypothesize",
    decomposability_score=0.8,
    sequential_dependency_score=0.2
)
# Returns: {"topology": "mas_independent", "max_agents": 3}

# 3. Check for degradation during execution
result = maestro_check_degradation(
    current_topology="mas_independent",
    successes=2, failures=5,
    redundancy_rate=0.9
)
# Returns: {"should_degrade": True, "reason": "High redundancy (0.9)..."}

# 4. Record results for calibration (Rule D)
maestro_record_coordination_result(
    topology="mas_independent",
    success=True,
    tokens_used=5000
)
```

**Paper Rules Implemented:**
- **Rule A**: Architecture depends on domain/task structure
- **Rule B**: Decomposable → MAS, Sequential → SAS
- **Rule C**: Coordination overhead is a first-class cost
- **Rule D**: Model family calibration is necessary

**Stage-specific Defaults:**
- **analyze/hypothesize**: `mas_independent` (parallel info gathering)
- **implement**: `mas_centralized` (tool-heavy, needs supervision)
- **debug**: `sas` (sequential dependency, MAS degrades 39-70%)
- **improve**: `mas_independent` (parallel review OK)

### Voting Calibration

Calibrate the voting parameter k before long workflows:

```python
# Calibrate k for a step type
result = maestro_calibrate(
    step_type="c1_minimal_patch",
    test_prompt="Generate a patch to fix the null check...",
    oracle_command="pytest tests/",  # Optional oracle
    target_success_rate=0.99,
    estimated_total_steps=50,
    num_samples=10
)
# Returns: {"recommended_k": 4, "estimated_p": 0.87, "insight": "With p=0.87 and k=4..."}
```

### MAKER-style Red-flagging

Reject malformed responses BEFORE selection:

```python
# Validate any content before using it
result = maestro_validate_content(
    content=response,
    content_type="json",  # or "diff", "general"
    require_json_fields=["hypothesis", "confidence"]
)
if not result["is_valid"]:
    print(f"Red-flagged: {result['reason']}")
```

Red-flag criteria:
- Too long (indicates rambling)
- Too short (incomplete)
- Hedging language ("I'm not sure")
- Invalid JSON/diff format
- Missing required fields

### First-to-ahead-by-k Voting

For micro-decisions, use voting instead of single-shot:

```python
# Which file is most likely the bug source?
result = maestro_consensus_vote(
    question="Which file is most likely the source of the authentication bug? Answer with ONLY the filename.",
    k=3,  # Need 3-vote lead to win
    providers=["codex", "gemini", "claude"]
)
print(f"Winner: {result['winner']} (confidence: {result['confidence']})")
```

Use for:
- File/module selection
- Binary yes/no decisions
- Key information extraction

DON'T use for:
- Code generation
- Complex implementation decisions

### Evidence Chain

Track reasoning for auditability:

```python
# Log an observation
obs_id = maestro_log_evidence(
    evidence_type="observation",
    stage="analyze",
    content={"error": "NullPointerException at line 42"},
    source="stack_trace.log"
)

# Log a hypothesis linked to the observation
hyp_id = maestro_log_evidence(
    evidence_type="hypothesis",
    stage="hypothesize",
    content={"claim": "Uninitialized user object"},
    source="reasoning",
    linked_evidence_ids=[obs_id]
)

# Query the chain
chain = maestro_get_evidence_chain(stage="analyze")
```

### Safe Patch Application

Apply patches with automatic backup:

```python
result = maestro_apply_patch(
    patch="""
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,7 @@
 def login(user):
+    if user is None:
+        raise ValueError("User required")
     return True
""",
    dry_run=False
)

# If something goes wrong, rollback
if tests_failed:
    maestro_restore_from_backup(result["backup_session"])
```

### Human-in-the-Loop (HITL) Approval System

**NEW**: Every workflow stage now requires explicit human approval before proceeding.

#### Why HITL?

Based on the principle: "매 stage마다 사용자의 의견을 매번 자세하게 꼼꼼히 물어보기" (Ask for detailed user feedback at every stage).

- Ensures human oversight at critical decision points
- Prevents automated mistakes from propagating
- Collects valuable feedback for continuous improvement

#### HITL Tools

| Tool | Purpose |
|------|---------|
| `maestro_workflow_with_hitl` | Start a workflow with approval gates |
| `maestro_run_stage_with_approval` | Run stage + auto-request approval |
| `maestro_request_approval` | Request approval for stage outputs |
| `maestro_submit_approval` | Submit approval decision |
| `maestro_get_pending_approvals` | View pending approvals |
| `maestro_get_approval_history` | Review past decisions |
| `maestro_get_stage_questions` | Preview review questions |

#### Usage Example

```python
# 1. Start HITL workflow
result = maestro_workflow_with_hitl(
    task="Fix authentication bug in login flow"
)

# 2. Run first stage with automatic approval request
result = maestro_run_stage_with_approval(
    stage="analyze",
    task="Fix authentication bug"
)
# Returns detailed report with questions for review

# 3. Review and submit approval
result = maestro_submit_approval(
    request_id=result["approval"]["request_id"],
    approved=True,
    feedback="Analysis looks complete",
    question_responses={
        "analyze_completeness": "Yes, all observations captured",
        "analyze_accuracy": "Yes, all accurate"
    }
)
# Returns: {"status": "approved", "next_action": "proceed_to_next_stage"}

# 4. Continue to next stage...
result = maestro_run_stage_with_approval(
    stage="hypothesize",
    task="Fix authentication bug",
    context_facts=["User object is null at line 42", ...]
)
```

#### Stage-Specific Review Questions

Each stage has carefully designed review questions:

**Analyze Stage** (5 questions):
- 🔴 Are all observations captured?
- 🔴 Are the facts accurate?
- 🟠 Are affected files identified correctly?
- 🟠 Are reproduction steps correct?
- 🟡 Any missing invariants?

**Hypothesize Stage** (5 questions):
- 🔴 Which hypothesis is most plausible?
- 🔴 Any missed potential causes?
- 🟠 Are verification methods feasible?
- 🟡 Agree with testing order?
- 🟡 Which hypothesis is riskiest if wrong?

**Implement Stage** (6 questions):
- 🔴 Is the approach correct?
- 🔴 Is the change scope minimal?
- 🔴 Any potential side effects?
- 🟠 Is verification command appropriate?
- 🟠 Any code quality concerns?
- 🟡 Auto-rollback on failure?

**Debug Stage** (5 questions):
- 🔴 Is error analysis correct?
- 🔴 Continue current hypothesis or try alternative?
- 🟠 Is proposed action appropriate?
- 🟡 How many more iterations?
- 🟠 Any debugging insights?

**Improve Stage** (5 questions):
- 🔴 Are improvements necessary?
- 🟠 Which improvements to prioritize?
- 🟠 Any improvements too risky?
- 🟡 Are regression tests sufficient?
- 🔴 Is code ready for merge?

Legend: 🔴 Critical | 🟠 High | 🟡 Medium

#### Bilingual Support

All reports and questions are provided in both English and Korean:

```
============================================================
🔍 STAGE APPROVAL REQUEST: Code Implementation
   코드 구현 (Code Implementation)
============================================================

📋 SUMMARY / 요약
----------------------------------------
EN: Implementation ready. Files to modify: src/auth.py
KO: 구현 준비 완료. 수정할 파일: src/auth.py

❓ REVIEW QUESTIONS / 검토 질문
----------------------------------------
🔴 [CRITICAL] Is this the right approach to fix the issue?
   이것이 문제를 해결하는 올바른 접근 방식인가요?
```

## Usage Examples

### Bug Investigation with Evidence Trail

```
User: "Debug the failing login test"

Claude Code will:
1. [ANALYZE]
   - maestro_pack_context(files=["tests/test_auth.py"], errors=[...])
   - maestro_log_evidence(type="observation", content={"error": "..."})

2. [HYPOTHESIZE]
   - maestro_ensemble_generate(task="Root causes for...", providers=["codex", "gemini"])
   - maestro_validate_content(content=..., content_type="json")  # Red-flag check
   - maestro_select_best(candidates=..., mode="tests_first")
   - maestro_log_evidence(type="hypothesis", ...)

3. [IMPLEMENT]
   - maestro_apply_patch(patch=..., dry_run=True)  # Validate first
   - maestro_apply_patch(patch=..., dry_run=False)  # Apply with backup

4. [VERIFY]
   - maestro_verify([{"command": "pytest tests/test_auth.py", "type": "unit_test"}])

5. [DEBUG] if tests fail
   - maestro_restore_from_backup(backup_session)  # Rollback
   - Single-agent iteration

6. [IMPROVE]
   - Add regression tests
   - maestro_log_evidence(type="decision", content={"action": "Added test coverage"})
```

### Micro-decision with Consensus

```python
# When unsure which file to investigate
result = maestro_consensus_vote(
    question="""
    Given this error: "TypeError: 'NoneType' object is not iterable"

    Which file should we investigate first?
    - src/parser.py
    - src/loader.py
    - src/processor.py

    Answer with ONLY the filename.
    """,
    k=3,
    providers=["codex", "gemini"]
)
# Result: {"winner": "src/loader.py", "confidence": 0.83}
```

## Configuration

### Environment Variables

```bash
# Provider configuration (best models)
MAESTRO_CODEX_CMD=codex
MAESTRO_CODEX_MODEL=gpt-5.2-xhigh
MAESTRO_CODEX_TIMEOUT=900

MAESTRO_GEMINI_CMD=gemini
MAESTRO_GEMINI_MODEL=gemini-3-pro-preview
MAESTRO_GEMINI_TIMEOUT=600

MAESTRO_CLAUDE_CMD=claude
MAESTRO_CLAUDE_MODEL=opus
MAESTRO_CLAUDE_TIMEOUT=600

# Coordination policy
MAESTRO_CAPABILITY_THRESHOLD=0.45    # Skip ensemble above this confidence
MAESTRO_MAX_CONSULT_PER_STAGE=2
MAESTRO_MAX_CONSULT_TOTAL=6

# Context packing
MAESTRO_CONTEXT_MAX_FILES=7
MAESTRO_CONTEXT_MAX_CHARS=40000

# Tracing
MAESTRO_TRACE_DIR=.maestro-traces
MAESTRO_LOG_LEVEL=INFO
```

### Disabling Tools

To reduce context overhead (as PAL MCP suggests):

```bash
MAESTRO_DISABLED_TOOLS=maestro_run_stage,maestro_workflow_state
```

## Paper-Aligned Metrics

The server tracks metrics from both papers:

| Metric | Definition | Target |
|--------|------------|--------|
| **Coordination Overhead (O%)** | (MAS_turns - SAS_turns) / SAS_turns × 100 | < 300% |
| **Efficiency Score (Ec)** | Success_rate / relative_overhead | > 0.4 |
| **Consults per Stage** | Total consults / stages completed | < 2 |
| **Test Coverage Rate** | Selections with test signals / total | > 80% |
| **Red-flag Rate** | Flagged responses / total responses | Track for quality |

## Project Structure

```
maestro-mcp/
├── server.py              # MCP server entry point (34 tools)
├── requirements.txt       # Dependencies
├── .env.example          # Configuration template
├── conf/
│   ├── cli_clients.yaml  # CLI command templates, roles, policies
│   └── skill_manifest.yaml  # Skill-to-tool mappings (NEW)
├── skills/               # Stage skill definitions
│   ├── stage1_example_analysis.md
│   ├── stage2_hypothesis.md
│   ├── stage3_implementation.md
│   ├── stage4_debug_loop.md
│   └── stage5_recursive_improve.md
├── roles/                # Persona prompts
│   ├── example_analyst.md
│   ├── hypothesis_scientist.md
│   ├── implementer.md
│   ├── debugger.md
│   ├── refiner.md
│   └── judge.md
├── schemas/              # Output JSON schemas
│   ├── stage1_output.json
│   ├── stage2_output.json
│   ├── stage3_output.json
│   ├── stage4_output.json
│   ├── stage5_output.json
│   └── judge_output.json
└── maestro/
    ├── __init__.py
    ├── config.py          # Configuration management
    ├── providers.py       # CLI provider implementations
    ├── context.py         # Context packing strategies
    ├── workflow.py        # 5-stage workflow engine
    ├── selection.py       # Candidate selection (tests_first + red-flagging)
    ├── consensus.py       # MAKER-style voting
    ├── verify.py          # Test/lint execution
    ├── workspace.py       # Safe file operations
    ├── tracing.py         # Metrics, evidence chain
    ├── maker.py           # MAKER core: vote_step, redflagger, calibrate
    ├── skills.py          # Dynamic skill loading + tool registry
    └── coordination.py    # Architecture Selection Engine (Rules A-D) (NEW)
```

## Contributing

1. Follow the measured coordination principles
2. Add tests for new features
3. Update SKILL.md for workflow changes
4. Document paper-aligned reasoning

## References

- [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296) (Kim et al., 2025)
- [Solving a Million-Step LLM Task With Zero Errors](https://arxiv.org/) (MAKER paper)
- [Poetiq ARC Solver](https://github.com/poetiq-ai/poetiq-arc-agi-solver) - Ensemble pattern inspiration
- [PAL MCP Server](https://github.com/BeehiveInnovations/pal-mcp-server) - CLI bridging patterns
- [Model Context Protocol](https://modelcontextprotocol.io/)

## License

MIT
