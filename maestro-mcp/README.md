# Maestro MCP

A Model Context Protocol (MCP) server implementing **Centralized Consult** architecture for Multi-LLM coding workflows. Like a maestro conducting an orchestra, Claude Code orchestrates multiple LLM models to produce harmonious, accurate output.

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
│                    Maestro MCP Server                            │
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

## Quick Start

### 1. Install Dependencies

```bash
cd maestro-mcp
pip install -r requirements.txt
```

### 2. Configure CLI Tools

Ensure these are installed and in your PATH:
- `codex` - OpenAI Codex CLI (`codex exec --model gpt-5.2-xhigh`)
- `gemini` - Google Gemini CLI (`gemini --model gemini-3-pro-preview`)
- `claude` - Anthropic Claude CLI (`claude -p --model opus`)

### 3. Register with Claude Code

The `.mcp.json` in the project root is pre-configured:

```json
{
  "mcpServers": {
    "maestro-mcp": {
      "command": "python3",
      "args": ["maestro-mcp/server.py"]
    }
  }
}
```

Run Claude Code and verify with `/mcp`:
```bash
claude
# Then type: /mcp
# Should show maestro-mcp as running
```

## Available Tools (34 Total)

### Core Consultation Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_consult` | Single model consultation | Analysis, review, specific questions |
| `zen_consult_with_role` | Consultation with persona | Stage-specific prompting |
| `zen_ensemble_generate` | Multi-model candidates | Hypothesis generation, exploration |
| `zen_select_best` | Pick best candidate | After ensemble, with test results |

### Workflow Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_run_stage` | Execute workflow stage | Structured 5-stage workflow |
| `zen_workflow_state` | Check progress | Monitor budget, see metrics |
| `zen_get_skill` | Get stage skill definition | Before starting a stage |
| `zen_get_role` | Get persona prompt | Role-based consultation |
| `zen_get_schema` | Get output schema | Validate stage outputs |
| `zen_get_coordination_policy` | Get paper-aligned rules | Understand when to use MAS |

### Verification Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_verify` | Run tests/lint/type-check | Before accepting any change |
| `zen_validate_content` | Red-flag validation | Validate LLM responses |

### Workspace Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_apply_patch` | Apply unified diff safely | Implementing code changes |
| `zen_restore_from_backup` | Rollback changes | When a patch introduces bugs |

### Consensus Tools (MAKER-style)

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_consensus_vote` | First-to-ahead-by-k voting | Micro-decisions (file selection, yes/no) |

### Evidence & Metrics Tools

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_log_evidence` | Log to reasoning chain | Track observations, decisions |
| `zen_get_evidence_chain` | Query evidence | Audit decision trail |
| `zen_get_metrics` | Paper-aligned metrics | Performance analysis |
| `zen_list_providers` | Show available CLIs | Configuration check |
| `zen_pack_context` | Smart context packing | Before any consultation |

### Dynamic Skill Loading Tools (NEW - Context Optimization)

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_enter_stage` | Enter stage + load tools | Starting a workflow stage |
| `zen_enter_skill` | Enter skill + load tools | Fine-grained tool loading |
| `zen_exit_stage` | Exit stage + unload tools | Transitioning between stages |
| `zen_get_loaded_tools` | Show loaded tools | Check available tools |
| `zen_recommend_tools` | Recommend tools for task | Before starting any work |

### MAKER-style Micro-step Tools (Error Correction)

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_get_micro_steps` | Get atomic step definitions | Planning decomposition |
| `zen_vote_micro_step` | Vote on micro-step result | Error-corrected execution |
| `zen_calibrate` | Calibrate voting k parameter | Before long workflows |
| `zen_red_flag_check` | Check for format errors | Before accepting any output |

### Coordination Tools (NEW - Architecture Selection)

| Tool | Purpose | Use When |
|------|---------|----------|
| `zen_classify_task` | Analyze task structure | Start of Stage 1 (Analyze) |
| `zen_select_architecture` | Choose SAS vs MAS topology | Before each stage |
| `zen_check_degradation` | Check if should fall back | During coordination |
| `zen_record_coordination_result` | Record for calibration | After each coordination |
| `zen_get_coordination_stats` | Get calibration data | Monitor effectiveness |
| `zen_get_stage_strategy` | Get stage-specific strategy | Before executing stage |

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
- Select with `zen_select_best`

### Stage 3: Implement
**Goal**: Apply minimal, testable changes.
- Single agent PREFERRED (tool-heavy)
- Test after EVERY change
- Use `zen_apply_patch` for safe modifications

### Stage 4: Debug
**Goal**: Fix without divergence.
- Single agent ONLY (paper shows MAS degrades here)
- 5 iteration limit
- Use `zen_restore_from_backup` if needed

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
result = zen_enter_stage("hypothesize")
# Returns: {"loaded_tools": ["zen_ensemble_generate", "zen_consensus_vote", ...]}

# Enter a specific skill - even finer-grained loading
result = zen_enter_skill("root_cause_analysis")
# Returns: {"loaded_tools": ["zen_consult", "zen_log_evidence"]}

# Check what's loaded
result = zen_get_loaded_tools()
# Returns: {"loaded_tools": [...], "context_cost": 600}

# Exit stage - unload to core tools only
result = zen_exit_stage()
# Returns: {"loaded_tools": ["zen_list_providers", "zen_get_skill", "zen_workflow_state"]}
```

Configure via environment variables:
```bash
ZEN_MAX_TOOLS=10           # Maximum tools to expose at once
ZEN_DISABLED_TOOLS=zen_run_stage,zen_get_metrics  # Always disable these
```

### MAKER-style Micro-step Decomposition (MAD)

Break tasks into atomic steps for error-corrected execution:

```python
# Get available micro-steps for a stage
result = zen_get_micro_steps("hypothesize")
# Returns micro-steps: h1_root_cause, h2_verification

# Vote on a micro-step with error correction
result = zen_vote_micro_step(
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
result = zen_classify_task(
    task_description="Fix the authentication bug in login flow",
    error_logs="NullPointerException at AuthService.java:42..."
)
# Returns: {
#   "features": {"decomposability_score": 0.3, "sequential_dependency_score": 0.7},
#   "recommended_topology": "sas",
#   "paper_rule_applied": "Rule B: Sequential → SAS"
# }

# 2. Select architecture for each stage
result = zen_select_architecture(
    stage="hypothesize",
    decomposability_score=0.8,
    sequential_dependency_score=0.2
)
# Returns: {"topology": "mas_independent", "max_agents": 3}

# 3. Check for degradation during execution
result = zen_check_degradation(
    current_topology="mas_independent",
    successes=2, failures=5,
    redundancy_rate=0.9
)
# Returns: {"should_degrade": True, "reason": "High redundancy (0.9)..."}

# 4. Record results for calibration (Rule D)
zen_record_coordination_result(
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
result = zen_calibrate(
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
result = zen_validate_content(
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
result = zen_consensus_vote(
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
obs_id = zen_log_evidence(
    evidence_type="observation",
    stage="analyze",
    content={"error": "NullPointerException at line 42"},
    source="stack_trace.log"
)

# Log a hypothesis linked to the observation
hyp_id = zen_log_evidence(
    evidence_type="hypothesis",
    stage="hypothesize",
    content={"claim": "Uninitialized user object"},
    source="reasoning",
    linked_evidence_ids=[obs_id]
)

# Query the chain
chain = zen_get_evidence_chain(stage="analyze")
```

### Safe Patch Application

Apply patches with automatic backup:

```python
result = zen_apply_patch(
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
    zen_restore_from_backup(result["backup_session"])
```

## Usage Examples

### Bug Investigation with Evidence Trail

```
User: "Debug the failing login test"

Claude Code will:
1. [ANALYZE]
   - zen_pack_context(files=["tests/test_auth.py"], errors=[...])
   - zen_log_evidence(type="observation", content={"error": "..."})

2. [HYPOTHESIZE]
   - zen_ensemble_generate(task="Root causes for...", providers=["codex", "gemini"])
   - zen_validate_content(content=..., content_type="json")  # Red-flag check
   - zen_select_best(candidates=..., mode="tests_first")
   - zen_log_evidence(type="hypothesis", ...)

3. [IMPLEMENT]
   - zen_apply_patch(patch=..., dry_run=True)  # Validate first
   - zen_apply_patch(patch=..., dry_run=False)  # Apply with backup

4. [VERIFY]
   - zen_verify([{"command": "pytest tests/test_auth.py", "type": "unit_test"}])

5. [DEBUG] if tests fail
   - zen_restore_from_backup(backup_session)  # Rollback
   - Single-agent iteration

6. [IMPROVE]
   - Add regression tests
   - zen_log_evidence(type="decision", content={"action": "Added test coverage"})
```

### Micro-decision with Consensus

```python
# When unsure which file to investigate
result = zen_consensus_vote(
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
ZEN_CAPABILITY_THRESHOLD=0.45    # Skip ensemble above this confidence
ZEN_MAX_CONSULT_PER_STAGE=2
ZEN_MAX_CONSULT_TOTAL=6

# Context packing
ZEN_CONTEXT_MAX_FILES=7
ZEN_CONTEXT_MAX_CHARS=40000

# Tracing
ZEN_TRACE_DIR=.zen-traces
ZEN_LOG_LEVEL=INFO
```

### Disabling Tools

To reduce context overhead (as PAL MCP suggests):

```bash
ZEN_DISABLED_TOOLS=zen_run_stage,zen_workflow_state
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
└── zen/
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
