# Maestro MCP - Installation & Usage Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [Usage Examples](#usage-examples)
6. [Workflow Patterns](#workflow-patterns)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **Python 3.10+**
- **Claude Code** (CLI) - [Installation Guide](https://docs.anthropic.com/claude-code)
- At least one of the following CLI tools:
  - `claude` CLI (Anthropic)
  - `codex` CLI (OpenAI)
  - `gemini` CLI (Google)

### Verify CLI Tools

```bash
# Check Claude CLI
claude --version

# Check Codex CLI (if using)
codex --version

# Check Gemini CLI (if using)
gemini --version
```

---

## Installation

### Step 1: Clone or Copy

```bash
# If part of a repo
cd your-project
cp -r path/to/maestro-mcp ./maestro-mcp

# Or clone directly
git clone <repo-url>
cd maestro-mcp
```

### Step 2: Install Dependencies

```bash
cd maestro-mcp
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your settings
nano .env  # or your preferred editor
```

### Step 4: Register with Claude Code

Create or update `.mcp.json` in your **project root** (not inside maestro-mcp):

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

### Step 5: Verify Installation

```bash
# Start Claude Code
claude

# Check MCP servers
/mcp

# Should show:
# ✓ maestro-mcp (running)
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# =============================================================================
# CLI Providers
# =============================================================================

# Codex (OpenAI)
MAESTRO_CODEX_CMD=codex
MAESTRO_CODEX_MODEL=gpt-5.2-xhigh
MAESTRO_CODEX_TIMEOUT=900

# Gemini (Google)
MAESTRO_GEMINI_CMD=gemini
MAESTRO_GEMINI_MODEL=gemini-3-pro-preview
MAESTRO_GEMINI_TIMEOUT=600

# Claude (Anthropic) - as consultant, not orchestrator
MAESTRO_CLAUDE_CMD=claude
MAESTRO_CLAUDE_MODEL=opus
MAESTRO_CLAUDE_TIMEOUT=600

# =============================================================================
# Coordination Policy
# =============================================================================

# Skip ensemble if baseline confidence > this (paper: ~45%)
MAESTRO_CAPABILITY_THRESHOLD=0.45

# Budget limits
MAESTRO_MAX_CONSULT_PER_STAGE=2
MAESTRO_MAX_CONSULT_TOTAL=6

# =============================================================================
# Context Management
# =============================================================================

# Maximum tools to expose at once (saves context)
MAESTRO_MAX_TOOLS=10

# Tools to always disable (comma-separated)
MAESTRO_DISABLED_TOOLS=

# Context packing limits
MAESTRO_CONTEXT_MAX_FILES=7
MAESTRO_CONTEXT_MAX_CHARS=40000

# =============================================================================
# Tracing & Logging
# =============================================================================

MAESTRO_TRACE_DIR=.maestro-traces
MAESTRO_LOG_LEVEL=INFO
```

### Minimal Configuration

If you only have Claude CLI:

```bash
MAESTRO_CODEX_CMD=
MAESTRO_GEMINI_CMD=
MAESTRO_CLAUDE_CMD=claude
MAESTRO_CLAUDE_MODEL=sonnet
```

---

## Quick Start

### 1. Start Claude Code

```bash
claude
```

### 2. Check Available Tools

```
You: What maestro tools are available?

Claude: Let me check the available tools.
[Uses maestro_list_providers]

Available providers: claude, codex, gemini
Available tools: maestro_consult, maestro_verify, maestro_apply_patch, ...
```

### 3. Basic Consultation

```
You: Use maestro to analyze this error: "TypeError: Cannot read property 'map' of undefined"

Claude: I'll consult the LLM ensemble to analyze this error.
[Uses maestro_consult with the error context]
```

### 4. Run a Full Workflow

```
You: Debug the failing test in tests/auth.test.js using the maestro workflow

Claude: I'll follow the 5-stage workflow to debug this.
[Uses maestro_enter_stage, maestro_classify_task, etc.]
```

---

## Usage Examples

### Example 1: Simple Consultation

Ask a single model for advice:

```
You: Use maestro_consult to ask codex how to optimize this function:
[paste function]

Claude: [Uses maestro_consult]
Result: Codex suggests using memoization...
```

### Example 2: Ensemble Generation

Generate multiple solutions and pick the best:

```
You: Generate 3 different approaches to implement user authentication

Claude: I'll use ensemble generation to get diverse solutions.
[Uses maestro_ensemble_generate with providers=["codex", "gemini", "claude"]]

Got 3 candidates:
1. JWT-based (Codex)
2. Session-based (Gemini)
3. OAuth2 (Claude)

[Uses maestro_select_best with test_results]
Selected: JWT-based approach (passes all tests, smallest diff)
```

### Example 3: Architecture Selection

Let the system decide SAS vs MAS:

```
You: I need to fix a complex bug that spans multiple files.
Should I use multi-agent or single-agent approach?

Claude: Let me analyze the task structure.
[Uses maestro_classify_task]

Result:
- decomposability_score: 0.3 (low - hard to parallelize)
- sequential_dependency_score: 0.8 (high - steps depend on each other)
- recommended_topology: "sas" (single agent)
- reason: "Rule B: Sequential → SAS"

I'll use single-agent approach for this debugging task.
```

### Example 4: MAKER-style Voting

Use error-corrected voting for critical decisions:

```
You: What's the root cause of this NullPointerException? Use voting to be sure.

Claude: I'll use MAKER-style voting for accuracy.
[Uses maestro_vote_micro_step with step_type="h1_root_cause", k=3]

Voting results:
- Round 1: "uninitialized user" (codex)
- Round 2: "uninitialized user" (gemini)
- Round 3: "null session" (codex)
- Round 4: "uninitialized user" (claude)

Winner: "uninitialized user object" (3 votes, margin=2)
Converged: True
Confidence: 0.75
```

### Example 5: Safe Patch Application

Apply code changes with automatic backup:

```
You: Apply this fix to src/auth.js

Claude: I'll apply the patch safely with backup.
[Uses maestro_apply_patch with dry_run=True first]

Dry run result:
- Files to modify: src/auth.js
- Lines changed: +5, -2
- No conflicts detected

[Uses maestro_apply_patch with dry_run=False]

Patch applied successfully.
Backup saved to: .maestro-backups/20250615-143022/

[Uses maestro_verify to run tests]
All tests passing.
```

### Example 6: Rollback on Failure

```
You: The tests are failing after the patch. Roll back.

Claude: I'll restore from backup.
[Uses maestro_restore_from_backup]

Restored files:
- src/auth.js

[Uses maestro_verify]
Tests passing again.
```

---

## Workflow Patterns

### Pattern 1: Full 5-Stage Debug Workflow

```
# Stage 1: Analyze
You: Debug the failing login test. Start with analysis.

Claude: [Uses maestro_enter_stage("analyze")]
        [Uses maestro_classify_task]
        [Uses maestro_pack_context with error logs]
        [Uses maestro_consult for initial analysis]
        [Uses maestro_log_evidence]

Analysis complete:
- Observations: [list]
- Task type: debugging (sequential_dependency=0.8)
- Recommended: Single-agent approach

# Stage 2: Hypothesize
You: Generate hypotheses for the root cause.

Claude: [Uses maestro_enter_stage("hypothesize")]
        [Uses maestro_ensemble_generate]  # OK for hypothesis gen
        [Uses maestro_select_best with scoring]

Top hypotheses:
1. Session timeout not handled (score: 0.85)
2. Token validation failing (score: 0.72)

# Stage 3: Implement
You: Implement the fix for hypothesis 1.

Claude: [Uses maestro_enter_stage("implement")]
        [Uses maestro_consult for patch generation]  # Single agent for impl
        [Uses maestro_apply_patch with dry_run=True]
        [Uses maestro_apply_patch]
        [Uses maestro_verify]

Patch applied. Running tests...

# Stage 4: Debug (if tests fail)
You: Tests still failing. Debug it.

Claude: [Uses maestro_enter_stage("debug")]
        [Uses maestro_get_stage_strategy]  # Confirms SAS

        # Single-agent debug loop (max 5 iterations)
        Iteration 1:
        [Uses maestro_verify to get failure details]
        [Uses maestro_consult to analyze]
        [Uses maestro_apply_patch for fix]
        [Uses maestro_verify]

        Tests passing after 2 iterations.

# Stage 5: Improve
You: Extract learnings from this fix.

Claude: [Uses maestro_enter_stage("improve")]
        [Uses maestro_log_evidence with fix pattern]
        [Uses maestro_record_coordination_result]

Recorded:
- Pattern: "Session timeout handling"
- Topology used: SAS
- Success: True
- Added to calibration data
```

### Pattern 2: Quick Fix (Skip Full Workflow)

For simple, obvious fixes:

```
You: Fix this typo in the error message

Claude: This is a simple fix, no need for full workflow.
        [Uses maestro_apply_patch directly]
        [Uses maestro_verify]

Done. Typo fixed, tests passing.
```

### Pattern 3: Exploration Mode

When you're not sure what's wrong:

```
You: Something's wrong with the API but I don't know what. Explore.

Claude: I'll use multi-agent exploration.
        [Uses maestro_classify_task]
        → decomposability: 0.7 (can parallelize exploration)

        [Uses maestro_select_architecture]
        → topology: mas_independent

        [Uses maestro_ensemble_generate]
        - Codex: Checks request/response formats
        - Gemini: Analyzes error patterns
        - Claude: Reviews authentication flow

        [Uses maestro_select_best]

Found: Authentication header malformed in 3 endpoints.
```

---

## Best Practices

### 1. Match Topology to Task

| Task Type | Topology | Reason |
|-----------|----------|--------|
| Bug analysis | mas_independent | Parallel info gathering |
| Hypothesis gen | mas_independent | Diverse perspectives |
| Code implementation | mas_centralized | Tool-heavy |
| Debugging loop | **sas** | Sequential dependency |
| Code review | mas_independent | Parallel review |

### 2. Use Tests as Judges

```
# GOOD: Let tests decide
maestro_select_best(candidates, mode="tests_first", test_results=[...])

# AVOID: Pure voting for code
maestro_select_best(candidates, mode="llm_judge")  # Only when tests unavailable
```

### 3. Red-flag Early

```
# Validate before using any LLM output
result = maestro_validate_content(response, content_type="json")
if not result["is_valid"]:
    # Discard and retry, don't try to fix
    retry()
```

### 4. Monitor Coordination Overhead

```
# After each coordination
maestro_record_coordination_result(topology, success, tokens_used)

# Periodically check
stats = maestro_get_coordination_stats()
if stats["best_topology"] != current:
    # Consider switching
```

### 5. Degrade Gracefully

```
# Check during long operations
degradation = maestro_check_degradation(
    current_topology="mas_independent",
    successes=2, failures=5
)
if degradation["should_degrade"]:
    # Switch to simpler topology
    switch_to_sas()
```

### 6. Minimize Context

```
# Load only what you need
maestro_enter_stage("debug")  # Loads ~6 tools

# Unload when done
maestro_exit_stage()  # Back to 3 core tools
```

---

## Troubleshooting

### Problem: "Provider not found"

```
Error: Provider 'codex' is not enabled
```

**Solution:** Check that the CLI is installed and configured:

```bash
# Verify CLI exists
which codex

# Check .env configuration
cat .env | grep CODEX
```

### Problem: "MCP server not running"

```
/mcp shows maestro-mcp as disconnected
```

**Solution:**

1. Check Python path:
```bash
which python3
```

2. Verify server.py location in .mcp.json:
```json
{
  "mcpServers": {
    "maestro-mcp": {
      "command": "python3",
      "args": ["maestro-mcp/server.py"]  // Check this path
    }
  }
}
```

3. Test server directly:
```bash
python3 maestro-mcp/server.py
# Should start without errors
```

### Problem: "Timeout errors"

```
Error: Command timed out after 300s
```

**Solution:** Increase timeout in .env:

```bash
MAESTRO_CODEX_TIMEOUT=900
MAESTRO_GEMINI_TIMEOUT=900
MAESTRO_CLAUDE_TIMEOUT=900
```

### Problem: "Too many tools, context overflow"

**Solution:** Reduce max tools or disable unused ones:

```bash
MAESTRO_MAX_TOOLS=8
MAESTRO_DISABLED_TOOLS=maestro_run_stage,maestro_get_metrics,maestro_get_coordination_policy
```

### Problem: "Red-flagged responses"

```
All candidates were red-flagged
```

**Solution:** Check red-flag thresholds:

```python
# In maestro_validate_content call, adjust limits
maestro_validate_content(
    content=response,
    max_chars=20000,  # Increase if responses are legitimately long
)
```

### Problem: "Degradation keeps triggering"

**Solution:** Check your success rate and adjust thresholds:

```python
# View current stats
stats = maestro_get_coordination_stats()

# If MAS is consistently failing, it might be correct to use SAS
# Or adjust the degradation thresholds in coordination policy
```

### Problem: "Tests not found"

```
maestro_verify: Command not in allowlist
```

**Solution:** The command must be in the allowlist. Check `maestro/verify.py`:

```python
ALLOWED_COMMANDS = [
    "pytest", "python -m pytest",
    "npm test", "npm run test",
    # Add your test command if needed
]
```

---

## Getting Help

1. **Check logs:**
   ```bash
   cat .maestro-traces/latest.jsonl
   ```

2. **Enable debug logging:**
   ```bash
   MAESTRO_LOG_LEVEL=DEBUG
   ```

3. **View evidence chain:**
   ```
   maestro_get_evidence_chain(limit=20)
   ```

4. **Report issues:**
   https://github.com/your-repo/issues

---

## Next Steps

1. **Read ARCHITECTURE.md** - Understand the system design
2. **Try the examples** - Start with simple consultations
3. **Monitor metrics** - Use `maestro_get_coordination_stats()`
4. **Customize** - Edit `conf/skill_manifest.yaml` for your workflow
