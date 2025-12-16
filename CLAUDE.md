# Maestro Skills - Claude Code Instructions

Instructions for Claude Code to effectively use Maestro Skills in this project.

## When to Use Maestro Skills

### Automatic Triggers (Use Maestro when these keywords appear)

Use Maestro Skills when the user's request contains:
- "maestro"
- "multi-llm", "multiple LLMs", "different LLMs"
- "human-in-the-loop", "HITL", "approval", "approve each step"
- "5-stage workflow", "workflow stages"

### Recommended Usage Scenarios

| Scenario | Tool to Use |
|----------|-------------|
| Complex bug debugging | `maestro_workflow_with_hitl` → `maestro_run_stage_with_approval` |
| Need opinion from other LLMs | `maestro_consult` or `maestro_ensemble_generate` |
| Careful code changes needed | `maestro_run_stage_with_approval` (approve each stage) |
| Hypothesis verification needed | `maestro_run_stage` (hypothesize stage) |
| Large-scale multi-file changes | HITL workflow with stage-by-stage approval |

## Maestro Skills Usage Rules

### 1. HITL (Human-in-the-Loop) Workflow
For complex tasks, always use the HITL workflow:
```
maestro_workflow_with_hitl(task="task description")
↓
maestro_run_stage_with_approval(stage="analyze", task="...")
↓ [Wait for user approval]
maestro_run_stage_with_approval(stage="hypothesize", task="...")
↓ [Wait for user approval]
...
```

### 2. Stage-by-Stage Approval Required
- After each stage (`analyze`, `hypothesize`, `implement`, `debug`, `improve`) completes
- Must receive user approval via `maestro_submit_approval` before proceeding
- If rejected, re-run with revision instructions

### 3. Provider Selection Guide
| Provider | Strength | When to Use |
|----------|----------|-------------|
| `codex` | Code generation | Implementation, patch generation |
| `gemini` | Long context | Analysis, processing long files |
| `claude` | Reasoning | Review, debugging, synthesis |

### 4. Stage-Specific Usage

**Analyze (Analysis)**
```python
maestro_run_stage_with_approval(
    stage="analyze",
    task="bug description",
    context_files=["relevant_file.py"],
    context_errors=["error logs"]
)
```

**Hypothesize (Hypothesis)**
```python
maestro_run_stage_with_approval(
    stage="hypothesize",
    task="bug description",
    context_facts=["facts discovered from analysis"]
)
```

**Implement (Implementation)**
```python
maestro_run_stage_with_approval(
    stage="implement",
    task="changes to make",
    providers=["codex"]  # codex recommended for code generation
)
```

**Debug (Debugging)**
```python
# Use single agent only (paper: MAS degrades 39-70%)
maestro_run_stage_with_approval(
    stage="debug",
    task="fix test failures",
    context_errors=["failed test logs"]
)
```

**Improve (Improvement)**
```python
maestro_run_stage_with_approval(
    stage="improve",
    task="code improvements",
    providers=["gemini", "claude"]  # Multiple LLMs OK for review
)
```

## Slash Commands

The following commands are available:
- `/maestro-debug` - Debug with HITL workflow
- `/maestro-analyze` - Run analysis only
- `/maestro-consult` - Ask another LLM
- `/maestro-workflow` - Run full 5-stage workflow

## Important Notes

1. **No MAS in Debug Stage**: According to the paper, multi-agent systems degrade performance by 39-70% in sequential debugging
2. **No Proceeding Without Approval**: Must use `maestro_submit_approval` before proceeding to next stage
3. **Provide Context**: Actively use `context_files`, `context_facts`, `context_errors`
4. **Monitor Metrics**: Check progress with `maestro_workflow_state`

## Example Conversations

```
User: "Use maestro to debug the login bug"

Claude Code:
1. Call maestro_workflow_with_hitl(task="Debug login bug")
2. Run maestro_run_stage_with_approval(stage="analyze", ...)
3. Display analysis report and request approval
4. Once approved, proceed to next stage...
```

```
User: "I want to get codex's opinion on this code"

Claude Code:
1. Call maestro_consult(prompt="...", provider="codex")
2. Return the result
```
