---
name: stage3_implementation
stage_id: implement
goal: "Apply minimal, testable changes - one change at a time"
persona: implementer
default_topology: SAS
max_consults: 1
ensemble_allowed: false
---

# Stage 3: Code Implementation

## Purpose

Apply the **smallest possible change** that tests the selected hypothesis. Every edit must be followed by verification.

> "Simplicity is the ultimate sophistication." — Leonardo da Vinci

## When to Use

- After a hypothesis is selected in Stage 2
- When making any code change
- When applying a fix or feature

## Why Single Agent (Paper-Aligned)

**Tool-Coordination Trade-off**: Implementation is tool-heavy:
- File edits
- Test runs
- Build commands
- Lint checks

The paper shows tool-heavy tasks **suffer from multi-agent overhead** (β = -0.330, p < 0.001).

Therefore: **SAS (Single Agent System) is optimal for implementation.**

## Input Requirements

```yaml
required:
  - selected_hypothesis: object  # From Stage 2
  - target_files: string[]       # Files to modify
  - test_command: string         # How to verify

optional:
  - constraints: string[]        # From Stage 1
  - style_guide: string          # Coding standards
  - existing_tests: string[]     # Tests to not break
```

## Process

### Step 1: Plan the Minimal Change
- Identify the **exact** lines to change
- Write the change as a **diff** before applying
- Verify the change addresses the hypothesis

### Step 2: Apply Single Change
- Edit ONE file at a time
- Make ONE logical change
- No "while I'm here" improvements

### Step 3: Run Verification Immediately
```bash
# Run the specific test
pytest tests/test_auth.py::test_login -v

# Check for regressions
pytest tests/ -x --tb=short

# Run linter
ruff check src/
```

### Step 4: Evaluate Result
- ✅ **Pass**: Proceed to Stage 5 (Improve)
- ❌ **Fail with same error**: Hypothesis might be wrong → back to Stage 2
- ❌ **Fail with new error**: → Proceed to Stage 4 (Debug)

## Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["changes", "test_result", "next_action"],
  "properties": {
    "changes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["file", "diff"],
        "properties": {
          "file": {
            "type": "string",
            "description": "Path to modified file"
          },
          "diff": {
            "type": "string",
            "description": "Unified diff of changes"
          },
          "description": {
            "type": "string",
            "description": "What this change does"
          },
          "lines_changed": {
            "type": "integer"
          }
        }
      }
    },
    "test_result": {
      "type": "object",
      "properties": {
        "command": { "type": "string" },
        "exit_code": { "type": "integer" },
        "passed": { "type": "boolean" },
        "output": { "type": "string" },
        "duration_ms": { "type": "integer" }
      }
    },
    "lint_result": {
      "type": "object",
      "properties": {
        "passed": { "type": "boolean" },
        "warnings": { "type": "integer" },
        "errors": { "type": "integer" }
      }
    },
    "next_action": {
      "type": "string",
      "enum": ["proceed_to_improve", "proceed_to_debug", "revert_and_retry", "escalate"],
      "description": "What to do next based on results"
    },
    "rollback_command": {
      "type": "string",
      "description": "How to undo this change if needed"
    }
  }
}
```

## Example Output

```json
{
  "changes": [
    {
      "file": "src/auth.py",
      "diff": "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -40,7 +40,9 @@ def get_active_user(users):\n-    return users[0]\n+    if not users:\n+        return None\n+    return users[0]",
      "description": "Add empty list guard before accessing first element",
      "lines_changed": 3
    }
  ],
  "test_result": {
    "command": "pytest tests/test_auth.py::test_login -v",
    "exit_code": 0,
    "passed": true,
    "output": "tests/test_auth.py::test_login PASSED",
    "duration_ms": 1234
  },
  "lint_result": {
    "passed": true,
    "warnings": 0,
    "errors": 0
  },
  "next_action": "proceed_to_improve",
  "rollback_command": "git checkout src/auth.py"
}
```

## Coordination Policy

| Scenario | Action |
|----------|--------|
| Simple fix, clear hypothesis | SAS - no external consult |
| Need diff suggestion | Consult Codex once for code |
| Test passes | → Stage 5 (Improve) |
| Test fails (same error) | → Stage 2 (re-hypothesize) |
| Test fails (new error) | → Stage 4 (Debug) |
| Max iterations reached | Escalate to human |

## Implementation Rules

### DO ✅
```python
# GOOD: Single, focused change
if not users:
    return None
return users[0]
```

### DON'T ❌
```python
# BAD: Multiple unrelated changes in one edit
if not users:  # Fix for bug
    return None
# Also refactoring while here
return users[0] if users else DEFAULT_USER  # Different approach
# And adding logging
logger.info(f"Found {len(users)} users")  # Unrelated
```

## Tool Usage

```yaml
allowed_tools:
  - repo.read_file      # Read code
  - repo.apply_patch    # Apply diff
  - repo.run_tests      # Verify
  - repo.run_lint       # Check style
  - repo.git_diff       # Review changes
  - repo.git_status     # Check state

forbidden_tools:
  - candidates.generate # No ensemble in implement
  - cli.call with multiple providers  # SAS only
```

## Anti-patterns to Avoid

1. **Large changes** - If diff > 20 lines, split it up
2. **Multiple concerns** - One change per edit
3. **Skipping tests** - ALWAYS test after change
4. **Ensemble generation** - Paper shows this hurts tool-heavy stages
5. **"Fixing" unrelated code** - Stay focused

## Exit Criteria

- [ ] Change applied successfully
- [ ] Test command executed
- [ ] Result evaluated
- [ ] Next action determined
- [ ] Rollback command documented

## Next Stage

- **Pass**: → Stage 5 (Recursive Improvement)
- **Fail (new error)**: → Stage 4 (Debug Loop)
- **Fail (same error)**: → Stage 2 (Re-hypothesize)
