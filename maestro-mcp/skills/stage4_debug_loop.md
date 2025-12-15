---
name: stage4_debug_loop
stage_id: debug
goal: "Fix without divergence - systematic iteration with minimal changes"
persona: debugger
default_topology: SAS
max_consults: 1
max_iterations: 5
ensemble_allowed: false
---

# Stage 4: Iterative Debugging Loop

## Purpose

When implementation introduces a **new error**, systematically debug without spiraling into complexity.

> "Debugging is twice as hard as writing the code in the first place." — Brian Kernighan

## When to Use

- After Stage 3 implementation fails with a **new** error
- When test passes but uncovers a different failure
- When the fix introduces a regression

## Why Single Agent (Paper-Critical)

**Sequential Task Degradation**: The paper shows that **all multi-agent variants** perform **39-70% worse** on sequential reasoning tasks.

Debugging is inherently sequential:
```
Error → Analysis → Fix → Test → (New Error?) → Loop
```

**Therefore: Single agent ONLY for debugging. No ensemble. No voting.**

## Input Requirements

```yaml
required:
  - previous_change: object     # What we just changed (Stage 3)
  - new_error: string           # The new error message
  - test_output: string         # Full test output

optional:
  - iteration_count: integer    # Which iteration is this?
  - hypothesis_history: array   # Previous hypotheses tried
  - max_iterations: integer     # Default: 5
```

## Process (Each Iteration)

### Step 1: Analyze the NEW Error
- Is this a **new** error or the **same** error?
- What changed between previous state and now?
- Is the error related to our change?

### Step 2: Update Hypothesis Confidence
- If original hypothesis seems wrong, reduce confidence
- If error is unrelated, our hypothesis might still be correct
- Document the evidence

### Step 3: Make SINGLE Smallest Change
```diff
# GOOD: Minimal fix
- return users[0]
+ return users[0] if users else None

# BAD: Multiple changes
- return users[0]
+ if not users:
+     logger.warning("No users found")
+     return self.get_default_user()
+ return users[0]
```

### Step 4: Test Again
- Run the **same** test command
- Compare output to previous iteration
- Decide next action

## Iteration Tracking

```yaml
iteration_1:
  error: "IndexError: list index out of range"
  change: "Added empty list guard"
  result: "New error: AttributeError"
  confidence: 0.6

iteration_2:
  error: "AttributeError: 'NoneType' has no attribute 'id'"
  change: "Handle None return value in caller"
  result: "Test passes"
  confidence: 0.9
  status: RESOLVED
```

## Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["iteration", "error_analysis", "change", "test_result", "next_action"],
  "properties": {
    "iteration": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10,
      "description": "Current iteration number"
    },
    "error_analysis": {
      "type": "object",
      "properties": {
        "error_type": {
          "type": "string",
          "description": "Type of error (e.g., IndexError, TypeError)"
        },
        "is_new_error": {
          "type": "boolean",
          "description": "Is this different from the original error?"
        },
        "is_related_to_change": {
          "type": "boolean",
          "description": "Does this error stem from our change?"
        },
        "root_cause_theory": {
          "type": "string",
          "description": "What we think caused this error"
        }
      }
    },
    "hypothesis_update": {
      "type": "object",
      "properties": {
        "original_hypothesis": { "type": "string" },
        "still_valid": { "type": "boolean" },
        "new_confidence": { "type": "number" },
        "evidence": { "type": "string" }
      }
    },
    "change": {
      "type": "object",
      "properties": {
        "file": { "type": "string" },
        "diff": { "type": "string" },
        "description": { "type": "string" },
        "lines_changed": {
          "type": "integer",
          "maximum": 10,
          "description": "Should be minimal"
        }
      }
    },
    "test_result": {
      "type": "object",
      "properties": {
        "command": { "type": "string" },
        "passed": { "type": "boolean" },
        "output_summary": { "type": "string" },
        "compared_to_previous": {
          "type": "string",
          "enum": ["same_error", "new_error", "passed", "regression"]
        }
      }
    },
    "next_action": {
      "type": "string",
      "enum": [
        "continue_debug",      # Another iteration
        "resolved",            # Tests pass → Stage 5
        "revert_all",          # Too many iterations, start over
        "escalate",            # Human intervention needed
        "re_hypothesize"       # Back to Stage 2
      ]
    },
    "iterations_remaining": {
      "type": "integer"
    }
  }
}
```

## Example Output

```json
{
  "iteration": 2,
  "error_analysis": {
    "error_type": "AttributeError",
    "is_new_error": true,
    "is_related_to_change": true,
    "root_cause_theory": "Our None return is not handled by caller"
  },
  "hypothesis_update": {
    "original_hypothesis": "Off-by-one error in array access",
    "still_valid": true,
    "new_confidence": 0.85,
    "evidence": "Original error fixed, new error is downstream effect"
  },
  "change": {
    "file": "src/views.py",
    "diff": "@@ -15,7 +15,8 @@\n user = get_active_user(users)\n-user_id = user.id\n+user_id = user.id if user else None",
    "description": "Handle None user in caller",
    "lines_changed": 1
  },
  "test_result": {
    "command": "pytest tests/test_auth.py::test_login -v",
    "passed": true,
    "output_summary": "1 passed in 0.12s",
    "compared_to_previous": "passed"
  },
  "next_action": "resolved",
  "iterations_remaining": 3
}
```

## Coordination Policy

| Iteration | Consult Allowed? | Rationale |
|-----------|------------------|-----------|
| 1-2 | ❌ No | Try single-agent first |
| 3 | ⚠️ Maybe | If stuck, one consult |
| 4-5 | ⚠️ Maybe | Final attempts |
| >5 | ⛔ Escalate | Human intervention |

### Why Limited Consults

From the paper:
- Sequential tasks degrade with multi-agent (-39% to -70%)
- Error amplification in debugging is dangerous
- "Discussion overhead" delays convergence

## Decision Tree

```
START: New error from Stage 3
│
├─ Is error related to our change?
│  ├─ YES → Make minimal fix, test again
│  └─ NO → Original hypothesis might be wrong
│         └─ Consider: re_hypothesize
│
├─ Iteration count check
│  ├─ < 3 → Continue single-agent debug
│  ├─ 3 → Consult ONE external agent if stuck
│  ├─ 4-5 → Final attempts
│  └─ > 5 → ESCALATE (human needed)
│
└─ Test result
   ├─ PASS → resolved → Stage 5
   ├─ SAME ERROR → Change didn't help
   │  └─ Try different approach or re_hypothesize
   ├─ NEW ERROR → Progress (maybe)
   │  └─ continue_debug
   └─ REGRESSION → Our fix broke something else
      └─ revert_all, reconsider approach
```

## Anti-patterns to Avoid

1. **Large fixes** - If you need >5 lines, step back
2. **Multiple fixes per iteration** - ONE change only
3. **Ensemble debugging** - Paper shows this fails
4. **Long discussions** - Debug, don't debate
5. **Ignoring iteration limit** - Escalate, don't spiral

## Emergency Revert

```bash
# If iterations > 5 or regression detected:
git checkout -- src/
# or
git stash
# Then re-evaluate from Stage 1 or 2
```

## Exit Criteria

- [ ] Test passes (→ Stage 5)
- [ ] OR max iterations reached (→ Escalate)
- [ ] OR regression detected (→ Revert, Stage 2)
- [ ] Changes are minimal and documented
- [ ] Hypothesis confidence updated

## Next Stage

- **Resolved**: → Stage 5 (Recursive Improvement)
- **Max iterations**: → Escalate to human
- **Regression**: → Revert, return to Stage 2
