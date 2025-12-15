---
name: stage5_recursive_improve
stage_id: improve
goal: "Refactor and stabilize - prevent recurrence, add coverage"
persona: refiner
default_topology: CENTRALIZED
max_consults: 2
ensemble_allowed: true
entry_condition: ALL_TESTS_PASS
---

# Stage 5: Recursive Improvement

## Purpose

After the fix works, **stabilize and prevent recurrence**. This is NOT about perfection—it's about sustainability.

> "Make it work, make it right, make it fast." — Kent Beck

## When to Use

- **ONLY** after all tests pass (Entry condition!)
- After Stage 3 or Stage 4 resolves the issue
- Before declaring the task complete

## Entry Condition (Critical)

```python
assert all_tests_pass(), "Cannot enter Stage 5 without passing tests"
```

**DO NOT enter this stage if tests are failing.** Return to Stage 4.

## Why Ensemble is OK Here (Paper-Aligned)

Unlike implementation and debugging:
- **Low tool intensity** - Mostly analysis and suggestions
- **Parallelizable** - Review perspectives can be generated independently
- **Central verification** - Human/orchestrator makes final decisions
- **Low error amplification** - Wrong suggestions don't break anything

## Input Requirements

```yaml
required:
  - all_changes: array          # All changes from Stage 3/4
  - test_results: object        # Proof that tests pass
  - original_observations: array # From Stage 1

optional:
  - constraints: string[]       # From Stage 1
  - similar_issues: string[]    # Historical data
  - coverage_report: object     # Test coverage
```

## Process

### Step 1: Review the Solution
- Does the fix address the root cause or just the symptom?
- Is the change minimal and focused?
- Are there any code smells introduced?

### Step 2: Identify Edge Cases
- What similar inputs could trigger the same bug?
- Are there boundary conditions not covered?
- Are there related functions with the same pattern?

### Step 3: Add Regression Tests
```python
# Example: If we fixed an empty-list bug, add these tests:
def test_empty_users_returns_none():
    assert get_active_user([]) is None

def test_single_user_returns_user():
    assert get_active_user([user1]).id == user1.id

def test_multiple_users_returns_first():
    assert get_active_user([user1, user2]).id == user1.id
```

### Step 4: Document Lessons Learned
- What was the root cause?
- Why wasn't it caught earlier?
- What could prevent similar bugs?

### Step 5: Termination Check (Poetiq-style)
Ask: "If I run another iteration, will I learn anything new?"
- If NO → **STOP** (diminishing returns)
- If YES → Continue with specific goal

## Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["summary", "regression_tests", "lessons", "status"],
  "properties": {
    "summary": {
      "type": "object",
      "properties": {
        "root_cause": {
          "type": "string",
          "description": "Final determination of root cause"
        },
        "fix_description": {
          "type": "string",
          "description": "What was changed and why"
        },
        "files_modified": {
          "type": "array",
          "items": { "type": "string" }
        },
        "total_lines_changed": {
          "type": "integer"
        }
      }
    },
    "regression_tests": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "test_name": { "type": "string" },
          "test_file": { "type": "string" },
          "description": { "type": "string" },
          "edge_case_covered": { "type": "string" }
        }
      }
    },
    "edge_cases_identified": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "case": { "type": "string" },
          "test_added": { "type": "boolean" },
          "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"]
          }
        }
      }
    },
    "code_quality": {
      "type": "object",
      "properties": {
        "lint_clean": { "type": "boolean" },
        "type_hints_complete": { "type": "boolean" },
        "docstrings_added": { "type": "boolean" },
        "complexity_acceptable": { "type": "boolean" }
      }
    },
    "lessons": {
      "type": "object",
      "properties": {
        "root_cause_category": {
          "type": "string",
          "enum": [
            "off_by_one",
            "null_reference",
            "type_error",
            "race_condition",
            "logic_error",
            "missing_validation",
            "configuration",
            "dependency",
            "other"
          ]
        },
        "prevention_checklist": {
          "type": "array",
          "items": { "type": "string" }
        },
        "similar_patterns_to_check": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "status": {
      "type": "string",
      "enum": ["complete", "needs_review", "escalate"]
    },
    "termination_reason": {
      "type": "string",
      "description": "Why we stopped improving"
    }
  }
}
```

## Example Output

```json
{
  "summary": {
    "root_cause": "Missing empty-list guard in get_active_user()",
    "fix_description": "Added None return for empty list, updated 2 callers to handle None",
    "files_modified": ["src/auth.py", "src/views.py"],
    "total_lines_changed": 5
  },
  "regression_tests": [
    {
      "test_name": "test_get_active_user_empty_list",
      "test_file": "tests/test_auth.py",
      "description": "Verify None returned for empty users list",
      "edge_case_covered": "Empty input"
    },
    {
      "test_name": "test_get_active_user_single_user",
      "test_file": "tests/test_auth.py",
      "description": "Verify single user returned correctly",
      "edge_case_covered": "Boundary: single element"
    }
  ],
  "edge_cases_identified": [
    {
      "case": "None passed instead of empty list",
      "test_added": true,
      "risk_level": "medium"
    },
    {
      "case": "List with None elements",
      "test_added": false,
      "risk_level": "low"
    }
  ],
  "code_quality": {
    "lint_clean": true,
    "type_hints_complete": true,
    "docstrings_added": true,
    "complexity_acceptable": true
  },
  "lessons": {
    "root_cause_category": "missing_validation",
    "prevention_checklist": [
      "Always validate collection is non-empty before indexing",
      "Return Optional types when None is possible",
      "Add explicit test for empty inputs"
    ],
    "similar_patterns_to_check": [
      "src/users.py:get_first_admin()",
      "src/teams.py:get_primary_member()"
    ]
  },
  "status": "complete",
  "termination_reason": "All edge cases covered, lint clean, no similar patterns found with same bug"
}
```

## Coordination Policy

| Task | Ensemble? | Rationale |
|------|-----------|-----------|
| Security review | ✅ Yes | Diverse perspectives valuable |
| Edge case brainstorm | ✅ Yes | Multiple angles help |
| Code quality check | ⚠️ Maybe | Usually SAS sufficient |
| Writing tests | ❌ No | Orchestrator does this |
| Documentation | ❌ No | Orchestrator does this |

## Termination Criteria (Poetiq-Inspired)

Stop when **any** of these are true:

1. **Coverage satisfied**: New regression tests cover the fix
2. **Diminishing returns**: Another iteration won't add value
3. **Budget exhausted**: Max consults reached
4. **Time limit**: Total workflow time exceeded
5. **Human satisfied**: User approves the solution

```python
def should_terminate(state):
    return (
        state.regression_tests_added >= 2 and
        state.lint_clean and
        state.no_new_edge_cases_found
    )
```

## What NOT to Do

1. **Over-engineer** - Don't add features beyond the fix
2. **Refactor everything** - Focus on changed code only
3. **Gold-plate** - Good enough is good enough
4. **Endless iteration** - Know when to stop
5. **Skip tests** - Regression tests are mandatory

## Lessons Learned Template

```markdown
## Bug Report: [Short Title]

**Date**: YYYY-MM-DD
**Files**: src/auth.py, tests/test_auth.py

### Root Cause
[1-2 sentences describing the actual cause]

### Fix Applied
[Brief description of changes]

### Prevention Checklist
- [ ] Check 1
- [ ] Check 2
- [ ] Check 3

### Similar Patterns to Audit
- `file:function` - reason
- `file:function` - reason
```

## Exit Criteria

- [ ] All tests still pass
- [ ] At least 1 regression test added
- [ ] Lessons learned documented
- [ ] No obvious edge cases unaddressed
- [ ] Code quality acceptable
- [ ] Termination reason documented

## Workflow Complete

After Stage 5 completes successfully:
1. Generate final summary
2. Update metrics (coordination overhead, efficiency)
3. Archive artifacts
4. Ready for commit/PR
