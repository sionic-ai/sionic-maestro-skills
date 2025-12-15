---
name: stage1_example_analysis
stage_id: analyze
goal: "Freeze facts before guessing - extract observations, constraints, and unknowns"
persona: example_analyst
default_topology: SAS
max_consults: 2
---

# Stage 1: Example Analysis

## Purpose

Before forming any hypothesis, we must establish ground truth. This stage is about **observation, not interpretation**.

> "The scientist is not a person who gives the right answers, he's one who asks the right questions." — Claude Lévi-Strauss

## When to Use

- Starting a bug investigation
- Analyzing unfamiliar code
- Understanding test failures
- Reviewing error logs

## Input Requirements

```yaml
required:
  - task_description: string  # What problem are we solving?
  - error_logs: string[]      # Stack traces, test output, etc.

optional:
  - source_files: string[]    # Relevant code files
  - test_files: string[]      # Test files if available
  - repro_steps: string[]     # Known reproduction steps
  - previous_attempts: string # What was already tried
```

## Process

### Step 1: Gather Context
- Read all provided files
- Identify the exact error message and location
- Note the test command and environment

### Step 2: Extract Observations (Facts Only)
- What does the error say literally?
- What line/function does it occur in?
- What are the input values at the point of failure?
- What is the expected vs actual behavior?

### Step 3: Identify Constraints
- What must NOT change? (API contracts, interfaces)
- What dependencies exist?
- What invariants must be preserved?

### Step 4: Document Unknowns
- What information is missing?
- What assumptions are being made?
- What needs verification?

## Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["observations", "constraints", "unknowns", "repro_steps", "confidence"],
  "properties": {
    "observations": {
      "type": "array",
      "description": "Factual observations without interpretation",
      "items": {
        "type": "object",
        "properties": {
          "fact": { "type": "string" },
          "source": { "type": "string" },
          "line_number": { "type": "integer" }
        }
      }
    },
    "constraints": {
      "type": "array",
      "description": "Things that must NOT change",
      "items": { "type": "string" }
    },
    "unknowns": {
      "type": "array",
      "description": "Missing information or assumptions",
      "items": { "type": "string" }
    },
    "repro_steps": {
      "type": "array",
      "description": "Minimal steps to reproduce",
      "items": { "type": "string" }
    },
    "affected_modules": {
      "type": "array",
      "items": { "type": "string" }
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Confidence that we have sufficient information"
    }
  }
}
```

## Example Output

```json
{
  "observations": [
    {
      "fact": "IndexError: list index out of range",
      "source": "src/auth.py",
      "line_number": 42
    },
    {
      "fact": "users list is empty when accessed",
      "source": "test_auth.py::test_login",
      "line_number": 15
    },
    {
      "fact": "Database fixture returns empty list for test user",
      "source": "conftest.py",
      "line_number": 23
    }
  ],
  "constraints": [
    "Must not change the User model interface",
    "Must preserve backward compatibility with existing API",
    "Test must remain independent (no shared state)"
  ],
  "unknowns": [
    "Is the empty list intentional for this test case?",
    "Are there other tests that depend on this behavior?",
    "When was this test last passing?"
  ],
  "repro_steps": [
    "cd project_root",
    "pytest tests/test_auth.py::test_login -v",
    "Observe IndexError on line 42"
  ],
  "affected_modules": [
    "src/auth.py",
    "tests/test_auth.py",
    "tests/conftest.py"
  ],
  "confidence": 0.7
}
```

## Coordination Policy

| Condition | Action |
|-----------|--------|
| Simple, clear error | SAS (no external consult) |
| Large codebase, many files | Consult Gemini for summarization |
| Complex dependencies | Consult Claude for dependency analysis |
| Max consults reached | Proceed with available observations |

## Anti-patterns to Avoid

1. **Jumping to conclusions** - Don't hypothesize yet
2. **Ignoring constraints** - Document what must be preserved
3. **Over-reading files** - Focus on relevant code only
4. **Missing the obvious** - Check the error message literally

## Exit Criteria

- [ ] All observations are factual (no "maybe" or "probably")
- [ ] Reproduction steps are verified
- [ ] Constraints are documented
- [ ] Unknowns are explicitly listed
- [ ] Confidence score reflects information completeness

## Next Stage

If confidence >= 0.6: Proceed to **Stage 2: Hypothesis**
If confidence < 0.6: Gather more information or escalate
