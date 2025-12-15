---
name: stage2_hypothesis
stage_id: hypothesize
goal: "Generate competing explanations with testable predictions"
persona: hypothesis_scientist
default_topology: CENTRALIZED
max_consults: 3
ensemble_allowed: true
---

# Stage 2: Hypothesis Formulation

## Purpose

Transform observations into **testable hypotheses**. Each hypothesis must have a clear **prediction** that can be verified or falsified.

> "A theory that explains everything explains nothing." — Karl Popper

## When to Use

- After Stage 1 analysis is complete
- When the root cause is unclear
- When multiple explanations are possible
- Before making any code changes

## Input Requirements

```yaml
required:
  - stage1_output: object     # Output from Example Analysis
  - task_description: string  # Original problem statement

optional:
  - codebase_context: string  # Additional code context
  - similar_issues: string[]  # Past similar bugs/fixes
```

## Process

### Step 1: Review Stage 1 Output
- Read all observations carefully
- Note the constraints (what must be preserved)
- Identify the unknowns (gaps in knowledge)

### Step 2: Generate Candidate Hypotheses
For each hypothesis:
1. State the **claim** (what do you think is wrong?)
2. Define the **prediction** (if this is true, what should happen when we test it?)
3. Specify the **verification test** (how do we prove/disprove it?)
4. Rate **confidence** (how likely is this hypothesis?)

### Step 3: Rank by Falsifiability
- Prefer hypotheses that are easy to test
- Prefer hypotheses that make specific predictions
- Avoid vague or unfalsifiable claims

### Step 4: Select Top Hypothesis
- Use tests-first selection (not voting!)
- If tests not available, use structured evaluation

## Ensemble Policy (Paper-Aligned)

**This is the BEST stage for multi-agent ensemble!**

| Condition | Ensemble Allowed |
|-----------|------------------|
| Multiple possible causes | ✅ Yes - generate diverse candidates |
| Baseline confidence < 45% | ✅ Yes - need exploration |
| Simple, obvious cause | ❌ No - SAS sufficient |
| Already have 3+ hypotheses | ❌ No - focus on evaluation |

### Why Ensemble Works Here

1. **Low tool intensity** - Hypothesis generation is text-heavy
2. **Parallelizable** - Each model can generate independently
3. **Central verification** - Final selection through tests, not voting
4. **Error containment** - Wrong hypotheses filtered by verification

## Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["hypotheses", "selected", "verification_plan"],
  "properties": {
    "hypotheses": {
      "type": "array",
      "minItems": 1,
      "maxItems": 5,
      "items": {
        "type": "object",
        "required": ["id", "claim", "prediction", "verification_test", "confidence"],
        "properties": {
          "id": {
            "type": "string",
            "pattern": "^H[0-9]+$",
            "description": "Hypothesis ID (H1, H2, etc.)"
          },
          "claim": {
            "type": "string",
            "description": "What is the hypothesized root cause?"
          },
          "prediction": {
            "type": "string",
            "description": "If true, what should we observe?"
          },
          "verification_test": {
            "type": "string",
            "description": "How to prove/disprove this hypothesis"
          },
          "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          },
          "counter_evidence": {
            "type": "string",
            "description": "What would disprove this hypothesis?"
          },
          "provider": {
            "type": "string",
            "description": "Which model generated this (for ensemble)"
          }
        }
      }
    },
    "selected": {
      "type": "string",
      "pattern": "^H[0-9]+$",
      "description": "ID of the selected hypothesis"
    },
    "selection_reason": {
      "type": "string",
      "description": "Why this hypothesis was selected"
    },
    "verification_plan": {
      "type": "object",
      "properties": {
        "test_command": {
          "type": "string",
          "description": "Command to run for verification"
        },
        "expected_result": {
          "type": "string",
          "description": "What result confirms the hypothesis?"
        },
        "timeout_seconds": {
          "type": "integer",
          "default": 60
        }
      }
    },
    "ensemble_used": {
      "type": "boolean",
      "description": "Was multi-model ensemble used?"
    }
  }
}
```

## Example Output

```json
{
  "hypotheses": [
    {
      "id": "H1",
      "claim": "Off-by-one error: code accesses users[0] but list can be empty",
      "prediction": "Adding an empty-list guard should prevent IndexError",
      "verification_test": "Add test case with empty users list, verify no crash",
      "confidence": 0.75,
      "counter_evidence": "If error persists with guard, cause is elsewhere",
      "provider": "codex"
    },
    {
      "id": "H2",
      "claim": "Test fixture bug: fixture should return at least one user",
      "prediction": "Fixing fixture to return mock user should make test pass",
      "verification_test": "Check if other tests using same fixture pass",
      "confidence": 0.60,
      "counter_evidence": "If other tests pass with same fixture, this is wrong",
      "provider": "gemini"
    },
    {
      "id": "H3",
      "claim": "Race condition: users populated asynchronously",
      "prediction": "Adding await/sync should resolve timing issue",
      "verification_test": "Add debug logging with timestamps",
      "confidence": 0.30,
      "counter_evidence": "If timestamps show sync execution, this is wrong",
      "provider": "claude"
    }
  ],
  "selected": "H1",
  "selection_reason": "Most testable, highest confidence, matches error pattern",
  "verification_plan": {
    "test_command": "pytest tests/test_auth.py::test_login_empty_users -v",
    "expected_result": "Test passes without IndexError",
    "timeout_seconds": 30
  },
  "ensemble_used": true
}
```

## Selection Algorithm (Poetiq-Style)

```
Priority Order:
1. TEST SIGNAL (highest priority)
   - If verification test can be run, prefer testable hypotheses
   - Hypothesis that passes its own verification test wins

2. LINT/STATIC ANALYSIS
   - Prefer hypotheses that identify static-analysis issues
   - Cross-reference with linter output

3. LLM JUDGE (if no test signal)
   - Central judge evaluates based on rubric:
     * Specificity (is the claim precise?)
     * Testability (can we verify it?)
     * Coherence with observations
     * Prior probability (how common is this bug type?)

4. VOTING (last resort only)
   - Only if above signals unavailable
   - NOT for final decision - only for candidate ranking
```

## Anti-patterns to Avoid

1. **Vague hypotheses** - "Something is wrong with auth" ❌
2. **Untestable claims** - "The code is bad" ❌
3. **Skipping verification** - Every hypothesis needs a test
4. **Independent voting** - Paper shows 17.2x error amplification!

## Exit Criteria

- [ ] At least 2 competing hypotheses generated
- [ ] Each hypothesis has a verification test
- [ ] One hypothesis is selected with clear reasoning
- [ ] Verification plan is executable

## Next Stage

Proceed to **Stage 3: Implementation** with selected hypothesis
