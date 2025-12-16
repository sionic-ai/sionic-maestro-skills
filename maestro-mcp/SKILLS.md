# Maestro MCP Tools Reference

This document contains detailed documentation for all Maestro MCP tools.
Use `maestro_help(topic)` to retrieve specific sections programmatically.

---

## Tool: maestro_consult

Consult an external LLM CLI as a 'sub-agent'.

### When to Use
- Example Analysis: Understanding code structure, summarizing logs
- Hypothesis Generation: Getting diverse perspectives on root causes
- Code Review: Safety, security, and maintainability checks
- Improvement Suggestions: Refactoring ideas

### When NOT to Use
- Direct file editing (orchestrator should edit)
- Running tests or commands (orchestrator should run)

### Paper Insight
"Tool-coordination trade-off" - limit consults in tool-heavy stages.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | str | required | The specific question or task for the consultant |
| provider | str | "codex" | Which CLI to use ('codex', 'gemini', 'claude') |
| model | str | None | Specific model (uses default if not specified) |
| context_files | list | [] | File paths to include in context |
| context_facts | list | [] | Established facts about the problem |
| context_errors | list | [] | Error logs or stack traces |
| context_constraints | list | [] | Requirements that must not be violated |
| stage | str | None | Current workflow stage (for context-aware packing) |
| timeout_sec | int | 300 | Timeout for the CLI call |

### Returns
Dictionary with 'ok', 'content', 'provider', 'model', 'elapsed_ms'.

---

## Tool: maestro_ensemble_generate

Generate multiple candidate solutions using different LLMs.

### When to Use
- Hypothesis generation (get diverse root cause theories)
- Solution exploration (multiple approaches to a problem)
- Review collection (different perspectives on code quality)

### When NOT to Use
- Implementation (creates merge conflicts)
- Sequential debugging (paper shows MAS degrades here)

### Paper Insight
"Independent agents allow diverse exploration" but "error amplification" means we MUST verify before accepting.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| task | str | required | The task to generate candidates for |
| providers | list | ["codex", "gemini"] | List of providers to use |
| context_files | list | [] | Files to include in context |
| context_facts | list | [] | Known facts |
| context_errors | list | [] | Error logs |
| n_per_provider | int | 1 | Number of candidates per provider |
| timeout_sec | int | 300 | Timeout per provider |

### Returns
Dictionary with 'candidates' list, each having id, provider, content.

---

## Tool: maestro_select_best

Select the best candidate from ensemble output.

### When to Use
- After `maestro_ensemble_generate` to pick the best solution
- When you have multiple candidate patches to evaluate
- Code review selection from multiple reviewers

### Paper Insight
"Centralized verification prevents error amplification." Independent agents amplify errors 17.2x without verification!

### Selection Priority (tests_first mode - RECOMMENDED)
1. Automated tests (pytest, npm test) - HIGHEST PRIORITY
2. Static analysis (lint scores)
3. LLM Judge evaluation
4. Provider trust scores

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| candidates | list | required | List of candidates from maestro_ensemble_generate |
| mode | str | "tests_first" | Selection strategy ('tests_first', 'llm_judge', 'hybrid') |
| test_results | list | None | Optional test results [{candidate_id, passed, output}] |
| lint_results | list | None | Optional lint results [{candidate_id, score, output}] |
| judge_provider | str | "claude" | Which LLM to use for judging |
| criteria | list | ["correctness", "safety", "completeness"] | Evaluation criteria |

### Returns
Dictionary with winner_id, winner, rationale, scores.

---

## Tool: maestro_pack_context

Pack context for sub-agent calls with smart truncation.

### When to Use
- Before calling maestro_consult with large context
- When context exceeds model limits
- Preparing stage-specific context

### Paper Insight
"Information fragmentation increases coordination tax." This tool minimizes overhead by smart excerpting and prioritization.

### Stage-specific Strategies
- **analyze**: Broad context, more files
- **hypothesize**: Focused on errors and facts
- **implement**: Narrow, specific files only
- **debug**: Error-heavy, recent changes
- **improve**: Full context for review

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| files | list | [] | File paths (supports globs like 'src/*.py') |
| facts | list | [] | Known facts about the problem |
| errors | list | [] | Error logs or stack traces |
| constraints | list | [] | Requirements that must not be violated |
| stage | str | None | Current workflow stage for optimized packing |
| max_chars | int | 40000 | Maximum context characters |

### Returns
Dictionary with 'packed_context', 'stats'.

---

## Tool: maestro_workflow_state

Get the current workflow state and metrics.

### When to Use
- Check current stage and progress
- Monitor consult budget usage
- Review coordination metrics

### Returns
Dictionary with workflow state, metrics, and summary including:
- Current stage
- Consult budget used/remaining
- Stage history
- Paper-aligned metrics (overhead, efficiency)

---

## Tool: maestro_run_stage

Execute a single stage of the 5-stage workflow.

### Stages
1. **analyze**: Freeze facts before guessing
2. **hypothesize**: Generate competing explanations
3. **implement**: Apply changes safely (single agent preferred)
4. **debug**: Fix without divergence (single agent preferred)
5. **improve**: Refactor and stabilize

### Paper Insights Applied
- Tool-heavy stages (implement, debug) use single agent
- Capability saturation: Skip ensemble if baseline > 45%
- Error amplification: All outputs need verification

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| stage | str | required | Which stage to run |
| task | str | required | The task description |
| context_files | list | [] | Relevant files |
| context_facts | list | [] | Known facts |
| context_errors | list | [] | Error logs |
| providers | list | None | Override default providers |
| baseline_confidence | float | 0.0 | Your confidence level (0.0-1.0) |

### Returns
Stage result with output and next stage recommendation.

---

## Tool: maestro_get_metrics

Get detailed metrics aligned with the paper.

### Paper Metrics
- **Coordination Overhead (O%)**: Extra turns vs single-agent
- **Efficiency Score (Ec)**: Success rate / relative overhead
- **Consults per Stage**: Average sub-agent calls per stage

### When to Use
- Monitor coordination costs
- Identify when to reduce/increase collaboration
- Compare against paper benchmarks

### Returns
Detailed metrics dictionary with summary.

---

## Tool: maestro_list_providers

List available CLI providers and their status.

### Returns
Dictionary with provider names and configurations including:
- enabled status
- command
- default_model
- timeout_sec

---

## Tool: maestro_get_skill

Get the skill definition and guidance for a workflow stage.

### What's Included
- Goal and purpose
- Process steps
- Output schema (JSON format)
- Coordination policy (when to use ensemble)
- Anti-patterns to avoid
- Exit criteria

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| stage | str | Which stage's skill to retrieve ('analyze', 'hypothesize', 'implement', 'debug', 'improve') |

### Returns
Skill definition with metadata, content, and output schema.

---

## Tool: maestro_get_role

Get the role/persona prompt for a specific role.

### Available Roles
- **example_analyst**: Extracts facts, observations, constraints
- **hypothesis_scientist**: Generates testable hypotheses
- **implementer**: Makes minimal, testable code changes
- **debugger**: Systematic debugging with iteration limits
- **refiner**: Quality improvement and regression testing
- **judge**: Objective candidate selection

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| role | str | Which role's prompt to retrieve |

### Returns
Role definition with prompt content and metadata.

---

## Tool: maestro_get_schema

Get a JSON schema for output validation.

### Available Schemas
- **stage1_output**: Example Analysis output
- **stage2_output**: Hypothesis output
- **stage3_output**: Implementation output
- **stage4_output**: Debug Loop output
- **stage5_output**: Recursive Improvement output
- **judge_output**: Candidate selection output
- **tools**: All tool input/output schemas

### Returns
JSON Schema definition.

---

## Tool: maestro_consult_with_role

Consult an LLM with a specific role/persona system prompt.

### How It Works
Combines maestro_consult with role-based prompting. The role's system prompt is prepended to shape the LLM's response style.

### Role Purposes
- **example_analyst**: Factual observation extraction (Stage 1)
- **hypothesis_scientist**: Testable hypothesis generation (Stage 2)
- **implementer**: Minimal code changes (Stage 3)
- **debugger**: Systematic iteration (Stage 4)
- **refiner**: Quality improvement (Stage 5)
- **judge**: Candidate selection (Any stage)

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | str | required | The specific question or task |
| role | str | required | Which persona to use |
| provider | str | "codex" | Which CLI to use |
| context_files | list | [] | File paths to include |
| context_facts | list | [] | Established facts |
| context_errors | list | [] | Error logs or stack traces |
| stage | str | None | Current workflow stage |
| timeout_sec | int | 300 | Timeout for the CLI call |

### Returns
Dictionary with response and metadata.

---

## Tool: maestro_get_coordination_policy

Get the coordination policies based on the paper.

### Policies from "Towards a Science of Scaling Agent Systems"
- **Capability threshold**: When to skip ensemble (~45%)
- **Tool-coordination trade-off**: Which stages prefer single agent
- **Error amplification**: Why Independent topology is forbidden
- **Sequential task handling**: Why debug uses single agent

### Paper Insights Returned
- tool_coordination_tradeoff
- capability_saturation
- error_amplification
- sequential_degradation

---

## Tool: maestro_verify

Run verification commands (tests, lint, type-check).

### Paper Insight
"Test, don't vote" - Deterministic signals (test pass/fail) are more reliable than LLM consensus for code correctness.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| commands | list | required | List of command specs: [{"command": "pytest -v", "type": "unit_test"}, ...] |
| cwd | str | None | Working directory for commands |
| stop_on_failure | bool | False | Stop after first failure |
| parallel | bool | False | Run commands in parallel |

### Command Types
unit_test, lint, type_check, format, build, custom

### Example
```python
maestro_verify([
    {"command": "python -m pytest tests/ -v", "type": "unit_test"},
    {"command": "python -m ruff check src/", "type": "lint"}
])
```

### Returns
VerificationReport with pass/fail status for each command.

---

## Tool: maestro_apply_patch

Apply a unified diff patch to the workspace safely.

### Paper Insight
"Tool execution is restricted to single Executor"

### Safety Features
- Path validation (no escape from workspace)
- Allowlist-based file patterns
- Automatic backups before modification
- Structured logging of all changes

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| patch | str | required | Unified diff content (git diff format) |
| workspace_root | str | cwd | Root directory for patch application |
| dry_run | bool | False | If True, validate patch but don't apply |

### Example
```python
maestro_apply_patch('''
--- a/src/main.py
+++ b/src/main.py
@@ -10,6 +10,7 @@
 def main():
+    print("Hello")
     pass
''')
```

### Returns
PatchResult with files changed, created, failed, and backup location.

---

## Tool: maestro_consensus_vote

Run MAKER-style first-to-ahead-by-k voting on a micro-decision.

### Paper Insight (MAKER)
From "Solving a Million-Step LLM Task With Zero Errors":
- For micro-decisions, voting with error correction dramatically reduces errors
- Red-flagging (rejecting malformed responses) improves accuracy
- First-to-ahead-by-k is more efficient than fixed-round voting

### When to Use
- "Which file is most likely the source of the bug?" (selection)
- "Is this patch safe to apply?" (binary yes/no)
- "What's the key error signal in this log?" (extraction)

### When NOT to Use
- Code generation (too complex for voting)
- Implementation decisions (need context, not consensus)

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| question | str | required | The micro-decision question |
| k | int | 3 | Votes ahead needed to win |
| max_rounds | int | 12 | Maximum voting rounds |
| providers | list | ["codex", "gemini", "claude"] | Which providers to query |
| require_json | bool | False | If True, responses must be valid JSON |
| max_response_chars | int | 2000 | Maximum response length |

### Returns
ConsensusResult with winner, confidence, and vote trace.

---

## Tool: maestro_validate_content

Validate content against MAKER-style red-flag criteria.

### Paper Insight
"Format errors are signals of reasoning errors" - Rejecting malformed responses BEFORE using them improves overall accuracy.

### Red-flag Criteria
- Content too long (indicates rambling, loss of focus)
- Content too short (indicates incomplete response)
- Hedging language ("I'm not sure", "it's unclear")
- Missing required JSON fields
- Invalid diff format (for patches)

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| content | str | required | The content to validate |
| content_type | str | "general" | Type: "general", "diff", "json" |
| max_chars | int | 15000 | Maximum allowed characters |
| require_json_fields | list | [] | Required fields if content is JSON |
| forbidden_patterns | list | [] | Additional patterns to reject |

### Returns
Validation result with is_valid and reason if invalid.

---

## Tool: maestro_log_evidence

Log evidence to the reasoning chain for auditability.

### Evidence-first Approach
Every decision should link to concrete evidence. This enables post-hoc analysis of why decisions were made.

### Evidence Types
- **observation**: Facts observed from code/logs
- **hypothesis**: Proposed explanation for an issue
- **decision**: Choice made with rationale
- **verification**: Test/lint result

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| evidence_type | str | required | Type of evidence |
| stage | str | required | Workflow stage |
| content | dict | required | The actual evidence content |
| source | str | required | Where the evidence came from |
| confidence | float | 1.0 | How reliable (0.0-1.0) |
| linked_evidence_ids | list | [] | IDs of related evidence |

### Returns
The evidence ID for future reference.

---

## Tool: maestro_get_evidence_chain

Query the evidence chain for audit and analysis.

### When to Use
- Review what evidence led to a decision
- Trace back through the reasoning chain
- Identify gaps in evidence

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| evidence_type | str | None | Filter by type |
| stage | str | None | Filter by workflow stage |
| limit | int | 50 | Maximum entries to return |

### Returns
List of evidence entries with red_flag_summary.

---

## Tool: maestro_restore_from_backup

Restore files from a backup session.

### When to Use
When a patch introduced bugs and you need to rollback.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| backup_session | str | required | The backup session ID (from maestro_apply_patch) |
| workspace_root | str | cwd | Root directory for restoration |
| files | list | None | Specific files to restore (None = all) |

### Returns
List of restored files and any failures.

---

## Tool: maestro_enter_stage

Enter a workflow stage and load appropriate tools dynamically.

### MAKER Insight
"Minimize context overhead" - Only load tools needed for current stage.

### Tool Loading by Stage
- **analyze**: context packing, evidence logging
- **hypothesize**: ensemble, voting, validation
- **implement**: patch, verify
- **debug**: verify, rollback, context
- **improve**: ensemble, patch, verify

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| stage | str | Which stage to enter |

### Returns
Loaded tools list and available skills.

---

## Tool: maestro_enter_skill

Enter a specific skill for fine-grained tool loading.

### Example Skills
- **spec_extraction** (analyze): Extract I/O specs
- **edge_case_generation** (analyze): Generate edge cases
- **root_cause_analysis** (hypothesize): Find root causes
- **patch_generation** (implement): Generate patches
- **failure_classification** (debug): Classify errors
- **refactoring** (improve): Apply refactorings

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| skill_name | str | Name of the skill to enter |

### Returns
Skill info and loaded tools.

---

## Tool: maestro_get_micro_steps

Get available micro-steps for MAKER-style decomposition.

### MAKER Insight
"Maximal Agentic Decomposition" - Break tasks into atomic steps. Each micro-step is small enough for voting to be effective.

### Micro-step Types by Stage
- **Analyze**: s1_spec_extract, s2_edge_case, s3_mre
- **Hypothesize**: h1_root_cause, h2_verification
- **Implement**: c1_minimal_patch, c2_compile_check
- **Debug**: d1_failure_label, d2_next_experiment
- **Improve**: r1_refactor, r2_perf

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| stage | str | None | Stage filter (uses current if not specified) |

### Returns
List of micro-step specifications.

---

## Tool: maestro_vote_micro_step

Run MAKER-style first-to-ahead-by-k voting on a micro-step.

### MAKER Paper Insight
- Step-level voting with small k dramatically reduces error rate
- Red-flagging rejects format errors BEFORE voting
- Different step types have different default k values

This is THE core mechanism for error correction in long-horizon tasks.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| step_type | str | required | Type of micro-step |
| prompt | str | required | The task prompt |
| context | str | "" | Additional context |
| k | int | None | Votes ahead needed (uses step default) |
| max_rounds | int | 15 | Maximum voting rounds |
| providers | list | ["codex", "gemini"] | Which providers to use |

### Returns
VoteResult with winner content, confidence, and voting trace.

---

## Tool: maestro_calibrate

Calibrate voting parameters (k) for a step type.

### MAKER Paper Insight
- Different step types have different accuracy p
- k should be calibrated based on p and target success rate
- With proper calibration, even low-accuracy steps can achieve high overall success

### When to Use
Run this before a long workflow to optimize k for each step type.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| step_type | str | required | Which micro-step type |
| test_prompt | str | required | Representative prompt for sampling |
| oracle_command | str | None | Command to verify correctness |
| target_success_rate | float | 0.99 | Desired overall success rate |
| estimated_total_steps | int | 100 | Expected total steps |
| num_samples | int | 10 | Number of calibration samples |
| providers | list | ["codex", "gemini"] | Which providers to sample |

### Returns
CalibrationResult with recommended k and accuracy estimates.

---

## Tool: maestro_red_flag_check

Check content for MAKER-style red flags.

### MAKER Paper Insight
"Format errors signal reasoning errors" - Don't try to repair malformed responses. Discard and resample instead. This reduces CORRELATED errors, not just average errors.

### Red Flag Types
- **too_long**: Response exceeds max length
- **too_short**: Response incomplete
- **hedging**: Contains hedging language
- **missing_fields**: Missing required JSON fields
- **invalid_diff**: Invalid diff format
- **multi_file**: Patch affects too many files
- **dangerous_code**: Contains dangerous patterns
- **dangerous_command**: Contains dangerous commands
- **forbidden_file**: References forbidden files

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| content | str | required | Content to validate |
| step_type | str | None | Step type for step-specific rules |
| rules | list | None | Explicit rules to check |

### Returns
Red flag result with is_flagged status and reasons.

---

## Tool: maestro_get_loaded_tools

Get the list of currently loaded tools.

### When to Use
Understand what tools are available in current context. Tools are dynamically loaded based on current stage/skill.

### Returns
List of loaded tool names, context cost, and usage stats.

---

## Tool: maestro_recommend_tools

Get recommended tools for a task description.

### When to Use
Bootstrapping before entering a specific stage.

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| task_description | str | Description of what you want to accomplish |

### Returns
Recommended tools and suggested stage.

---

## Tool: maestro_exit_stage

Exit current stage and unload non-core tools.

### When to Use
Transitioning between stages to minimize context.

### What Remains Loaded
Core tools only: maestro_list_providers, maestro_get_skill, maestro_workflow_state

### Returns
Updated tool state after unloading.

---

## Tool: maestro_classify_task

Classify task structure to determine optimal coordination topology.

### Paper Rule A
"Domain/task structure dependency is absolute" - This tool analyzes the task to extract features that determine whether to use MAS (multi-agent) or SAS (single-agent).

### Key Features Extracted
- **decomposability_score**: Can task be split into parallel subtasks?
- **sequential_dependency_score**: How much does each step depend on previous?
- **tool_complexity**: How many/complex tools are needed?

### When to Use
FIRST in Stage 1 (Analyze) to inform architecture selection.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| task_description | str | required | Description of the task/problem |
| code_context | str | None | Optional code snippets |
| error_logs | str | None | Optional error logs |

### Returns
TaskStructureFeatures with scores and recommendations.

---

## Tool: maestro_select_architecture

Select the optimal coordination topology for a workflow stage.

### Paper Rules Applied
- **Rule A**: Architecture depends on task structure
- **Rule B**: Decomposable -> MAS, Sequential -> SAS
- **Rule C**: Coordination overhead is a cost to minimize
- **Rule D**: Use calibration data when available

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| stage | str | required | Current workflow stage |
| decomposability_score | float | 0.5 | How parallelizable (0-1) |
| sequential_dependency_score | float | 0.5 | How sequential (0-1) |
| tool_complexity | float | 0.3 | How complex (0-1) |
| force_topology | str | None | Override ("sas", "mas_independent", "mas_centralized") |

### Returns
CoordinationDecision with topology, parameters, and fallback plan.

---

## Tool: maestro_check_degradation

Check if coordination should degrade to a simpler topology.

### Paper Rule C
"Coordination overhead is a first-class cost function" - When MAS isn't improving results, automatically fall back to SAS.

### Degradation Triggers
- High coordination overhead without success improvement
- Error amplification > 1.0 (MAS making more errors than SAS)
- High redundancy (agents producing identical outputs)
- Consecutive format errors

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| current_topology | str | required | Current topology |
| total_messages | int | 0 | Total messages sent |
| total_rounds | int | 0 | Total coordination rounds |
| successes | int | 0 | Successful outcomes |
| failures | int | 0 | Failed outcomes |
| redundancy_rate | float | 0.0 | Agent output similarity |

### Returns
Degradation decision with new topology if needed.

---

## Tool: maestro_record_coordination_result

Record coordination result for calibration (Rule D).

### Paper Rule D
"Model family calibration is necessary" - By recording results, the system learns which topology works best for your specific codebase and task types.

### When to Use
Call after each coordination attempt to build calibration data.

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| topology | str | required | Which topology was used |
| success | bool | required | Whether coordination succeeded |
| tokens_used | int | 0 | Tokens consumed |
| messages_sent | int | 0 | Messages sent |
| rounds | int | 1 | Coordination rounds |
| outputs | list | None | Agent outputs (for redundancy) |

### Returns
Updated statistics for the topology.

---

## Tool: maestro_get_coordination_stats

Get coordination calibration statistics.

### Paper Rule D
Use calibration data to inform architecture selection.

### Statistics Per Topology
- Success rate
- Average overhead
- Average token usage
- Number of samples

### Returns
Statistics per topology and recommendations.

---

## Tool: maestro_get_stage_strategy

Get the recommended coordination strategy for a workflow stage.

### Stage-specific Strategies
- **analyze**: Parallel info gathering, score-based selection
- **hypothesize**: Parallel hypothesis gen, falsifiability scoring
- **implement**: Parallel patch gen, TEST-FIRST selection
- **debug**: SEQUENTIAL (SAS), minimize coordination
- **improve**: Parallel review, skill extraction

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| stage | str | The workflow stage |

### Returns
Detailed strategy with topology, voting mode, and red-flag rules.

---

## Topic: workflow

The Maestro 5-Stage Workflow based on "Towards a Science of Scaling Agent Systems".

### Stage Overview

1. **Analyze** (SAS): Freeze facts before guessing
2. **Hypothesize** (MAS-Centralized): Generate competing explanations
3. **Implement** (SAS): Apply changes safely
4. **Debug** (SAS only): Fix without divergence
5. **Improve** (MAS-Centralized): Refactor and stabilize

### Key Paper Insights

- **Tool-coordination trade-off**: Tool-heavy tasks suffer from MAS overhead
- **Capability saturation**: When baseline >= 45%, MAS returns diminish
- **Error amplification**: Independent agents amplify errors 17.2x
- **Sequential degradation**: Sequential tasks degrade 39-70% with MAS

---

## Topic: paper_insights

Key insights from "Towards a Science of Scaling Agent Systems".

### Rule A: Domain/Task Structure
Architecture depends on task structure. No universal best topology.

### Rule B: Decomposability
- Decomposable tasks -> MAS (parallel)
- Sequential tasks -> SAS (single agent)

### Rule C: Coordination Overhead
Coordination overhead is a first-class cost function.
O(%) = (MAS_turns - SAS_turns) / SAS_turns

### Rule D: Calibration
Model family calibration is necessary. Track success rates per topology.

### Error Amplification (Critical!)
- Independent topology: 17.2x error amplification
- Centralized topology: 4.4x error amplification
- ALWAYS use centralized verification, never pure voting

### Tool-Coordination Trade-off
β = -0.330 for tool complexity. More tools = prefer SAS.

---

## Topic: human_in_the_loop

Human-in-the-Loop (HITL) approval system for workflow stages.

### Overview
Each stage completion requires explicit human approval before proceeding.
This ensures human oversight at every critical decision point.

### Key Features
- Detailed stage reports in English and Korean (한국어)
- Priority-based review questions (Critical, High, Medium, Low)
- Feedback collection and incorporation
- Revision request support

### Workflow
1. Run stage with `maestro_run_stage_with_approval()`
2. Review the detailed report and questions
3. Submit approval with `maestro_submit_approval()`
4. If approved, proceed to next stage
5. If revision requested, re-run with updated context

---

## Tool: maestro_request_approval

Request human approval before proceeding to the next stage.

### When to Use
- After any stage completion when human oversight is required
- Automatically called by `maestro_run_stage_with_approval`

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| stage | str | required | The workflow stage (analyze, hypothesize, implement, debug, improve) |
| outputs | dict | required | The stage outputs to review |
| duration_ms | float | 0.0 | Stage execution duration |
| metrics | dict | None | Additional metrics |

### Returns
- `request_id`: Unique approval request ID
- `display`: Formatted report for display
- `report`: Full stage report with questions
- `critical_questions`: High-priority questions requiring review

---

## Tool: maestro_submit_approval

Submit approval decision for a pending stage.

### When to Use
- After reviewing stage report and answering questions
- To approve, reject, or request revision

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| request_id | str | required | The approval request ID |
| approved | bool | required | True to approve, False to reject |
| feedback | str | None | General feedback or comments |
| question_responses | dict | None | Responses to specific questions (by question ID) |
| revision_instructions | str | None | Instructions for revision if not approved |

### Returns
- `status`: approved, rejected, or revision_requested
- `next_action`: proceed_to_next_stage, revise_current_stage, or stop_workflow
- `next_stage`: The next stage if approved

---

## Tool: maestro_run_stage_with_approval

Execute a workflow stage AND automatically request human approval.

### When to Use
- This is the recommended way to run stages with human-in-the-loop
- Combines `maestro_run_stage` and `maestro_request_approval`

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| stage | str | required | The workflow stage |
| task | str | required | The task description |
| context_files | list | [] | Files to include |
| context_facts | list | [] | Known facts |
| context_errors | list | [] | Error logs |
| providers | list | None | LLM providers to use |
| baseline_confidence | float | 0.0 | Baseline confidence score |

### Returns
- `stage_result`: The stage execution result
- `approval`: The approval request details
- `action_required`: Instructions for the human reviewer

---

## Tool: maestro_get_stage_questions

Get the review questions that will be asked for a specific stage.

### When to Use
- Preview questions before running a stage
- Prepare answers in advance

### Stage Questions Summary

| Stage | Critical | High | Medium | Total |
|-------|----------|------|--------|-------|
| analyze | 2 | 2 | 1 | 5 |
| hypothesize | 2 | 1 | 2 | 5 |
| implement | 3 | 2 | 1 | 6 |
| debug | 2 | 2 | 1 | 5 |
| improve | 2 | 2 | 1 | 5 |

### Parameters
| Param | Type | Description |
|-------|------|-------------|
| stage | str | The workflow stage |

### Returns
- `questions`: List of all questions with priorities
- `by_priority`: Questions grouped by priority level

---

## Tool: maestro_get_pending_approvals

Get all pending approval requests.

### When to Use
- Check what stages are awaiting approval
- Resume work after interruption

### Returns
- `count`: Number of pending requests
- `pending_requests`: List of request details

---

## Tool: maestro_get_approval_history

Get history of approval decisions.

### When to Use
- Review past decisions
- Analyze approval patterns

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| stage | str | None | Filter by stage name |
| limit | int | 20 | Maximum records to return |

### Returns
- `history`: List of approval records
- `summary`: Counts by status (approved, rejected, revision_requested)

---

## Tool: maestro_workflow_with_hitl

Start a full workflow with human-in-the-loop at every stage.

### When to Use
- Initialize a new HITL workflow
- Get guidance on the workflow process

### Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| task | str | required | The task description |
| start_stage | str | "analyze" | Which stage to start from |
| auto_approve_low_risk | bool | False | Auto-approve low-risk stages |

### Returns
- Workflow initialization status
- Step-by-step instructions (EN/KO)
- Questions preview for each stage
