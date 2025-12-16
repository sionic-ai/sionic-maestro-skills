Run the full 5-stage Maestro Skills workflow with Human-in-the-Loop approval.

## Instructions

1. Initialize the HITL workflow:
   ```
   maestro_workflow_with_hitl(task="$ARGUMENTS")
   ```

2. Execute each stage sequentially, waiting for my approval after each:
   - **Stage 1: Analyze** - Gather facts and observations
   - **Stage 2: Hypothesize** - Generate competing hypotheses
   - **Stage 3: Implement** - Apply minimal, testable changes
   - **Stage 4: Debug** - Fix any test failures (single agent only)
   - **Stage 5: Improve** - Refactor and add regression tests

3. For each stage, use:
   ```
   maestro_run_stage_with_approval(stage="<stage_name>", task="$ARGUMENTS")
   ```

4. After each stage:
   - Show me the detailed report
   - Answer my review questions
   - Wait for my approval via `maestro_submit_approval`

## Important Rules

- **Never skip stages** without explicit approval
- **Debug stage**: Use single agent only (MAS degrades 39-70% per paper)
- **Provide context**: Pass `context_files`, `context_facts`, `context_errors` between stages

## Task to Execute

$ARGUMENTS
