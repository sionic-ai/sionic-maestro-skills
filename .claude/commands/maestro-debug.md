Debug this issue using Maestro Skills with Human-in-the-Loop approval.

## Instructions

1. First, initialize the HITL workflow:
   ```
   maestro_workflow_with_hitl(task="$ARGUMENTS")
   ```

2. Run the analyze stage and wait for approval:
   ```
   maestro_run_stage_with_approval(stage="analyze", task="$ARGUMENTS")
   ```

3. After I approve, run hypothesize stage:
   ```
   maestro_run_stage_with_approval(stage="hypothesize", task="$ARGUMENTS")
   ```

4. After I approve, run implement stage:
   ```
   maestro_run_stage_with_approval(stage="implement", task="$ARGUMENTS")
   ```

5. After I approve, run debug stage (if tests fail):
   ```
   maestro_run_stage_with_approval(stage="debug", task="$ARGUMENTS")
   ```

6. After I approve, run improve stage:
   ```
   maestro_run_stage_with_approval(stage="improve", task="$ARGUMENTS")
   ```

**Important**:
- Wait for my approval at each stage before proceeding
- Use `maestro_submit_approval` to record my decision
- Do NOT skip stages without approval

## Task to Debug

$ARGUMENTS
