Analyze this code or issue using Maestro Skills.

## Instructions

Run the analyze stage only:
```
maestro_run_stage_with_approval(
    stage="analyze",
    task="$ARGUMENTS"
)
```

After analysis is complete:
1. Show me the analysis report
2. Wait for my approval before taking any further action
3. Do NOT proceed to other stages unless I explicitly ask

## What to Analyze

$ARGUMENTS
