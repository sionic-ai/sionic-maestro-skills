Consult another LLM using Maestro Skills.

## Instructions

Use `maestro_consult` to get an opinion from another LLM:
```
maestro_consult(
    prompt="$ARGUMENTS",
    provider="codex"  # or "gemini" or "claude"
)
```

If I want multiple opinions, use:
```
maestro_ensemble_generate(
    task="$ARGUMENTS",
    providers=["codex", "gemini"]
)
```

## Provider Recommendations

- **codex**: Best for code generation and implementation questions
- **gemini**: Best for long context analysis and documentation
- **claude**: Best for reasoning, review, and complex logic

## Question to Ask

$ARGUMENTS
