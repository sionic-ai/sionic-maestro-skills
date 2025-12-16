# Maestro Skills

A Claude Code Skill implementing **Centralized Consult** architecture for Multi-LLM coding workflows. Like a maestro conducting an orchestra, Claude Code orchestrates multiple LLM models to produce harmonious, accurate output.

Based on:
- **"Towards a Science of Scaling Agent Systems"** (Kim et al., 2025)
- **"Solving a Million-Step LLM Task With Zero Errors"** (MAKER)

## Key Features

- **Multi-LLM Orchestration**: Coordinate Codex, Gemini, and Claude CLIs
- **5-Stage Workflow**: Analyze → Hypothesize → Implement → Debug → Improve
- **Human-in-the-Loop (HITL)**: Require human approval at each stage
- **Paper-Aligned**: Implements measured coordination principles
- **40+ Tools**: Comprehensive toolkit for complex coding tasks

## Quick Start

```bash
cd maestro-mcp
make install        # Install with uv (Python 3.11)
make check          # Verify installation
make mcp-config     # Get MCP config for Claude Code
```

Or manually:
```bash
cd maestro-mcp
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code (Orchestrator)                   │
│  - File editing, test running, tool execution                    │
│  - Final decision maker                                          │
└─────────────────┬───────────────────────────────────────────────┘
                  │ MCP Protocol
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Maestro Skills Server                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Codex     │  │   Gemini    │  │   Claude    │  Consultants │
│  │   (Code)    │  │  (Context)  │  │  (Review)   │  (Text only) │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  5-Stage Workflow Engine (FSM) + Human-in-the-Loop        │ │
│  │  Analyze → Hypothesize → Implement → Debug → Improve       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## How to Use Maestro Skills

### Method 1: Explicit Request
Simply mention "maestro" in your request:
```
"Use maestro to debug this authentication bug"
"Debug this with maestro and get approval at each step"
"Ask codex using maestro what it thinks about this code"
```

### Method 2: Slash Commands
Use the built-in slash commands:
```
/maestro-debug Fix the login bug in auth.py
/maestro-analyze Review the payment processing code
/maestro-consult What's the best way to handle this error?
/maestro-workflow Implement user session management
```

### Method 3: Direct Tool Calls
Call Maestro tools directly in conversation:
```
"Call maestro_workflow_with_hitl to start debugging"
"Use maestro_consult with codex to get a second opinion"
```

## Human-in-the-Loop (HITL) Workflow

Every workflow stage requires explicit human approval before proceeding:

```python
# 1. Start HITL workflow
maestro_workflow_with_hitl(task="Fix authentication bug")

# 2. Run stage with automatic approval request
result = maestro_run_stage_with_approval(stage="analyze", task="Fix auth bug")

# 3. Review report and submit approval
maestro_submit_approval(
    request_id=result["approval"]["request_id"],
    approved=True,
    feedback="Analysis looks complete"
)

# 4. Continue to next stage...
```

Each stage presents:
- Detailed bilingual reports (EN/KO)
- Priority-based review questions (Critical, High, Medium)
- Key findings and risk assessments
- Next stage preview

## Available Tools (40+)

### Core Tools
| Tool | Purpose |
|------|---------|
| `maestro_consult` | Single model consultation |
| `maestro_ensemble_generate` | Multi-model candidate generation |
| `maestro_select_best` | Pick best candidate with tests |
| `maestro_verify` | Run tests/lint/type-check |

### HITL Tools
| Tool | Purpose |
|------|---------|
| `maestro_workflow_with_hitl` | Start HITL workflow |
| `maestro_run_stage_with_approval` | Run stage + request approval |
| `maestro_submit_approval` | Submit approval decision |
| `maestro_get_pending_approvals` | View pending approvals |

### Workflow Tools
| Tool | Purpose |
|------|---------|
| `maestro_run_stage` | Execute workflow stage |
| `maestro_workflow_state` | Check progress |
| `maestro_classify_task` | Analyze task structure |
| `maestro_select_architecture` | Choose SAS vs MAS |

## Configuration

### Environment Variables

```bash
# Provider Models
MAESTRO_CODEX_MODEL=gpt-5.1-codex-max
MAESTRO_GEMINI_MODEL=gemini-3-pro-preview
MAESTRO_CLAUDE_MODEL=opus

# Timeouts
MAESTRO_CODEX_TIMEOUT=900
MAESTRO_GEMINI_TIMEOUT=600
MAESTRO_CLAUDE_TIMEOUT=600

# Coordination
MAESTRO_CAPABILITY_THRESHOLD=0.45
MAESTRO_MAX_CONSULT_PER_STAGE=2
MAESTRO_MAX_CONSULT_TOTAL=6
```

### MCP Configuration

Add to `~/.claude.json` (global) or `.mcp.json` (project):

```json
{
  "mcpServers": {
    "maestro-mcp": {
      "command": "/path/to/maestro-mcp/.venv/bin/python",
      "args": ["/path/to/maestro-mcp/server.py"]
    }
  }
}
```

## Project Structure

```
maestro-mcp/
├── Makefile               # make install, make check, etc.
├── server.py              # MCP server entry point (40+ tools)
├── requirements.txt       # Dependencies
├── SKILLS.md              # Tool documentation
├── conf/
│   ├── cli_clients.yaml   # CLI command templates
│   └── skill_manifest.yaml
├── skills/                # Stage skill definitions
├── roles/                 # Persona prompts
├── schemas/               # Output JSON schemas
└── maestro/
    ├── config.py          # Configuration management
    ├── providers.py       # CLI provider implementations
    ├── workflow.py        # 5-stage workflow engine
    ├── human_loop.py      # Human-in-the-Loop system
    ├── consensus.py       # MAKER-style voting
    ├── coordination.py    # Architecture Selection Engine
    └── ...
```

## Paper-Aligned Principles

| Finding | Implementation |
|---------|---------------|
| Tool-Coordination Trade-off | Only orchestrator runs tools |
| Capability Saturation (~45%) | Skip ensemble when confident |
| Error Amplification (17.2x) | Tests-first selection |
| Sequential Task Degradation | Single-agent for debug |
| Red-flagging (MAKER) | Reject malformed responses |

## References

- [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296) (Kim et al., 2025)
- [Solving a Million-Step LLM Task With Zero Errors](https://arxiv.org/) (MAKER)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## License

MIT
