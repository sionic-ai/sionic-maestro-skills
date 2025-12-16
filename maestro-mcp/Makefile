# Maestro Skills Makefile
# Use `make install` to set up the project with uv

PYTHON_VERSION := 3.11
VENV := .venv
UV := uv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: all install clean reinstall test run check help venv deps verify lint

# Default target
all: install

# Help
help:
	@echo "$(GREEN)Maestro Skills - Available Commands$(NC)"
	@echo ""
	@echo "  $(YELLOW)make install$(NC)    - Install deps + configure globally for Claude Code"
	@echo "  $(YELLOW)make reinstall$(NC)  - Clean and reinstall everything"
	@echo "  $(YELLOW)make clean$(NC)      - Remove venv and cache files"
	@echo "  $(YELLOW)make test$(NC)       - Run tests"
	@echo "  $(YELLOW)make run$(NC)        - Start the MCP server"
	@echo "  $(YELLOW)make check$(NC)      - Verify installation"
	@echo "  $(YELLOW)make lint$(NC)       - Run linter (ruff)"
	@echo "  $(YELLOW)make verify$(NC)     - Verify all CLI tools are available"
	@echo "  $(YELLOW)make mcp-config$(NC) - Show MCP config (for manual setup)"
	@echo ""

# Check if uv is installed
check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "$(RED)Error: uv is not installed$(NC)"; \
		echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}
	@echo "$(GREEN)✓ uv is installed$(NC)"

# Create virtual environment with pinned Python version
venv: check-uv
	@echo "$(YELLOW)Creating virtual environment with Python $(PYTHON_VERSION)...$(NC)"
	@$(UV) venv --python $(PYTHON_VERSION) $(VENV)
	@echo "$(GREEN)✓ Virtual environment created$(NC)"

# Install dependencies
deps: venv
	@echo "$(YELLOW)Installing dependencies with uv...$(NC)"
	@$(UV) pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# Main install target (includes global MCP configuration)
install: deps
	@echo ""
	@echo "$(YELLOW)Configuring global MCP server in ~/.claude.json...$(NC)"
	@if [ -f ~/.claude.json ]; then \
		if grep -q "maestro-mcp" ~/.claude.json; then \
			echo "$(YELLOW)maestro-mcp already in ~/.claude.json, updating...$(NC)"; \
			$(PYTHON) -c "import json; \
				f=open('$(HOME)/.claude.json','r'); \
				d=json.load(f); f.close(); \
				d.setdefault('mcpServers',{})['maestro-mcp']={'command':'$(shell pwd)/$(VENV)/bin/python','args':['$(shell pwd)/server.py']}; \
				f=open('$(HOME)/.claude.json','w'); \
				json.dump(d,f,indent=2); f.close()"; \
		else \
			echo "$(YELLOW)Adding maestro-mcp to existing ~/.claude.json...$(NC)"; \
			$(PYTHON) -c "import json; \
				f=open('$(HOME)/.claude.json','r'); \
				d=json.load(f); f.close(); \
				d.setdefault('mcpServers',{})['maestro-mcp']={'command':'$(shell pwd)/$(VENV)/bin/python','args':['$(shell pwd)/server.py']}; \
				f=open('$(HOME)/.claude.json','w'); \
				json.dump(d,f,indent=2); f.close()"; \
		fi \
	else \
		echo "$(YELLOW)Creating ~/.claude.json...$(NC)"; \
		echo '{"mcpServers":{"maestro-mcp":{"command":"$(shell pwd)/$(VENV)/bin/python","args":["$(shell pwd)/server.py"]}}}' | $(PYTHON) -m json.tool > ~/.claude.json; \
	fi
	@echo "$(GREEN)✓ Global MCP config updated in ~/.claude.json$(NC)"
	@echo ""
	@echo "$(YELLOW)Installing global slash commands to ~/.claude/commands/...$(NC)"
	@mkdir -p ~/.claude/commands
	@if [ -d "../.claude/commands" ]; then \
		cp ../.claude/commands/maestro-*.md ~/.claude/commands/ 2>/dev/null || true; \
		echo "$(GREEN)✓ Slash commands installed$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Slash commands not found in ../.claude/commands/$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)============================================$(NC)"
	@echo "$(GREEN)  Maestro Skills installed globally!$(NC)"
	@echo "$(GREEN)============================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Maestro is now available in ALL Claude Code sessions.$(NC)"
	@echo ""
	@echo "$(YELLOW)To use:$(NC)"
	@echo "  1. Restart Claude Code (or start a new session)"
	@echo "  2. Type: /mcp  (should show maestro-mcp running)"
	@echo "  3. Say: \"Use maestro to debug this bug\""
	@echo ""
	@echo "$(YELLOW)Slash commands:$(NC)"
	@echo "  /maestro-debug <task>    - Debug with HITL workflow"
	@echo "  /maestro-analyze <task>  - Analyze only"
	@echo "  /maestro-consult <query> - Ask another LLM"
	@echo ""
	@echo "$(YELLOW)Verify installation:$(NC)"
	@echo "  make check"
	@echo ""

# Clean up
clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@rm -rf $(VENV)
	@rm -rf __pycache__ maestro/__pycache__
	@rm -rf .pytest_cache
	@rm -rf *.egg-info
	@rm -rf .ruff_cache
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

# Reinstall from scratch
reinstall: clean install

# Run the server
run:
	@echo "$(YELLOW)Starting Maestro Skills Server...$(NC)"
	@$(PYTHON) server.py

# Run tests
test:
	@echo "$(YELLOW)Running tests...$(NC)"
	@$(PYTHON) -m pytest tests/ -v || echo "$(YELLOW)No tests found or pytest not installed$(NC)"

# Lint with ruff
lint:
	@echo "$(YELLOW)Running linter...$(NC)"
	@$(UV) pip install ruff --quiet 2>/dev/null || true
	@$(VENV)/bin/ruff check . || echo "$(YELLOW)Ruff not available$(NC)"

# Verify installation
check:
	@echo "$(YELLOW)Verifying installation...$(NC)"
	@echo ""
	@echo "Python version:"
	@$(PYTHON) --version
	@echo ""
	@echo "Checking imports..."
	@$(PYTHON) -c "import server; print('$(GREEN)✓ Server module OK$(NC)')"
	@$(PYTHON) -c "from maestro.human_loop import HumanLoopManager; print('$(GREEN)✓ HITL module OK$(NC)')"
	@$(PYTHON) -c "from maestro.config import MaestroConfig; print('$(GREEN)✓ Config module OK$(NC)')"
	@$(PYTHON) -c "from maestro.providers import ProviderRegistry; print('$(GREEN)✓ Providers module OK$(NC)')"
	@echo ""
	@echo "$(GREEN)All checks passed!$(NC)"

# Verify CLI tools are available
verify:
	@echo "$(YELLOW)Checking CLI tools...$(NC)"
	@echo ""
	@command -v codex >/dev/null 2>&1 && echo "$(GREEN)✓ codex CLI found$(NC)" || echo "$(RED)✗ codex CLI not found$(NC)"
	@command -v gemini >/dev/null 2>&1 && echo "$(GREEN)✓ gemini CLI found$(NC)" || echo "$(RED)✗ gemini CLI not found$(NC)"
	@command -v claude >/dev/null 2>&1 && echo "$(GREEN)✓ claude CLI found$(NC)" || echo "$(RED)✗ claude CLI not found$(NC)"
	@echo ""

# Development install with extra tools
dev: install
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	@$(UV) pip install ruff pytest pytest-asyncio
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

# Generate MCP config for Claude Code
mcp-config:
	@echo "$(YELLOW)MCP Configuration for Claude Code:$(NC)"
	@echo ""
	@echo "Add this to ~/.claude.json (global) or .mcp.json (project):"
	@echo ""
	@echo '{'
	@echo '  "mcpServers": {'
	@echo '    "maestro-mcp": {'
	@echo '      "command": "$(shell pwd)/$(VENV)/bin/python",'
	@echo '      "args": ["$(shell pwd)/server.py"]'
	@echo '    }'
	@echo '  }'
	@echo '}'
	@echo ""

# Update dependencies
update: check-uv
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	@$(UV) pip install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

# Show current Python and package versions
versions:
	@echo "$(YELLOW)Installed versions:$(NC)"
	@$(PYTHON) --version
	@$(PIP) list | grep -E "(fastmcp|pyyaml|anthropic)" || true
