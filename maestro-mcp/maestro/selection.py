"""
Selection Engine for choosing the best candidate from ensemble outputs.

Implements the Poetiq ARC solver pattern with modifications based on
"Towards a Science of Scaling Agent Systems" and MAKER:

Selection Priority (tests_first):
1. Automated testing (pytest, npm test, etc.) - HIGHEST PRIORITY
2. Static analysis (lint, type check)
3. LLM Judge with rubric
4. Simple voting - LAST RESORT

Key insights:
- "Independent agents amplify errors 17.2x without verification" - Always verify
- MAKER red-flagging: Malformed responses indicate reasoning errors - reject them

Red-flagging criteria (from MAKER):
- Response too long (> max_tokens)
- Response too short (empty/trivial)
- Schema validation failure
- Forbidden patterns (hedging, uncertainty markers)
- Diff format violations (for code patches)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple

logger = logging.getLogger("maestro.selection")


# ============================================================================
# Red-Flagging Configuration (MAKER-style)
# ============================================================================

@dataclass
class RedFlagConfig:
    """
    Configuration for red-flagging invalid/suspicious responses.

    Based on MAKER paper: "Format errors are signals of reasoning errors"
    Rejecting malformed responses BEFORE voting dramatically improves accuracy.
    """
    # Length constraints
    max_chars: int = 15000  # ~3750 tokens
    min_chars: int = 10

    # For diff/patch content specifically
    max_diff_chars: int = 50000
    require_diff_format: bool = False  # If True, must contain valid diff markers

    # Forbidden patterns (hedging, uncertainty = reasoning failure)
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r"(?i)i('m| am) not (sure|certain)",
        r"(?i)i don'?t know",
        r"(?i)i cannot (determine|say|tell)",
        r"(?i)it'?s (unclear|ambiguous|hard to say)",
        r"(?i)there('s| is) no (clear|obvious|definitive)",
        r"(?i)this (could|might|may) be (wrong|incorrect)",
    ])

    # Required patterns (must match at least one if specified)
    required_patterns: List[str] = field(default_factory=list)

    # JSON schema validation
    require_json: bool = False
    json_required_fields: List[str] = field(default_factory=list)

    # Diff format validation
    diff_required_markers: List[str] = field(default_factory=lambda: [
        r"^[+-]{3}",  # --- or +++
        r"^@@.*@@",   # Hunk headers
    ])


class RedFlagResult(Enum):
    """Result of red-flag validation."""
    VALID = "valid"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    FORBIDDEN_PATTERN = "forbidden_pattern"
    MISSING_REQUIRED = "missing_required"
    INVALID_JSON = "invalid_json"
    MISSING_JSON_FIELD = "missing_json_field"
    INVALID_DIFF = "invalid_diff"
    CUSTOM_FAILURE = "custom_failure"


def validate_candidate_content(
    content: str,
    config: Optional[RedFlagConfig] = None,
    content_type: str = "general",  # "general", "diff", "json"
) -> Tuple[bool, RedFlagResult, Optional[str]]:
    """
    Validate candidate content against red-flag criteria.

    Args:
        content: The candidate content to validate
        config: Red-flag configuration
        content_type: Type of content for specialized validation

    Returns:
        (is_valid, result_code, error_message)
    """
    cfg = config or RedFlagConfig()

    # Length checks
    max_len = cfg.max_diff_chars if content_type == "diff" else cfg.max_chars
    if len(content) > max_len:
        return False, RedFlagResult.TOO_LONG, f"Content too long: {len(content)} > {max_len}"

    if len(content) < cfg.min_chars:
        return False, RedFlagResult.TOO_SHORT, f"Content too short: {len(content)} < {cfg.min_chars}"

    # Forbidden patterns (hedging/uncertainty)
    for pattern in cfg.forbidden_patterns:
        if re.search(pattern, content):
            return False, RedFlagResult.FORBIDDEN_PATTERN, f"Contains forbidden pattern: {pattern}"

    # Required patterns
    if cfg.required_patterns:
        if not any(re.search(p, content) for p in cfg.required_patterns):
            return False, RedFlagResult.MISSING_REQUIRED, "Missing required pattern"

    # JSON validation
    if cfg.require_json or content_type == "json":
        try:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content.strip())

            # Check required fields
            if cfg.json_required_fields and isinstance(parsed, dict):
                missing = [f for f in cfg.json_required_fields if f not in parsed]
                if missing:
                    return False, RedFlagResult.MISSING_JSON_FIELD, f"Missing JSON fields: {missing}"

        except json.JSONDecodeError as e:
            return False, RedFlagResult.INVALID_JSON, f"Invalid JSON: {e}"

    # Diff format validation
    if content_type == "diff" or cfg.require_diff_format:
        has_diff_markers = False
        for marker in cfg.diff_required_markers:
            if re.search(marker, content, re.MULTILINE):
                has_diff_markers = True
                break

        # Also check for code block with diff
        if not has_diff_markers:
            if "```diff" in content or "```patch" in content:
                has_diff_markers = True

        if cfg.require_diff_format and not has_diff_markers:
            return False, RedFlagResult.INVALID_DIFF, "Missing diff format markers"

    return True, RedFlagResult.VALID, None


# ============================================================================
# Selection Mode and Data Classes
# ============================================================================

class SelectionMode(Enum):
    """Selection strategies in order of preference."""
    TESTS_FIRST = "tests_first"      # Run tests, pick passing candidate
    LLM_JUDGE = "llm_judge"          # Have an LLM evaluate and pick
    HYBRID = "hybrid"                 # Tests + LLM for ties
    VOTING = "voting"                 # Simple majority (last resort)


@dataclass
class Candidate:
    """A single candidate solution from ensemble generation."""
    id: str
    provider: str
    content: str
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Evaluation results (filled in by selection process)
    test_passed: Optional[bool] = None
    test_output: Optional[str] = None
    lint_score: Optional[float] = None
    lint_output: Optional[str] = None
    judge_score: Optional[float] = None
    judge_rationale: Optional[str] = None
    final_score: Optional[float] = None

    # Red-flag validation (MAKER-style)
    red_flagged: bool = False
    red_flag_reason: Optional[str] = None
    red_flag_result: Optional[RedFlagResult] = None


@dataclass
class SelectionResult:
    """Result of the selection process."""
    winner_id: str
    winner: Candidate
    rationale: str
    scores: Dict[str, float]
    mode_used: SelectionMode
    all_candidates: List[Candidate]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSignal:
    """Test results for a candidate."""
    candidate_id: str
    passed: bool
    output: str
    exit_code: int = 0
    duration_ms: float = 0.0


@dataclass
class LintSignal:
    """Lint/static analysis results for a candidate."""
    candidate_id: str
    score: float  # 0.0 to 1.0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    output: str = ""


class SelectionEngine:
    """
    Selects the best candidate from an ensemble using multiple signals.

    Philosophy: "Test, don't vote" + MAKER red-flagging
    - Paper shows Independent agents amplify errors without verification
    - Automated tests catch errors that LLMs miss
    - Red-flagging: Reject malformed responses before selection (MAKER)
    - LLM judge is a fallback, not primary
    """

    # Scoring weights for hybrid mode
    DEFAULT_WEIGHTS = {
        "test": 0.60,      # Tests are king
        "lint": 0.15,      # Code quality
        "judge": 0.15,     # LLM evaluation
        "minimal_diff": 0.05,  # Prefer smaller changes
        "provider_trust": 0.05,  # Provider-specific trust scores
    }

    # Provider trust scores (can be tuned based on experience)
    PROVIDER_TRUST = {
        "codex": 0.85,     # Good at code
        "claude": 0.80,    # Good at reasoning
        "gemini": 0.75,    # Good at context
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        judge_provider: str = "claude",
        judge_rubric: Optional[List[str]] = None,
        red_flag_config: Optional[RedFlagConfig] = None,
        enable_red_flagging: bool = True,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.judge_provider = judge_provider
        self.judge_rubric = judge_rubric or [
            "correctness",
            "safety",
            "completeness",
            "maintainability",
        ]
        self.red_flag_config = red_flag_config or RedFlagConfig()
        self.enable_red_flagging = enable_red_flagging

    def select(
        self,
        candidates: List[Candidate],
        mode: SelectionMode = SelectionMode.TESTS_FIRST,
        test_signals: Optional[List[TestSignal]] = None,
        lint_signals: Optional[List[LintSignal]] = None,
        judge_callback: Optional[Callable] = None,
        content_type: str = "general",  # For red-flag validation
    ) -> SelectionResult:
        """
        Select the best candidate.

        Args:
            candidates: List of Candidate objects
            mode: Selection strategy
            test_signals: Test results (if available)
            lint_signals: Lint results (if available)
            judge_callback: Function to call LLM judge (provider_registry.run)
            content_type: Type of content for red-flag validation ("general", "diff", "json")

        Returns:
            SelectionResult with winner and rationale
        """
        if not candidates:
            raise ValueError("No candidates provided")

        # Apply red-flagging FIRST (MAKER principle)
        if self.enable_red_flagging:
            self._apply_red_flags(candidates, content_type)
            valid_candidates = [c for c in candidates if not c.red_flagged]

            if not valid_candidates:
                # All candidates red-flagged - return best of bad options with warning
                logger.warning("All candidates were red-flagged!")
                return SelectionResult(
                    winner_id=candidates[0].id,
                    winner=candidates[0],
                    rationale="ALL CANDIDATES RED-FLAGGED - selection unreliable",
                    scores={c.id: 0.0 for c in candidates},
                    mode_used=mode,
                    all_candidates=candidates,
                    metadata={
                        "warning": "All candidates failed validation",
                        "requires_review": True,
                        "red_flagged_count": len(candidates),
                        "red_flag_reasons": {c.id: c.red_flag_reason for c in candidates},
                    },
                )

            # Log red-flag statistics
            flagged_count = len(candidates) - len(valid_candidates)
            if flagged_count > 0:
                logger.info(f"Red-flagged {flagged_count}/{len(candidates)} candidates")
        else:
            valid_candidates = candidates

        if len(valid_candidates) == 1:
            # Single valid candidate
            winner = valid_candidates[0]
            flagged_count = len(candidates) - 1
            rationale = "Single valid candidate"
            if flagged_count > 0:
                rationale += f" ({flagged_count} red-flagged)"

            return SelectionResult(
                winner_id=winner.id,
                winner=winner,
                rationale=rationale,
                scores={winner.id: 1.0},
                mode_used=mode,
                all_candidates=candidates,
                metadata={"red_flagged_count": flagged_count} if flagged_count > 0 else {},
            )

        # Apply signals to valid candidates
        self._apply_test_signals(valid_candidates, test_signals)
        self._apply_lint_signals(valid_candidates, lint_signals)

        # Select based on mode (using only valid candidates)
        if mode == SelectionMode.TESTS_FIRST:
            result = self._select_tests_first(valid_candidates)
        elif mode == SelectionMode.LLM_JUDGE:
            result = self._select_llm_judge(valid_candidates, judge_callback)
        elif mode == SelectionMode.HYBRID:
            result = self._select_hybrid(valid_candidates, judge_callback)
        elif mode == SelectionMode.VOTING:
            result = self._select_voting(valid_candidates)
        else:
            raise ValueError(f"Unknown selection mode: {mode}")

        # Include all candidates in result (including red-flagged)
        result.all_candidates = candidates

        # Add red-flag metadata
        flagged_count = len(candidates) - len(valid_candidates)
        if flagged_count > 0:
            result.metadata["red_flagged_count"] = flagged_count
            result.metadata["red_flag_reasons"] = {
                c.id: c.red_flag_reason for c in candidates if c.red_flagged
            }

        return result

    def _apply_red_flags(
        self,
        candidates: List[Candidate],
        content_type: str = "general",
    ) -> None:
        """Apply red-flag validation to all candidates."""
        for cand in candidates:
            is_valid, result, reason = validate_candidate_content(
                cand.content,
                self.red_flag_config,
                content_type,
            )
            cand.red_flagged = not is_valid
            cand.red_flag_result = result
            cand.red_flag_reason = reason

    def _apply_test_signals(
        self,
        candidates: List[Candidate],
        signals: Optional[List[TestSignal]],
    ) -> None:
        """Apply test signals to candidates."""
        if not signals:
            return

        signal_map = {s.candidate_id: s for s in signals}
        for cand in candidates:
            if cand.id in signal_map:
                sig = signal_map[cand.id]
                cand.test_passed = sig.passed
                cand.test_output = sig.output

    def _apply_lint_signals(
        self,
        candidates: List[Candidate],
        signals: Optional[List[LintSignal]],
    ) -> None:
        """Apply lint signals to candidates."""
        if not signals:
            return

        signal_map = {s.candidate_id: s for s in signals}
        for cand in candidates:
            if cand.id in signal_map:
                sig = signal_map[cand.id]
                cand.lint_score = sig.score
                cand.lint_output = sig.output

    def _select_tests_first(self, candidates: List[Candidate]) -> SelectionResult:
        """
        Select based on test results first.

        Priority:
        1. Tests pass -> winner
        2. If multiple pass -> use lint scores
        3. If none pass -> pick best lint score
        4. If no lint -> pick first (arbitrary)
        """
        passing = [c for c in candidates if c.test_passed is True]

        if len(passing) == 1:
            winner = passing[0]
            return SelectionResult(
                winner_id=winner.id,
                winner=winner,
                rationale=f"Only passing candidate ({winner.provider})",
                scores=self._compute_scores(candidates),
                mode_used=SelectionMode.TESTS_FIRST,
                all_candidates=candidates,
            )

        if len(passing) > 1:
            # Tie-break with lint scores
            for c in passing:
                if c.lint_score is None:
                    c.lint_score = 0.5  # Neutral default

            best = max(passing, key=lambda c: c.lint_score or 0)
            return SelectionResult(
                winner_id=best.id,
                winner=best,
                rationale=f"Best lint score among {len(passing)} passing candidates",
                scores=self._compute_scores(candidates),
                mode_used=SelectionMode.TESTS_FIRST,
                all_candidates=candidates,
            )

        # No passing candidates - fall back to lint
        if any(c.lint_score is not None for c in candidates):
            best = max(candidates, key=lambda c: c.lint_score or 0)
            return SelectionResult(
                winner_id=best.id,
                winner=best,
                rationale="No tests passed; selected by lint score",
                scores=self._compute_scores(candidates),
                mode_used=SelectionMode.TESTS_FIRST,
                all_candidates=candidates,
                metadata={"warning": "No tests passed"},
            )

        # No signals at all - need test results!
        logger.warning("No test or lint signals - selection is unreliable")
        return SelectionResult(
            winner_id=candidates[0].id,
            winner=candidates[0],
            rationale="No signals available - REQUIRES MANUAL REVIEW",
            scores=self._compute_scores(candidates),
            mode_used=SelectionMode.TESTS_FIRST,
            all_candidates=candidates,
            metadata={"warning": "No verification signals", "requires_review": True},
        )

    def _select_llm_judge(
        self,
        candidates: List[Candidate],
        judge_callback: Optional[Callable],
    ) -> SelectionResult:
        """Select using LLM judge evaluation."""
        if not judge_callback:
            logger.warning("No judge callback provided, falling back to voting")
            return self._select_voting(candidates)

        # Build judge prompt
        prompt = self._build_judge_prompt(candidates)

        try:
            # Call judge
            response = judge_callback(self.judge_provider, prompt)

            if response.ok:
                # Parse judge response
                winner_id, scores, rationale = self._parse_judge_response(
                    response.stdout, candidates
                )

                for cand in candidates:
                    if cand.id in scores:
                        cand.judge_score = scores[cand.id]

                winner = next((c for c in candidates if c.id == winner_id), candidates[0])

                return SelectionResult(
                    winner_id=winner_id,
                    winner=winner,
                    rationale=rationale,
                    scores=scores,
                    mode_used=SelectionMode.LLM_JUDGE,
                    all_candidates=candidates,
                )
        except Exception as e:
            logger.exception("Judge evaluation failed")

        # Fallback
        return self._select_voting(candidates)

    def _select_hybrid(
        self,
        candidates: List[Candidate],
        judge_callback: Optional[Callable],
    ) -> SelectionResult:
        """
        Hybrid selection: Combine test, lint, and judge scores.
        """
        # Get judge scores if available
        if judge_callback:
            try:
                prompt = self._build_judge_prompt(candidates)
                response = judge_callback(self.judge_provider, prompt)
                if response.ok:
                    _, judge_scores, _ = self._parse_judge_response(
                        response.stdout, candidates
                    )
                    for cand in candidates:
                        if cand.id in judge_scores:
                            cand.judge_score = judge_scores[cand.id]
            except Exception as e:
                logger.warning(f"Judge failed in hybrid mode: {e}")

        # Compute final scores
        final_scores = {}
        for cand in candidates:
            score = 0.0

            # Test score (binary: 1.0 if passed, 0.0 if failed, 0.5 if unknown)
            if cand.test_passed is True:
                test_score = 1.0
            elif cand.test_passed is False:
                test_score = 0.0
            else:
                test_score = 0.5
            score += self.weights["test"] * test_score

            # Lint score
            lint_score = cand.lint_score if cand.lint_score is not None else 0.5
            score += self.weights["lint"] * lint_score

            # Judge score
            judge_score = cand.judge_score if cand.judge_score is not None else 0.5
            score += self.weights["judge"] * judge_score

            # Provider trust
            trust = self.PROVIDER_TRUST.get(cand.provider, 0.7)
            score += self.weights["provider_trust"] * trust

            # Minimal diff bonus (estimate from content length)
            # Shorter is better, but this is a weak signal
            avg_len = sum(len(c.content) for c in candidates) / len(candidates)
            diff_score = 1.0 - min(1.0, len(cand.content) / (avg_len * 2))
            score += self.weights["minimal_diff"] * diff_score

            cand.final_score = score
            final_scores[cand.id] = score

        # Pick winner
        winner = max(candidates, key=lambda c: c.final_score or 0)

        return SelectionResult(
            winner_id=winner.id,
            winner=winner,
            rationale=f"Hybrid score: {winner.final_score:.3f}",
            scores=final_scores,
            mode_used=SelectionMode.HYBRID,
            all_candidates=candidates,
        )

    def _select_voting(self, candidates: List[Candidate]) -> SelectionResult:
        """
        Simple voting fallback.

        Warning: Paper shows this amplifies errors!
        Only use when no other signals are available.
        """
        logger.warning(
            "Using voting selection - this may amplify errors. "
            "Consider providing test/lint signals."
        )

        # Provider diversity as a weak signal
        providers = [c.provider for c in candidates]
        unique_providers = len(set(providers))

        # If all from same provider, just pick first
        if unique_providers == 1:
            return SelectionResult(
                winner_id=candidates[0].id,
                winner=candidates[0],
                rationale="Single provider, no diversity for voting",
                scores={c.id: 1.0 / len(candidates) for c in candidates},
                mode_used=SelectionMode.VOTING,
                all_candidates=candidates,
                metadata={"warning": "Voting without test signals is unreliable"},
            )

        # Pick the one from most "trusted" provider
        best = max(candidates, key=lambda c: self.PROVIDER_TRUST.get(c.provider, 0.5))
        return SelectionResult(
            winner_id=best.id,
            winner=best,
            rationale=f"Selected by provider trust ({best.provider})",
            scores={c.id: self.PROVIDER_TRUST.get(c.provider, 0.5) for c in candidates},
            mode_used=SelectionMode.VOTING,
            all_candidates=candidates,
            metadata={"warning": "Voting without test signals is unreliable"},
        )

    def _compute_scores(self, candidates: List[Candidate]) -> Dict[str, float]:
        """Compute scores for all candidates."""
        scores = {}
        for cand in candidates:
            score = 0.0
            if cand.test_passed is True:
                score += 0.6
            if cand.lint_score is not None:
                score += 0.2 * cand.lint_score
            if cand.judge_score is not None:
                score += 0.2 * cand.judge_score
            scores[cand.id] = score
        return scores

    def _build_judge_prompt(self, candidates: List[Candidate]) -> str:
        """Build the prompt for LLM judge."""
        candidates_text = ""
        for cand in candidates:
            candidates_text += f"\n--- Candidate {cand.id} ({cand.provider}) ---\n"
            # Truncate for judging
            content = cand.content[:3000] if len(cand.content) > 3000 else cand.content
            candidates_text += content
            candidates_text += "\n"

        return f"""You are a Lead Engineer reviewing candidate solutions.

**Evaluation Criteria**: {', '.join(self.judge_rubric)}

**Candidates**:
{candidates_text}

**Instructions**:
1. Score each candidate 0.0-1.0 on each criterion
2. Identify the winner with rationale
3. Note any risks in the winning solution

**Output Format** (JSON):
```json
{{
  "winner_id": "c1",
  "scores": {{
    "c1": 0.85,
    "c2": 0.72
  }},
  "rationale": "...",
  "risks": ["..."]
}}
```"""

    def _parse_judge_response(
        self,
        response: str,
        candidates: List[Candidate],
    ) -> tuple[str, Dict[str, float], str]:
        """Parse the judge's response."""
        # Try to extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return (
                    data.get("winner_id", candidates[0].id),
                    data.get("scores", {}),
                    data.get("rationale", "Judge selected"),
                )
            except json.JSONDecodeError:
                pass

        # Fallback: look for patterns
        for cand in candidates:
            if f"winner" in response.lower() and cand.id in response:
                return cand.id, {}, "Parsed from judge response"

        # Default to first
        return candidates[0].id, {}, "Could not parse judge response"
