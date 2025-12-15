"""
Consensus Engine implementing MAKER-style micro-decision voting.

Based on "Solving a Million-Step LLM Task With Zero Errors" (MAKER):
- First-to-ahead-by-k voting: Stop when one answer leads by k votes
- Red-flagging: Reject malformed/overlong responses before voting
- Semantic equivalence: Group equivalent answers before counting

Key insight: For micro-decisions (small, verifiable choices), voting with
error correction dramatically reduces cumulative error rates.

Use cases in coding:
- "Which file is most likely the source of the bug?" (Top-k selection)
- "Is this patch safe to apply?" (Binary decision)
- "What's the key error signal in this log?" (Extraction)
- "Rank these hypotheses by likelihood" (Ordering)
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
import time

logger = logging.getLogger("maestro.consensus")


class VoteStatus(Enum):
    """Status of a single vote."""
    VALID = "valid"
    RED_FLAGGED = "red_flagged"
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"


class StopReason(Enum):
    """Reason why voting stopped."""
    AHEAD_BY_K = "ahead_by_k"         # Winner achieved k-vote lead
    MAX_ROUNDS = "max_rounds"         # Hit maximum rounds limit
    UNANIMOUS = "unanimous"           # All votes agree
    BUDGET_EXHAUSTED = "budget"       # Cost/time budget exhausted
    NO_VALID_VOTES = "no_valid"       # All responses red-flagged


@dataclass
class Vote:
    """A single vote from a provider."""
    round: int
    provider: str
    model: str
    raw_response: str
    parsed_value: Optional[Any]
    value_hash: str  # For equivalence grouping
    status: VoteStatus
    red_flag_reason: Optional[str] = None
    elapsed_ms: float = 0.0


@dataclass
class RedFlagConfig:
    """Configuration for red-flagging invalid responses."""
    # Maximum response length (tokens approximated by chars/4)
    max_tokens: int = 1200
    max_chars: int = 4800  # ~1200 tokens

    # Minimum response length (reject empty/trivial)
    min_chars: int = 1

    # Required patterns (must match at least one)
    required_patterns: List[str] = field(default_factory=list)

    # Forbidden patterns (reject if any match)
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r'(?i)i don\'t know',
        r'(?i)i\'m not sure',
        r'(?i)i cannot determine',
        r'(?i)error:.*exception',
    ])

    # JSON schema validation (if output should be JSON)
    require_json: bool = False
    json_schema: Optional[Dict[str, Any]] = None

    # Custom validator function
    custom_validator: Optional[Callable[[str], Tuple[bool, str]]] = None


@dataclass
class ConsensusConfig:
    """Configuration for consensus voting."""
    # First-to-ahead-by-k parameter
    k: int = 3

    # Maximum voting rounds
    max_rounds: int = 15

    # Providers to use (cycled through)
    providers: List[str] = field(default_factory=lambda: ["codex", "gemini", "claude"])

    # Equivalence mode: 'exact', 'normalized', 'semantic'
    equivalence_mode: str = "normalized"

    # Red flag configuration
    red_flags: RedFlagConfig = field(default_factory=RedFlagConfig)

    # Early stop if unanimous after this many votes
    unanimous_threshold: int = 3

    # Cost budget (arbitrary units, provider-specific)
    max_cost: Optional[float] = None


@dataclass
class ConsensusResult:
    """Result of consensus voting."""
    winner: Any
    winner_hash: str
    votes_for_winner: int
    total_valid_votes: int
    total_votes: int
    stop_reason: StopReason
    vote_trace: List[Vote]
    vote_distribution: Dict[str, int]  # hash -> count
    confidence: float  # winner_votes / total_valid
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "winner_hash": self.winner_hash,
            "votes_for_winner": self.votes_for_winner,
            "total_valid_votes": self.total_valid_votes,
            "total_votes": self.total_votes,
            "stop_reason": self.stop_reason.value,
            "confidence": self.confidence,
            "vote_distribution": self.vote_distribution,
            "vote_trace": [
                {
                    "round": v.round,
                    "provider": v.provider,
                    "status": v.status.value,
                    "value_hash": v.value_hash[:8] if v.value_hash else None,
                    "red_flag": v.red_flag_reason,
                }
                for v in self.vote_trace
            ],
            "metadata": self.metadata,
        }


class ConsensusEngine:
    """
    Implements MAKER-style first-to-ahead-by-k voting for micro-decisions.

    Algorithm:
    1. Query providers in round-robin
    2. Red-flag and reject malformed responses
    3. Hash valid responses for equivalence grouping
    4. Track vote counts per hash
    5. Stop when one answer leads by k votes

    Key adaptations from MAKER for coding:
    - Semantic equivalence for code/text (not just exact match)
    - Schema validation for structured outputs
    - Provider-specific red-flag thresholds
    """

    def __init__(
        self,
        config: Optional[ConsensusConfig] = None,
        provider_callback: Optional[Callable] = None,
    ):
        """
        Args:
            config: Voting configuration
            provider_callback: Function(provider, prompt) -> response
        """
        self.config = config or ConsensusConfig()
        self.provider_callback = provider_callback

    def vote(
        self,
        question: str,
        response_parser: Optional[Callable[[str], Any]] = None,
        equivalence_fn: Optional[Callable[[Any, Any], bool]] = None,
        provider_callback: Optional[Callable] = None,
    ) -> ConsensusResult:
        """
        Run first-to-ahead-by-k voting on a question.

        Args:
            question: The micro-decision question
            response_parser: Function to parse response into structured value
            equivalence_fn: Custom equivalence checker for two parsed values
            provider_callback: Override the default provider callback

        Returns:
            ConsensusResult with winner and vote trace
        """
        callback = provider_callback or self.provider_callback
        if not callback:
            raise ValueError("No provider_callback specified")

        votes: List[Vote] = []
        vote_counts: Dict[str, int] = {}  # hash -> count
        hash_to_value: Dict[str, Any] = {}  # hash -> parsed value
        provider_idx = 0

        for round_num in range(1, self.config.max_rounds + 1):
            # Select provider (round-robin)
            provider = self.config.providers[provider_idx % len(self.config.providers)]
            provider_idx += 1

            # Query provider
            start_time = time.time()
            try:
                response = callback(provider, question)
                raw_text = response.stdout if hasattr(response, 'stdout') else str(response)
                elapsed_ms = (time.time() - start_time) * 1000
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
                votes.append(Vote(
                    round=round_num,
                    provider=provider,
                    model="",
                    raw_response="",
                    parsed_value=None,
                    value_hash="",
                    status=VoteStatus.TIMEOUT,
                    elapsed_ms=(time.time() - start_time) * 1000,
                ))
                continue

            # Red-flag check
            is_valid, flag_reason = self._check_red_flags(raw_text)
            if not is_valid:
                votes.append(Vote(
                    round=round_num,
                    provider=provider,
                    model=getattr(response, 'model', ''),
                    raw_response=raw_text[:500],
                    parsed_value=None,
                    value_hash="",
                    status=VoteStatus.RED_FLAGGED,
                    red_flag_reason=flag_reason,
                    elapsed_ms=elapsed_ms,
                ))
                logger.debug(f"Red-flagged response from {provider}: {flag_reason}")
                continue

            # Parse response
            try:
                if response_parser:
                    parsed = response_parser(raw_text)
                else:
                    parsed = self._default_parser(raw_text)
            except Exception as e:
                votes.append(Vote(
                    round=round_num,
                    provider=provider,
                    model=getattr(response, 'model', ''),
                    raw_response=raw_text[:500],
                    parsed_value=None,
                    value_hash="",
                    status=VoteStatus.PARSE_ERROR,
                    red_flag_reason=str(e),
                    elapsed_ms=elapsed_ms,
                ))
                continue

            # Compute hash for equivalence grouping
            value_hash = self._compute_hash(parsed, equivalence_fn, hash_to_value)
            hash_to_value[value_hash] = parsed

            # Record vote
            vote_counts[value_hash] = vote_counts.get(value_hash, 0) + 1
            votes.append(Vote(
                round=round_num,
                provider=provider,
                model=getattr(response, 'model', ''),
                raw_response=raw_text[:500],
                parsed_value=parsed,
                value_hash=value_hash,
                status=VoteStatus.VALID,
                elapsed_ms=elapsed_ms,
            ))

            # Check stopping conditions
            stop_reason, winner_hash = self._check_stop_conditions(vote_counts, len(votes))
            if stop_reason:
                winner = hash_to_value.get(winner_hash)
                total_valid = sum(1 for v in votes if v.status == VoteStatus.VALID)

                return ConsensusResult(
                    winner=winner,
                    winner_hash=winner_hash,
                    votes_for_winner=vote_counts.get(winner_hash, 0),
                    total_valid_votes=total_valid,
                    total_votes=len(votes),
                    stop_reason=stop_reason,
                    vote_trace=votes,
                    vote_distribution=vote_counts,
                    confidence=vote_counts.get(winner_hash, 0) / max(total_valid, 1),
                    metadata={"rounds": round_num},
                )

        # Max rounds reached - pick plurality winner
        if vote_counts:
            winner_hash = max(vote_counts, key=vote_counts.get)
            winner = hash_to_value.get(winner_hash)
        else:
            winner_hash = ""
            winner = None

        total_valid = sum(1 for v in votes if v.status == VoteStatus.VALID)

        return ConsensusResult(
            winner=winner,
            winner_hash=winner_hash,
            votes_for_winner=vote_counts.get(winner_hash, 0) if winner_hash else 0,
            total_valid_votes=total_valid,
            total_votes=len(votes),
            stop_reason=StopReason.MAX_ROUNDS if vote_counts else StopReason.NO_VALID_VOTES,
            vote_trace=votes,
            vote_distribution=vote_counts,
            confidence=vote_counts.get(winner_hash, 0) / max(total_valid, 1) if winner_hash else 0,
            metadata={"rounds": self.config.max_rounds},
        )

    def _check_red_flags(self, response: str) -> Tuple[bool, Optional[str]]:
        """Check response against red-flag criteria."""
        rf = self.config.red_flags

        # Length checks
        if len(response) > rf.max_chars:
            return False, f"Response too long ({len(response)} > {rf.max_chars} chars)"

        if len(response) < rf.min_chars:
            return False, f"Response too short ({len(response)} chars)"

        # Required patterns
        if rf.required_patterns:
            if not any(re.search(p, response) for p in rf.required_patterns):
                return False, "Missing required pattern"

        # Forbidden patterns
        for pattern in rf.forbidden_patterns:
            if re.search(pattern, response):
                return False, f"Contains forbidden pattern: {pattern}"

        # JSON validation
        if rf.require_json:
            try:
                # Try to extract JSON from response
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json.loads(json_match.group(1))
                else:
                    json.loads(response)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"

        # Custom validator
        if rf.custom_validator:
            is_valid, reason = rf.custom_validator(response)
            if not is_valid:
                return False, reason

        return True, None

    def _default_parser(self, response: str) -> Any:
        """Default response parser - try JSON, fall back to text."""
        # Try to extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try direct JSON parse
        response_stripped = response.strip()
        if response_stripped.startswith("{") or response_stripped.startswith("["):
            try:
                return json.loads(response_stripped)
            except json.JSONDecodeError:
                pass

        # Fall back to normalized text
        return self._normalize_text(response)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Lowercase
        text = text.lower()
        # Remove common filler words
        text = re.sub(r'\b(the|a|an|is|are|was|were)\b', '', text)
        return text.strip()

    def _compute_hash(
        self,
        value: Any,
        equivalence_fn: Optional[Callable],
        existing_values: Dict[str, Any],
    ) -> str:
        """Compute hash for a value, checking equivalence with existing values."""
        # If custom equivalence, check against existing values
        if equivalence_fn:
            for existing_hash, existing_value in existing_values.items():
                try:
                    if equivalence_fn(value, existing_value):
                        return existing_hash
                except Exception:
                    continue

        # Compute hash based on equivalence mode
        if self.config.equivalence_mode == "exact":
            hash_input = json.dumps(value, sort_keys=True, default=str)
        elif self.config.equivalence_mode == "normalized":
            if isinstance(value, str):
                hash_input = self._normalize_text(value)
            elif isinstance(value, dict):
                # Normalize dict keys and string values
                normalized = self._normalize_dict(value)
                hash_input = json.dumps(normalized, sort_keys=True, default=str)
            else:
                hash_input = json.dumps(value, sort_keys=True, default=str)
        else:  # semantic - use exact for now, could add embedding-based
            hash_input = json.dumps(value, sort_keys=True, default=str)

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _normalize_dict(self, d: Dict) -> Dict:
        """Recursively normalize a dictionary."""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k.lower()] = self._normalize_text(v)
            elif isinstance(v, dict):
                result[k.lower()] = self._normalize_dict(v)
            elif isinstance(v, list):
                result[k.lower()] = [
                    self._normalize_dict(x) if isinstance(x, dict)
                    else self._normalize_text(x) if isinstance(x, str)
                    else x
                    for x in v
                ]
            else:
                result[k.lower()] = v
        return result

    def _check_stop_conditions(
        self,
        vote_counts: Dict[str, int],
        total_votes: int,
    ) -> Tuple[Optional[StopReason], str]:
        """Check if voting should stop."""
        if not vote_counts:
            return None, ""

        # Sort by count
        sorted_hashes = sorted(vote_counts.keys(), key=lambda h: vote_counts[h], reverse=True)
        leader_hash = sorted_hashes[0]
        leader_count = vote_counts[leader_hash]

        # Check unanimous
        if len(vote_counts) == 1 and leader_count >= self.config.unanimous_threshold:
            return StopReason.UNANIMOUS, leader_hash

        # Check ahead-by-k
        if len(sorted_hashes) >= 2:
            second_count = vote_counts[sorted_hashes[1]]
            if leader_count - second_count >= self.config.k:
                return StopReason.AHEAD_BY_K, leader_hash
        elif leader_count >= self.config.k:
            # Only one option seen, and it has k votes
            return StopReason.AHEAD_BY_K, leader_hash

        return None, ""


# Convenience functions for common micro-decisions

def binary_vote(
    question: str,
    provider_callback: Callable,
    k: int = 3,
    max_rounds: int = 9,
    providers: Optional[List[str]] = None,
) -> Tuple[bool, float]:
    """
    Vote on a yes/no question.

    Returns:
        (answer: bool, confidence: float)
    """
    # Initialize mutable defaults
    providers = providers or ["codex", "gemini", "claude"]

    prompt = f"""{question}

Answer with ONLY "yes" or "no" (lowercase, no explanation)."""

    def parser(response: str) -> bool:
        text = response.strip().lower()
        if "yes" in text[:10]:
            return True
        if "no" in text[:10]:
            return False
        raise ValueError(f"Could not parse yes/no: {text[:50]}")

    config = ConsensusConfig(
        k=k,
        max_rounds=max_rounds,
        providers=providers,
        red_flags=RedFlagConfig(max_chars=100),
    )

    engine = ConsensusEngine(config, provider_callback)
    result = engine.vote(prompt, response_parser=parser)

    return result.winner if result.winner is not None else False, result.confidence


def select_from_options(
    question: str,
    options: List[str],
    provider_callback: Callable,
    k: int = 3,
    max_rounds: int = 12,
    providers: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Vote to select one option from a list.

    Returns:
        (selected_option: str, confidence: float)
    """
    # Initialize mutable defaults
    providers = providers or ["codex", "gemini", "claude"]

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    prompt = f"""{question}

Options:
{options_text}

Respond with ONLY the number (1, 2, 3, etc.) of your choice."""

    def parser(response: str) -> str:
        # Extract number
        match = re.search(r'\b(\d+)\b', response)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(options):
                return options[idx]
        raise ValueError(f"Could not parse option number: {response[:50]}")

    config = ConsensusConfig(
        k=k,
        max_rounds=max_rounds,
        providers=providers,
        red_flags=RedFlagConfig(max_chars=50),
    )

    engine = ConsensusEngine(config, provider_callback)
    result = engine.vote(prompt, response_parser=parser)

    return result.winner if result.winner else options[0], result.confidence


def rank_items(
    question: str,
    items: List[str],
    provider_callback: Callable,
    k: int = 2,
    max_rounds: int = 9,
    providers: Optional[List[str]] = None,
) -> Tuple[List[str], float]:
    """
    Vote to rank items by some criterion.

    Returns:
        (ranked_items: List[str], confidence: float)
    """
    # Initialize mutable defaults
    providers = providers or ["codex", "gemini"]

    items_text = "\n".join(f"- {item}" for item in items)
    prompt = f"""{question}

Items:
{items_text}

Respond with the items in ranked order (best first), one per line, exactly as written above."""

    def parser(response: str) -> List[str]:
        lines = [l.strip().lstrip("- ").lstrip("0123456789.").strip()
                 for l in response.strip().split("\n") if l.strip()]

        # Match back to original items
        ranked = []
        remaining = items.copy()
        for line in lines:
            line_lower = line.lower()
            for item in remaining:
                if item.lower() in line_lower or line_lower in item.lower():
                    ranked.append(item)
                    remaining.remove(item)
                    break

        # Add any missed items
        ranked.extend(remaining)
        return ranked

    def equivalence(a: List[str], b: List[str]) -> bool:
        # Same order
        return a == b

    config = ConsensusConfig(
        k=k,
        max_rounds=max_rounds,
        providers=providers,
        red_flags=RedFlagConfig(max_chars=500),
    )

    engine = ConsensusEngine(config, provider_callback)
    result = engine.vote(prompt, response_parser=parser, equivalence_fn=equivalence)

    return result.winner if result.winner else items, result.confidence
