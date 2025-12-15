"""
MAKER-style Error Correction Module

Implements key concepts from "Solving a Million-Step LLM Task With Zero Errors":
1. MAD (Maximal Agentic Decomposition) - Micro-step definitions
2. vote_step() - First-to-ahead-by-k voting for error correction
3. redflagger - Validation policies (discard, don't repair)
4. calibrate() - Estimate p and auto-set k for target success rate

The core insight: System-level error correction scales better than model intelligence.
"""

import json
import math
import time
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import Counter


# =============================================================================
# MICRO-STEP DEFINITIONS (MAD - Maximal Agentic Decomposition)
# =============================================================================

class StageType(Enum):
    """5-stage workflow stages."""
    ANALYZE = "analyze"
    HYPOTHESIZE = "hypothesize"
    IMPLEMENT = "implement"
    DEBUG = "debug"
    IMPROVE = "improve"


class MicroStepType(Enum):
    """
    Atomic micro-steps within each stage.
    Each step should be small enough that voting is meaningful.
    """
    # Analyze stage (S)
    S1_SPEC_EXTRACT = "s1_spec_extract"        # Extract 1 input/output spec
    S2_EDGE_CASE = "s2_edge_case"              # Generate 1 edge case
    S3_MRE = "s3_mre"                          # Create minimal reproducible example

    # Hypothesize stage (H)
    H1_ROOT_CAUSE = "h1_root_cause"            # Propose 1 root cause candidate
    H2_VERIFICATION = "h2_verification"        # Design 1 verification experiment

    # Implement stage (C)
    C1_MINIMAL_PATCH = "c1_minimal_patch"      # Generate 1 minimal patch for 1 file
    C2_COMPILE_CHECK = "c2_compile_check"      # Ensure patch passes compile/typecheck

    # Debug stage (D)
    D1_FAILURE_LABEL = "d1_failure_label"      # Classify failure type
    D2_NEXT_EXPERIMENT = "d2_next_experiment"  # Decide next single experiment

    # Improve stage (R)
    R1_REFACTOR = "r1_refactor"                # 1 refactoring + safety checklist
    R2_PERF = "r2_perf"                        # 1 perf improvement + measurement


@dataclass
class MicroStepSpec:
    """Specification for a micro-step."""
    step_type: MicroStepType
    stage: StageType
    description: str
    output_schema: Dict[str, Any]
    # Recommended k for voting (can be calibrated)
    default_k: int = 3
    # Whether this step has a tool-based oracle (test, compile, etc.)
    has_oracle: bool = False
    # Red-flag rules specific to this step type
    red_flag_rules: List[str] = field(default_factory=list)
    # Required MCP tools for this step
    required_tools: List[str] = field(default_factory=list)


# Registry of micro-step specifications
MICRO_STEP_SPECS: Dict[MicroStepType, MicroStepSpec] = {
    # Analyze steps
    MicroStepType.S1_SPEC_EXTRACT: MicroStepSpec(
        step_type=MicroStepType.S1_SPEC_EXTRACT,
        stage=StageType.ANALYZE,
        description="Extract exactly one input/output specification from given context",
        output_schema={
            "type": "object",
            "required": ["input_spec", "output_spec", "source_file"],
            "properties": {
                "input_spec": {"type": "string"},
                "output_spec": {"type": "string"},
                "source_file": {"type": "string"}
            }
        },
        default_k=2,
        has_oracle=False,
        red_flag_rules=["too_long", "missing_fields"],
        required_tools=["maestro_pack_context", "maestro_log_evidence"]
    ),
    MicroStepType.S2_EDGE_CASE: MicroStepSpec(
        step_type=MicroStepType.S2_EDGE_CASE,
        stage=StageType.ANALYZE,
        description="Generate exactly one edge case for the given spec",
        output_schema={
            "type": "object",
            "required": ["edge_case_input", "expected_behavior", "rationale"],
            "properties": {
                "edge_case_input": {"type": "string"},
                "expected_behavior": {"type": "string"},
                "rationale": {"type": "string"}
            }
        },
        default_k=2,
        has_oracle=False,
        red_flag_rules=["too_long", "missing_fields"],
        required_tools=["maestro_pack_context", "maestro_log_evidence"]
    ),
    MicroStepType.S3_MRE: MicroStepSpec(
        step_type=MicroStepType.S3_MRE,
        stage=StageType.ANALYZE,
        description="Create a minimal reproducible example that demonstrates the issue",
        output_schema={
            "type": "object",
            "required": ["mre_code", "steps_to_reproduce", "expected_vs_actual"],
            "properties": {
                "mre_code": {"type": "string"},
                "steps_to_reproduce": {"type": "array", "items": {"type": "string"}},
                "expected_vs_actual": {"type": "string"}
            }
        },
        default_k=3,
        has_oracle=True,  # Can run the MRE to verify
        red_flag_rules=["too_long", "missing_fields", "dangerous_code"],
        required_tools=["maestro_pack_context", "maestro_verify", "maestro_log_evidence"]
    ),

    # Hypothesize steps
    MicroStepType.H1_ROOT_CAUSE: MicroStepSpec(
        step_type=MicroStepType.H1_ROOT_CAUSE,
        stage=StageType.HYPOTHESIZE,
        description="Propose exactly one root cause hypothesis with file/line location",
        output_schema={
            "type": "object",
            "required": ["hypothesis", "suspected_file", "suspected_location", "confidence"],
            "properties": {
                "hypothesis": {"type": "string"},
                "suspected_file": {"type": "string"},
                "suspected_location": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        default_k=3,
        has_oracle=False,
        red_flag_rules=["too_long", "missing_fields", "hedging"],
        required_tools=["maestro_consult", "maestro_log_evidence"]
    ),
    MicroStepType.H2_VERIFICATION: MicroStepSpec(
        step_type=MicroStepType.H2_VERIFICATION,
        stage=StageType.HYPOTHESIZE,
        description="Design exactly one experiment to verify/falsify the hypothesis",
        output_schema={
            "type": "object",
            "required": ["experiment_type", "command", "expected_if_true", "expected_if_false"],
            "properties": {
                "experiment_type": {"type": "string", "enum": ["test", "log", "probe", "inspect"]},
                "command": {"type": "string"},
                "expected_if_true": {"type": "string"},
                "expected_if_false": {"type": "string"}
            }
        },
        default_k=2,
        has_oracle=True,  # Can run the experiment
        red_flag_rules=["too_long", "missing_fields", "dangerous_command"],
        required_tools=["maestro_verify", "maestro_log_evidence"]
    ),

    # Implement steps
    MicroStepType.C1_MINIMAL_PATCH: MicroStepSpec(
        step_type=MicroStepType.C1_MINIMAL_PATCH,
        stage=StageType.IMPLEMENT,
        description="Generate a minimal patch for exactly one file",
        output_schema={
            "type": "object",
            "required": ["target_file", "patch_diff", "change_summary"],
            "properties": {
                "target_file": {"type": "string"},
                "patch_diff": {"type": "string"},
                "change_summary": {"type": "string"}
            }
        },
        default_k=3,
        has_oracle=True,  # Can apply and test
        red_flag_rules=["too_long", "missing_fields", "multi_file", "invalid_diff"],
        required_tools=["maestro_apply_patch", "maestro_verify", "maestro_log_evidence"]
    ),
    MicroStepType.C2_COMPILE_CHECK: MicroStepSpec(
        step_type=MicroStepType.C2_COMPILE_CHECK,
        stage=StageType.IMPLEMENT,
        description="Verify patch passes compile/typecheck and fix if needed",
        output_schema={
            "type": "object",
            "required": ["passes_compile", "passes_typecheck", "fix_patch"],
            "properties": {
                "passes_compile": {"type": "boolean"},
                "passes_typecheck": {"type": "boolean"},
                "fix_patch": {"type": ["string", "null"]}
            }
        },
        default_k=2,
        has_oracle=True,  # Compiler is the oracle
        red_flag_rules=["missing_fields"],
        required_tools=["maestro_verify", "maestro_apply_patch"]
    ),

    # Debug steps
    MicroStepType.D1_FAILURE_LABEL: MicroStepSpec(
        step_type=MicroStepType.D1_FAILURE_LABEL,
        stage=StageType.DEBUG,
        description="Classify the failure into exactly one category",
        output_schema={
            "type": "object",
            "required": ["failure_type", "error_message", "stack_trace_summary"],
            "properties": {
                "failure_type": {"type": "string", "enum": [
                    "assertion_failure", "null_pointer", "type_error",
                    "timeout", "crash", "logic_error", "unknown"
                ]},
                "error_message": {"type": "string"},
                "stack_trace_summary": {"type": "string"}
            }
        },
        default_k=2,
        has_oracle=False,
        red_flag_rules=["missing_fields"],
        required_tools=["maestro_pack_context", "maestro_log_evidence"]
    ),
    MicroStepType.D2_NEXT_EXPERIMENT: MicroStepSpec(
        step_type=MicroStepType.D2_NEXT_EXPERIMENT,
        stage=StageType.DEBUG,
        description="Decide exactly one next experiment to run",
        output_schema={
            "type": "object",
            "required": ["experiment_command", "expected_insight", "abort_condition"],
            "properties": {
                "experiment_command": {"type": "string"},
                "expected_insight": {"type": "string"},
                "abort_condition": {"type": "string"}
            }
        },
        default_k=3,
        has_oracle=True,
        red_flag_rules=["missing_fields", "dangerous_command"],
        required_tools=["maestro_verify", "maestro_log_evidence"]
    ),

    # Improve steps
    MicroStepType.R1_REFACTOR: MicroStepSpec(
        step_type=MicroStepType.R1_REFACTOR,
        stage=StageType.IMPROVE,
        description="Propose exactly one refactoring with safety checklist",
        output_schema={
            "type": "object",
            "required": ["refactor_type", "target_file", "patch_diff", "safety_checks"],
            "properties": {
                "refactor_type": {"type": "string"},
                "target_file": {"type": "string"},
                "patch_diff": {"type": "string"},
                "safety_checks": {"type": "array", "items": {"type": "string"}}
            }
        },
        default_k=2,
        has_oracle=True,  # Tests are oracle
        red_flag_rules=["too_long", "missing_fields", "multi_file", "invalid_diff"],
        required_tools=["maestro_apply_patch", "maestro_verify", "maestro_log_evidence"]
    ),
    MicroStepType.R2_PERF: MicroStepSpec(
        step_type=MicroStepType.R2_PERF,
        stage=StageType.IMPROVE,
        description="Propose exactly one performance improvement with measurement",
        output_schema={
            "type": "object",
            "required": ["improvement_type", "target_file", "patch_diff", "measurement_command"],
            "properties": {
                "improvement_type": {"type": "string"},
                "target_file": {"type": "string"},
                "patch_diff": {"type": "string"},
                "measurement_command": {"type": "string"}
            }
        },
        default_k=2,
        has_oracle=True,
        red_flag_rules=["too_long", "missing_fields", "invalid_diff"],
        required_tools=["maestro_apply_patch", "maestro_verify", "maestro_log_evidence"]
    ),
}


# =============================================================================
# RED-FLAGGER (Validation - Discard, Don't Repair)
# =============================================================================

class RedFlagReason(Enum):
    """Reasons for red-flagging a candidate."""
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    MISSING_FIELDS = "missing_fields"
    INVALID_JSON = "invalid_json"
    INVALID_DIFF = "invalid_diff"
    HEDGING = "hedging"
    MULTI_FILE = "multi_file"
    DANGEROUS_CODE = "dangerous_code"
    DANGEROUS_COMMAND = "dangerous_command"
    FORBIDDEN_FILE = "forbidden_file"
    EXCESSIVE_CHANGES = "excessive_changes"
    CONTRADICTION = "contradiction"
    FORMAT_ERROR = "format_error"


@dataclass
class RedFlagResult:
    """Result of red-flag validation."""
    is_flagged: bool
    reasons: List[RedFlagReason] = field(default_factory=list)
    details: str = ""


@dataclass
class RedFlaggerConfig:
    """Configuration for red-flagger."""
    # Length limits
    max_chars: int = 10000
    min_chars: int = 10
    max_diff_lines: int = 200
    max_files_per_patch: int = 1

    # Hedging patterns (discard if present)
    hedging_patterns: List[str] = field(default_factory=lambda: [
        r"i'm not sure",
        r"i don't know",
        r"maybe",
        r"perhaps",
        r"it might be",
        r"could be",
        r"i think maybe",
        r"uncertain",
    ])

    # Dangerous patterns
    dangerous_code_patterns: List[str] = field(default_factory=lambda: [
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess\.call.*shell\s*=\s*True",
        r"os\.system",
    ])

    dangerous_command_patterns: List[str] = field(default_factory=lambda: [
        r"rm\s+-rf",
        r"rm\s+-r\s+/",
        r"mkfs",
        r"dd\s+if=",
        r">\s*/dev/",
        r"chmod\s+777",
        r"curl.*\|\s*bash",
        r"wget.*\|\s*sh",
    ])

    # Forbidden files
    forbidden_file_patterns: List[str] = field(default_factory=lambda: [
        r"\.env$",
        r"\.env\.",
        r"secrets?\.json",
        r"credentials?\.json",
        r"\.pem$",
        r"\.key$",
        r"id_rsa",
        r"\.lock$",
        r"package-lock\.json",
        r"yarn\.lock",
        r"poetry\.lock",
    ])


class RedFlagger:
    """
    MAKER-style red-flagger.

    Key insight from paper: "Format errors signal reasoning errors"
    - Don't try to repair malformed responses
    - Discard and resample instead
    - This reduces correlated errors, not just average errors
    """

    def __init__(self, config: Optional[RedFlaggerConfig] = None):
        self.config = config or RedFlaggerConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._hedging_re = [
            re.compile(p, re.IGNORECASE) for p in self.config.hedging_patterns
        ]
        self._dangerous_code_re = [
            re.compile(p, re.IGNORECASE) for p in self.config.dangerous_code_patterns
        ]
        self._dangerous_cmd_re = [
            re.compile(p, re.IGNORECASE) for p in self.config.dangerous_command_patterns
        ]
        self._forbidden_file_re = [
            re.compile(p, re.IGNORECASE) for p in self.config.forbidden_file_patterns
        ]

    def validate(
        self,
        content: str,
        step_type: Optional[MicroStepType] = None,
        rules: Optional[List[str]] = None
    ) -> RedFlagResult:
        """
        Validate content and return red-flag result.

        Args:
            content: The LLM response content
            step_type: Optional micro-step type for step-specific rules
            rules: Optional list of rule names to apply

        Returns:
            RedFlagResult indicating if content should be discarded
        """
        reasons = []
        details = []

        # Get rules to apply
        if rules is None and step_type:
            spec = MICRO_STEP_SPECS.get(step_type)
            rules = spec.red_flag_rules if spec else []
        rules = rules or []

        # Length checks (always apply)
        if len(content) > self.config.max_chars:
            reasons.append(RedFlagReason.TOO_LONG)
            details.append(f"Content too long: {len(content)} > {self.config.max_chars}")

        if len(content) < self.config.min_chars:
            reasons.append(RedFlagReason.TOO_SHORT)
            details.append(f"Content too short: {len(content)} < {self.config.min_chars}")

        # Rule-specific checks
        if "hedging" in rules:
            for pattern in self._hedging_re:
                if pattern.search(content):
                    reasons.append(RedFlagReason.HEDGING)
                    details.append(f"Hedging detected: {pattern.pattern}")
                    break

        if "dangerous_code" in rules:
            for pattern in self._dangerous_code_re:
                if pattern.search(content):
                    reasons.append(RedFlagReason.DANGEROUS_CODE)
                    details.append(f"Dangerous code pattern: {pattern.pattern}")
                    break

        if "dangerous_command" in rules:
            for pattern in self._dangerous_cmd_re:
                if pattern.search(content):
                    reasons.append(RedFlagReason.DANGEROUS_COMMAND)
                    details.append(f"Dangerous command pattern: {pattern.pattern}")
                    break

        if "invalid_diff" in rules:
            if not self._is_valid_diff(content):
                reasons.append(RedFlagReason.INVALID_DIFF)
                details.append("Invalid diff format")

        if "multi_file" in rules:
            if self._count_diff_files(content) > self.config.max_files_per_patch:
                reasons.append(RedFlagReason.MULTI_FILE)
                details.append(f"Patch affects too many files")

        if "missing_fields" in rules and step_type:
            spec = MICRO_STEP_SPECS.get(step_type)
            if spec:
                missing = self._check_required_fields(content, spec.output_schema)
                if missing:
                    reasons.append(RedFlagReason.MISSING_FIELDS)
                    details.append(f"Missing required fields: {missing}")

        # Check forbidden files if content looks like a diff or file reference
        for pattern in self._forbidden_file_re:
            if pattern.search(content):
                reasons.append(RedFlagReason.FORBIDDEN_FILE)
                details.append(f"References forbidden file pattern: {pattern.pattern}")
                break

        return RedFlagResult(
            is_flagged=len(reasons) > 0,
            reasons=reasons,
            details="; ".join(details)
        )

    def _is_valid_diff(self, content: str) -> bool:
        """Check if content is a valid unified diff."""
        # Look for diff markers
        has_minus = "---" in content or content.strip().startswith("-")
        has_plus = "+++" in content or "\n+" in content
        has_hunk = "@@" in content

        # Basic validity: has some diff-like structure
        return (has_minus and has_plus) or has_hunk

    def _count_diff_files(self, content: str) -> int:
        """Count number of files in a diff."""
        # Count --- markers (each file starts with ---)
        return content.count("\n---") + (1 if content.startswith("---") else 0)

    def _check_required_fields(self, content: str, schema: Dict) -> List[str]:
        """Check if JSON content has required fields."""
        try:
            # Try to extract JSON from content
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if not json_match:
                return schema.get("required", [])

            data = json.loads(json_match.group())
            required = schema.get("required", [])
            return [f for f in required if f not in data]
        except (json.JSONDecodeError, Exception):
            return schema.get("required", [])


# =============================================================================
# VOTE_STEP (First-to-ahead-by-k Voting)
# =============================================================================

@dataclass
class VoteCandidate:
    """A candidate in the voting process."""
    id: str
    content: str
    provider: str
    raw_response: str
    votes: int = 0
    red_flag_result: Optional[RedFlagResult] = None
    normalized_key: Optional[str] = None  # For equivalence comparison


@dataclass
class VoteResult:
    """Result of a voting process."""
    winner: Optional[VoteCandidate]
    total_rounds: int
    total_samples: int
    red_flagged_count: int
    vote_distribution: Dict[str, int]
    all_candidates: List[VoteCandidate]
    converged: bool
    final_margin: int


class VoteStep:
    """
    MAKER-style first-to-ahead-by-k voting.

    Algorithm:
    1. Sample candidates from LLM(s)
    2. Red-flag invalid responses (discard)
    3. Normalize valid responses for equivalence comparison
    4. Count votes for each unique answer
    5. If leader is ahead by k votes, return winner
    6. Otherwise, continue sampling

    Key insight: "first-to-ahead-by-k" is more stable than "first-to-k"
    """

    def __init__(
        self,
        k: int = 3,
        max_rounds: int = 20,
        red_flagger: Optional[RedFlagger] = None,
        equivalence_fn: Optional[Callable[[str, str], bool]] = None,
        normalize_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Args:
            k: Required margin to declare winner
            max_rounds: Maximum sampling rounds before giving up
            red_flagger: RedFlagger instance for validation
            equivalence_fn: Custom function to check if two responses are equivalent
            normalize_fn: Custom function to normalize response for comparison
        """
        self.k = k
        self.max_rounds = max_rounds
        self.red_flagger = red_flagger or RedFlagger()
        self.equivalence_fn = equivalence_fn
        self.normalize_fn = normalize_fn or self._default_normalize

    def _default_normalize(self, content: str) -> str:
        """Default normalization: strip whitespace, lowercase."""
        return content.strip().lower()

    def _generate_candidate_id(self, content: str) -> str:
        """Generate unique ID for candidate based on content hash."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def vote(
        self,
        sample_fn: Callable[[], Tuple[str, str, str]],
        step_type: Optional[MicroStepType] = None,
        oracle_fn: Optional[Callable[[str], bool]] = None,
    ) -> VoteResult:
        """
        Run first-to-ahead-by-k voting.

        Args:
            sample_fn: Function that returns (content, provider, raw_response)
            step_type: Optional micro-step type for red-flag rules
            oracle_fn: Optional oracle to validate candidates (e.g., test runner)

        Returns:
            VoteResult with winner and statistics
        """
        candidates: Dict[str, VoteCandidate] = {}  # normalized_key -> candidate
        vote_counts: Counter = Counter()
        red_flagged_count = 0
        round_num = 0

        while round_num < self.max_rounds:
            round_num += 1

            # Sample a new candidate
            try:
                content, provider, raw_response = sample_fn()
            except Exception as e:
                continue  # Sampling failed, try again

            # Red-flag check
            red_flag_result = self.red_flagger.validate(content, step_type)
            if red_flag_result.is_flagged:
                red_flagged_count += 1
                continue  # Discard, don't repair

            # Oracle check (if available)
            if oracle_fn:
                try:
                    if not oracle_fn(content):
                        continue  # Failed oracle, discard
                except Exception:
                    continue

            # Normalize for comparison
            normalized_key = self.normalize_fn(content)

            # Check equivalence with existing candidates
            matched_key = None
            if self.equivalence_fn:
                for existing_key, existing_candidate in candidates.items():
                    if self.equivalence_fn(content, existing_candidate.content):
                        matched_key = existing_key
                        break
            else:
                matched_key = normalized_key if normalized_key in candidates else None

            # Add vote
            if matched_key:
                candidates[matched_key].votes += 1
                vote_counts[matched_key] += 1
            else:
                candidate = VoteCandidate(
                    id=self._generate_candidate_id(content),
                    content=content,
                    provider=provider,
                    raw_response=raw_response,
                    votes=1,
                    red_flag_result=red_flag_result,
                    normalized_key=normalized_key,
                )
                candidates[normalized_key] = candidate
                vote_counts[normalized_key] = 1

            # Check if we have a winner (first-to-ahead-by-k)
            if len(vote_counts) >= 1:
                sorted_counts = vote_counts.most_common()
                leader_key, leader_votes = sorted_counts[0]

                if len(sorted_counts) == 1:
                    # Only one candidate, needs k votes
                    if leader_votes >= self.k:
                        return VoteResult(
                            winner=candidates[leader_key],
                            total_rounds=round_num,
                            total_samples=sum(vote_counts.values()) + red_flagged_count,
                            red_flagged_count=red_flagged_count,
                            vote_distribution=dict(vote_counts),
                            all_candidates=list(candidates.values()),
                            converged=True,
                            final_margin=leader_votes,
                        )
                else:
                    # Multiple candidates, check margin
                    second_votes = sorted_counts[1][1]
                    margin = leader_votes - second_votes

                    if margin >= self.k:
                        return VoteResult(
                            winner=candidates[leader_key],
                            total_rounds=round_num,
                            total_samples=sum(vote_counts.values()) + red_flagged_count,
                            red_flagged_count=red_flagged_count,
                            vote_distribution=dict(vote_counts),
                            all_candidates=list(candidates.values()),
                            converged=True,
                            final_margin=margin,
                        )

        # Max rounds reached without convergence
        if vote_counts:
            leader_key, leader_votes = vote_counts.most_common(1)[0]
            second_votes = vote_counts.most_common(2)[1][1] if len(vote_counts) > 1 else 0

            return VoteResult(
                winner=candidates[leader_key],
                total_rounds=round_num,
                total_samples=sum(vote_counts.values()) + red_flagged_count,
                red_flagged_count=red_flagged_count,
                vote_distribution=dict(vote_counts),
                all_candidates=list(candidates.values()),
                converged=False,
                final_margin=leader_votes - second_votes,
            )

        return VoteResult(
            winner=None,
            total_rounds=round_num,
            total_samples=red_flagged_count,
            red_flagged_count=red_flagged_count,
            vote_distribution={},
            all_candidates=[],
            converged=False,
            final_margin=0,
        )


# =============================================================================
# CALIBRATE (p Estimation and k Auto-setting)
# =============================================================================

@dataclass
class CalibrationResult:
    """Result of calibration process."""
    estimated_p: float  # Step accuracy
    estimated_v: float  # Valid rate (non-red-flagged)
    recommended_k: int
    target_success_rate: float
    total_steps_estimate: int
    expected_cost_multiplier: float
    calibration_samples: int
    per_step_type_stats: Dict[str, Dict[str, float]]


class Calibrator:
    """
    Calibrate voting parameters based on empirical sampling.

    From the paper:
    - Estimate p (step accuracy) from representative samples
    - Calculate required k for target overall success rate
    - Account for valid rate v (non-red-flagged samples)
    """

    def __init__(self, red_flagger: Optional[RedFlagger] = None):
        self.red_flagger = red_flagger or RedFlagger()

    def estimate_step_accuracy(
        self,
        sample_fn: Callable[[], Tuple[str, str, str]],
        oracle_fn: Callable[[str], bool],
        step_type: Optional[MicroStepType] = None,
        num_samples: int = 20,
    ) -> Tuple[float, float]:
        """
        Estimate p (accuracy) and v (valid rate) from samples.

        Args:
            sample_fn: Function that returns (content, provider, raw_response)
            oracle_fn: Function that returns True if content is correct
            step_type: Optional micro-step type for red-flag rules
            num_samples: Number of samples to take

        Returns:
            (p: accuracy among valid, v: valid rate)
        """
        valid_count = 0
        correct_count = 0

        for _ in range(num_samples):
            try:
                content, _, _ = sample_fn()
            except Exception:
                continue

            # Check red-flag
            red_flag_result = self.red_flagger.validate(content, step_type)
            if red_flag_result.is_flagged:
                continue

            valid_count += 1

            # Check oracle
            try:
                if oracle_fn(content):
                    correct_count += 1
            except Exception:
                pass

        v = valid_count / num_samples if num_samples > 0 else 0
        p = correct_count / valid_count if valid_count > 0 else 0

        return p, v

    def calculate_k(
        self,
        p: float,
        total_steps: int,
        target_success_rate: float = 0.99,
    ) -> int:
        """
        Calculate required k for target overall success rate.

        From paper's analysis:
        - Step decision accuracy with k margin: approx 1 - 2 * (1-p)^k / p^k
        - Overall success = (step_accuracy)^total_steps

        We solve for k that achieves target.
        """
        if p >= 1.0:
            return 1
        if p <= 0.5:
            return 10  # Very unreliable, need high k

        # Binary search for k
        for k in range(1, 20):
            # Approximate step decision accuracy
            # Using simplified model: p_step ≈ 1 - ((1-p)/p)^k
            ratio = (1 - p) / p
            step_accuracy = 1 - (ratio ** k)

            # Overall success
            overall = step_accuracy ** total_steps

            if overall >= target_success_rate:
                return k

        return 10  # Default max

    def calibrate(
        self,
        sample_fn: Callable[[], Tuple[str, str, str]],
        oracle_fn: Callable[[str], bool],
        total_steps: int,
        target_success_rate: float = 0.99,
        step_type: Optional[MicroStepType] = None,
        num_samples: int = 20,
    ) -> CalibrationResult:
        """
        Full calibration to determine optimal k.

        Args:
            sample_fn: Sample generator
            oracle_fn: Correctness checker
            total_steps: Expected total steps in workflow
            target_success_rate: Desired overall success rate
            step_type: Optional step type for rules
            num_samples: Samples for estimation

        Returns:
            CalibrationResult with recommended k and statistics
        """
        p, v = self.estimate_step_accuracy(
            sample_fn, oracle_fn, step_type, num_samples
        )

        k = self.calculate_k(p, total_steps, target_success_rate)

        # Estimate cost multiplier (expected samples per step)
        # Rough estimate: k / (p * v) for convergence
        expected_cost = k / (p * v) if (p * v) > 0 else float('inf')

        return CalibrationResult(
            estimated_p=p,
            estimated_v=v,
            recommended_k=k,
            target_success_rate=target_success_rate,
            total_steps_estimate=total_steps,
            expected_cost_multiplier=min(expected_cost, 100),
            calibration_samples=num_samples,
            per_step_type_stats={},
        )

    def calibrate_all_steps(
        self,
        sample_fn_factory: Callable[[MicroStepType], Callable[[], Tuple[str, str, str]]],
        oracle_fn_factory: Callable[[MicroStepType], Callable[[str], bool]],
        steps_per_type: Dict[MicroStepType, int],
        target_success_rate: float = 0.99,
        num_samples: int = 10,
    ) -> CalibrationResult:
        """
        Calibrate k for each step type and compute overall.

        Args:
            sample_fn_factory: Returns sample_fn for each step type
            oracle_fn_factory: Returns oracle_fn for each step type
            steps_per_type: Expected count of each step type
            target_success_rate: Desired overall success
            num_samples: Samples per step type

        Returns:
            CalibrationResult with per-type stats
        """
        per_type_stats = {}
        total_steps = sum(steps_per_type.values())

        for step_type, count in steps_per_type.items():
            try:
                sample_fn = sample_fn_factory(step_type)
                oracle_fn = oracle_fn_factory(step_type)

                p, v = self.estimate_step_accuracy(
                    sample_fn, oracle_fn, step_type, num_samples
                )

                per_type_stats[step_type.value] = {
                    "p": p,
                    "v": v,
                    "count": count,
                    "recommended_k": self.calculate_k(
                        p, count, target_success_rate ** (count / total_steps)
                    )
                }
            except Exception as e:
                per_type_stats[step_type.value] = {
                    "p": 0.5,
                    "v": 0.5,
                    "count": count,
                    "recommended_k": 5,
                    "error": str(e)
                }

        # Aggregate statistics
        weighted_p = sum(
            stats["p"] * stats["count"]
            for stats in per_type_stats.values()
        ) / total_steps if total_steps > 0 else 0.5

        weighted_v = sum(
            stats["v"] * stats["count"]
            for stats in per_type_stats.values()
        ) / total_steps if total_steps > 0 else 0.5

        overall_k = self.calculate_k(weighted_p, total_steps, target_success_rate)

        return CalibrationResult(
            estimated_p=weighted_p,
            estimated_v=weighted_v,
            recommended_k=overall_k,
            target_success_rate=target_success_rate,
            total_steps_estimate=total_steps,
            expected_cost_multiplier=overall_k / (weighted_p * weighted_v) if (weighted_p * weighted_v) > 0 else 100,
            calibration_samples=num_samples * len(steps_per_type),
            per_step_type_stats=per_type_stats,
        )


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def get_tools_for_step(step_type: MicroStepType) -> List[str]:
    """Get required MCP tools for a micro-step type."""
    spec = MICRO_STEP_SPECS.get(step_type)
    return spec.required_tools if spec else []


def get_tools_for_stage(stage: StageType) -> Set[str]:
    """Get all required MCP tools for a stage (union of all micro-steps)."""
    tools = set()
    for spec in MICRO_STEP_SPECS.values():
        if spec.stage == stage:
            tools.update(spec.required_tools)
    return tools


def get_all_micro_steps_for_stage(stage: StageType) -> List[MicroStepSpec]:
    """Get all micro-step specs for a stage."""
    return [
        spec for spec in MICRO_STEP_SPECS.values()
        if spec.stage == stage
    ]


def get_default_k_for_step(step_type: MicroStepType) -> int:
    """Get default k value for a micro-step type."""
    spec = MICRO_STEP_SPECS.get(step_type)
    return spec.default_k if spec else 3


def step_has_oracle(step_type: MicroStepType) -> bool:
    """Check if a micro-step type has a tool-based oracle."""
    spec = MICRO_STEP_SPECS.get(step_type)
    return spec.has_oracle if spec else False
