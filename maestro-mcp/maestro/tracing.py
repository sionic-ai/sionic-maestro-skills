"""
Tracing and Metrics for Maestro Skills MCP.

Provides observability into:
- Workflow execution
- Provider calls (latency, success rate)
- Selection decisions (including red-flag statistics)
- Consensus voting (MAKER-style)
- Evidence chain (step-by-step reasoning trail)
- Verification results
- Overhead measurements (aligned with paper metrics)

Paper-aligned metrics:
- Coordination overhead O%
- Efficiency Ec = Success / Overhead
- Error amplification Ae
- Message density c

Evidence logging (from tech spec):
- Each step produces structured evidence
- Evidence is stored as JSONL for analysis
- "Why this decision" links to evidence, not LLM prose
"""

import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("maestro.tracing")


class EvidenceType(Enum):
    """Types of evidence that can be logged."""
    OBSERVATION = "observation"       # Facts observed from code/logs
    HYPOTHESIS = "hypothesis"         # Proposed explanation
    VERIFICATION = "verification"     # Test/lint result
    DECISION = "decision"             # Choice made with rationale
    RED_FLAG = "red_flag"             # Rejected candidate/response
    CONSENSUS = "consensus"           # Voting result
    PATCH = "patch"                   # Code change applied
    ROLLBACK = "rollback"             # Change reverted


@dataclass
class Evidence:
    """
    A single piece of evidence in the reasoning chain.

    Evidence-first approach: Every decision should link to concrete evidence,
    not just LLM rationale. This enables:
    - Post-hoc analysis of why decisions were made
    - Identification of reasoning failures
    - Learning from successful patterns
    """
    id: str                          # Unique ID (e.g., "E001")
    timestamp: str
    evidence_type: EvidenceType
    stage: str                       # Which workflow stage
    content: Dict[str, Any]          # The actual evidence
    source: str                      # Where it came from (file, test, provider, etc.)
    confidence: float = 1.0          # 0.0-1.0, how reliable is this evidence
    links: List[str] = field(default_factory=list)  # IDs of related evidence

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_type"] = self.evidence_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class TraceEntry:
    """A single trace entry."""
    timestamp: str
    event_type: str  # "provider_call", "stage_complete", "selection", "workflow_start", "workflow_end", "evidence", "verification", "consensus"
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ProviderCallTrace:
    """Trace of a single provider call."""
    timestamp: str
    provider: str
    model: str
    prompt_hash: str  # Hash of prompt (privacy)
    prompt_chars: int
    response_chars: int
    elapsed_ms: float
    success: bool
    stage: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageTrace:
    """Trace of a workflow stage."""
    timestamp: str
    stage: str
    success: bool
    consults_used: int
    elapsed_ms: float
    output_summary: str  # Brief summary, not full output
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionTrace:
    """Trace of a selection decision."""
    timestamp: str
    mode: str
    num_candidates: int
    winner_id: str
    winner_provider: str
    scores: Dict[str, float]
    rationale: str
    had_test_signals: bool
    had_lint_signals: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationTrace:
    """Trace of a verification run (tests/lint)."""
    timestamp: str
    verification_type: str  # "unit_test", "lint", "type_check", etc.
    command: str
    passed: bool
    exit_code: int
    duration_ms: float
    output_preview: str  # First 500 chars of output
    stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsensusTrace:
    """Trace of a consensus voting session (MAKER-style)."""
    timestamp: str
    question_hash: str  # Hash of question for privacy
    winner: Optional[str]
    winner_hash: str
    votes_for_winner: int
    total_valid_votes: int
    total_votes: int
    red_flagged_count: int
    stop_reason: str
    confidence: float
    rounds: int
    providers_used: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceStore:
    """
    Stores and manages traces for a workflow session.

    Features:
    - Append-only JSONL file storage
    - In-memory recent traces
    - Evidence chain tracking
    - Verification result tracking
    - Consensus voting tracking
    - Red-flag statistics
    - Metrics computation
    """

    def __init__(
        self,
        trace_dir: str = ".maestro-traces",
        session_id: Optional[str] = None,
        max_memory_entries: int = 1000,
    ):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_file = self.trace_dir / f"trace_{self.session_id}.jsonl"
        self.evidence_file = self.trace_dir / f"evidence_{self.session_id}.jsonl"

        self.entries: List[TraceEntry] = []
        self.provider_calls: List[ProviderCallTrace] = []
        self.stage_traces: List[StageTrace] = []
        self.selection_traces: List[SelectionTrace] = []
        self.verification_traces: List[VerificationTrace] = []
        self.consensus_traces: List[ConsensusTrace] = []
        self.evidence_chain: List[Evidence] = []

        self.max_memory_entries = max_memory_entries
        self._workflow_start_time: Optional[datetime] = None
        self._evidence_counter: int = 0

        # Red-flag statistics
        self.red_flag_stats = {
            "total_flagged": 0,
            "by_reason": defaultdict(int),
            "by_provider": defaultdict(int),
        }

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _hash_prompt(self, prompt: str) -> str:
        """Hash prompt for privacy while allowing dedup detection."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _append(self, entry: TraceEntry) -> None:
        """Append entry to memory and file."""
        self.entries.append(entry)

        # Write to file
        try:
            with open(self.trace_file, 'a') as f:
                f.write(entry.to_json() + '\n')
        except Exception as e:
            logger.warning(f"Failed to write trace: {e}")

        # Trim memory if needed
        if len(self.entries) > self.max_memory_entries:
            self.entries = self.entries[-self.max_memory_entries:]

    def log_workflow_start(self, task: str, config: Dict[str, Any]) -> None:
        """Log workflow start."""
        self._workflow_start_time = datetime.now()
        entry = TraceEntry(
            timestamp=self._now(),
            event_type="workflow_start",
            data={
                "task_hash": self._hash_prompt(task),
                "task_preview": task[:100],
                "config": config,
            },
        )
        self._append(entry)

    def log_workflow_end(self, success: bool, summary: Dict[str, Any]) -> None:
        """Log workflow end."""
        elapsed_ms = 0.0
        if self._workflow_start_time:
            elapsed_ms = (datetime.now() - self._workflow_start_time).total_seconds() * 1000

        entry = TraceEntry(
            timestamp=self._now(),
            event_type="workflow_end",
            data={
                "success": success,
                "total_elapsed_ms": elapsed_ms,
                "total_consults": len(self.provider_calls),
                "total_stages": len(self.stage_traces),
                **summary,
            },
        )
        self._append(entry)

    def log_provider_call(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        elapsed_ms: float,
        success: bool,
        stage: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a provider call."""
        trace = ProviderCallTrace(
            timestamp=self._now(),
            provider=provider,
            model=model,
            prompt_hash=self._hash_prompt(prompt),
            prompt_chars=len(prompt),
            response_chars=len(response),
            elapsed_ms=elapsed_ms,
            success=success,
            stage=stage,
            error=error,
        )
        self.provider_calls.append(trace)

        entry = TraceEntry(
            timestamp=trace.timestamp,
            event_type="provider_call",
            data=trace.to_dict(),
        )
        self._append(entry)

    def log_stage(self, stage_result: "StageResult") -> None:
        """Log a stage completion."""
        from .workflow import StageResult

        trace = StageTrace(
            timestamp=self._now(),
            stage=stage_result.stage.value,
            success=stage_result.success,
            consults_used=stage_result.consults_used,
            elapsed_ms=stage_result.elapsed_ms,
            output_summary=str(stage_result.output)[:200],
            error=stage_result.error,
        )
        self.stage_traces.append(trace)

        entry = TraceEntry(
            timestamp=trace.timestamp,
            event_type="stage_complete",
            data=trace.to_dict(),
        )
        self._append(entry)

    def log_selection(self, result: "SelectionResult") -> None:
        """Log a selection decision."""
        from .selection import SelectionResult

        trace = SelectionTrace(
            timestamp=self._now(),
            mode=result.mode_used.value,
            num_candidates=len(result.all_candidates),
            winner_id=result.winner_id,
            winner_provider=result.winner.provider,
            scores=result.scores,
            rationale=result.rationale[:200],
            had_test_signals=any(c.test_passed is not None for c in result.all_candidates),
            had_lint_signals=any(c.lint_score is not None for c in result.all_candidates),
        )
        self.selection_traces.append(trace)

        entry = TraceEntry(
            timestamp=trace.timestamp,
            event_type="selection",
            data=trace.to_dict(),
        )
        self._append(entry)

    def get_recent_entries(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get recent trace entries."""
        return [e.to_dict() for e in self.entries[-n:]]

    # =========================================================================
    # Evidence Chain Logging (New)
    # =========================================================================

    def log_evidence(
        self,
        evidence_type: EvidenceType,
        stage: str,
        content: Dict[str, Any],
        source: str,
        confidence: float = 1.0,
        links: Optional[List[str]] = None,
    ) -> str:
        """
        Log a piece of evidence to the chain.

        Returns:
            The evidence ID for linking to other evidence/decisions.
        """
        self._evidence_counter += 1
        evidence_id = f"E{self._evidence_counter:04d}"

        evidence = Evidence(
            id=evidence_id,
            timestamp=self._now(),
            evidence_type=evidence_type,
            stage=stage,
            content=content,
            source=source,
            confidence=confidence,
            links=links or [],
        )
        self.evidence_chain.append(evidence)

        # Write to evidence file
        try:
            with open(self.evidence_file, 'a') as f:
                f.write(evidence.to_json() + '\n')
        except Exception as e:
            logger.warning(f"Failed to write evidence: {e}")

        # Also log as trace entry
        entry = TraceEntry(
            timestamp=evidence.timestamp,
            event_type="evidence",
            data=evidence.to_dict(),
        )
        self._append(entry)

        return evidence_id

    def log_observation(
        self,
        stage: str,
        observation: str,
        source: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an observation (fact) as evidence."""
        return self.log_evidence(
            evidence_type=EvidenceType.OBSERVATION,
            stage=stage,
            content={"observation": observation, **(details or {})},
            source=source,
        )

    def log_hypothesis(
        self,
        stage: str,
        hypothesis: str,
        confidence: float,
        evidence_ids: List[str],
        verification_plan: Optional[str] = None,
    ) -> str:
        """Log a hypothesis as evidence."""
        return self.log_evidence(
            evidence_type=EvidenceType.HYPOTHESIS,
            stage=stage,
            content={
                "hypothesis": hypothesis,
                "verification_plan": verification_plan,
            },
            source="reasoning",
            confidence=confidence,
            links=evidence_ids,
        )

    def log_decision(
        self,
        stage: str,
        decision: str,
        rationale: str,
        evidence_ids: List[str],
        alternatives_considered: Optional[List[str]] = None,
    ) -> str:
        """Log a decision with its supporting evidence."""
        return self.log_evidence(
            evidence_type=EvidenceType.DECISION,
            stage=stage,
            content={
                "decision": decision,
                "rationale": rationale,
                "alternatives": alternatives_considered or [],
            },
            source="orchestrator",
            links=evidence_ids,
        )

    # =========================================================================
    # Verification Logging (New)
    # =========================================================================

    def log_verification(
        self,
        verification_type: str,
        command: str,
        passed: bool,
        exit_code: int,
        duration_ms: float,
        output: str,
        stage: Optional[str] = None,
    ) -> str:
        """
        Log a verification result (test, lint, type-check).

        Returns:
            Evidence ID for the verification result.
        """
        trace = VerificationTrace(
            timestamp=self._now(),
            verification_type=verification_type,
            command=command,
            passed=passed,
            exit_code=exit_code,
            duration_ms=duration_ms,
            output_preview=output[:500] if output else "",
            stage=stage,
        )
        self.verification_traces.append(trace)

        entry = TraceEntry(
            timestamp=trace.timestamp,
            event_type="verification",
            data=trace.to_dict(),
        )
        self._append(entry)

        # Also log as evidence
        return self.log_evidence(
            evidence_type=EvidenceType.VERIFICATION,
            stage=stage or "unknown",
            content={
                "type": verification_type,
                "command": command,
                "passed": passed,
                "exit_code": exit_code,
            },
            source=f"verification:{verification_type}",
            confidence=1.0,  # Verification results are deterministic
        )

    # =========================================================================
    # Consensus Voting Logging (New - MAKER style)
    # =========================================================================

    def log_consensus(
        self,
        question: str,
        result: "ConsensusResult",
        providers_used: List[str],
    ) -> str:
        """
        Log a consensus voting session result.

        Returns:
            Evidence ID for the consensus result.
        """
        # Count red-flagged votes
        red_flagged = sum(1 for v in result.vote_trace if v.status.value == "red_flagged")

        trace = ConsensusTrace(
            timestamp=self._now(),
            question_hash=self._hash_prompt(question),
            winner=str(result.winner)[:200] if result.winner else None,
            winner_hash=result.winner_hash,
            votes_for_winner=result.votes_for_winner,
            total_valid_votes=result.total_valid_votes,
            total_votes=result.total_votes,
            red_flagged_count=red_flagged,
            stop_reason=result.stop_reason.value,
            confidence=result.confidence,
            rounds=result.metadata.get("rounds", 0),
            providers_used=providers_used,
        )
        self.consensus_traces.append(trace)

        entry = TraceEntry(
            timestamp=trace.timestamp,
            event_type="consensus",
            data=trace.to_dict(),
        )
        self._append(entry)

        # Also log as evidence
        return self.log_evidence(
            evidence_type=EvidenceType.CONSENSUS,
            stage="consensus",
            content={
                "winner_hash": result.winner_hash,
                "confidence": result.confidence,
                "votes": f"{result.votes_for_winner}/{result.total_valid_votes}",
                "stop_reason": result.stop_reason.value,
            },
            source="consensus_voting",
            confidence=result.confidence,
        )

    # =========================================================================
    # Red-Flag Logging (New - MAKER style)
    # =========================================================================

    def log_red_flag(
        self,
        candidate_id: str,
        provider: str,
        reason: str,
        content_preview: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> str:
        """Log a red-flagged candidate."""
        # Update statistics
        self.red_flag_stats["total_flagged"] += 1
        self.red_flag_stats["by_reason"][reason] += 1
        self.red_flag_stats["by_provider"][provider] += 1

        return self.log_evidence(
            evidence_type=EvidenceType.RED_FLAG,
            stage=stage or "unknown",
            content={
                "candidate_id": candidate_id,
                "provider": provider,
                "reason": reason,
                "content_preview": content_preview[:200] if content_preview else None,
            },
            source=f"red_flag:{provider}",
            confidence=1.0,
        )

    # =========================================================================
    # Patch/Rollback Logging (New)
    # =========================================================================

    def log_patch(
        self,
        files_changed: List[str],
        files_created: List[str],
        stage: str,
        hypothesis_id: Optional[str] = None,
        backup_dir: Optional[str] = None,
    ) -> str:
        """Log a patch application."""
        return self.log_evidence(
            evidence_type=EvidenceType.PATCH,
            stage=stage,
            content={
                "files_changed": files_changed,
                "files_created": files_created,
                "backup_dir": backup_dir,
            },
            source="workspace",
            links=[hypothesis_id] if hypothesis_id else [],
        )

    def log_rollback(
        self,
        files_restored: List[str],
        reason: str,
        stage: str,
        patch_id: Optional[str] = None,
    ) -> str:
        """Log a rollback operation."""
        return self.log_evidence(
            evidence_type=EvidenceType.ROLLBACK,
            stage=stage,
            content={
                "files_restored": files_restored,
                "reason": reason,
            },
            source="workspace",
            links=[patch_id] if patch_id else [],
        )

    # =========================================================================
    # Evidence Chain Queries (New)
    # =========================================================================

    def get_evidence_chain(
        self,
        evidence_type: Optional[EvidenceType] = None,
        stage: Optional[str] = None,
    ) -> List[Evidence]:
        """Get evidence chain filtered by type and/or stage."""
        chain = self.evidence_chain

        if evidence_type:
            chain = [e for e in chain if e.evidence_type == evidence_type]

        if stage:
            chain = [e for e in chain if e.stage == stage]

        return chain

    def get_evidence_by_id(self, evidence_id: str) -> Optional[Evidence]:
        """Get a specific evidence by ID."""
        for e in self.evidence_chain:
            if e.id == evidence_id:
                return e
        return None

    def get_linked_evidence(self, evidence_id: str) -> List[Evidence]:
        """Get all evidence linked to a specific evidence ID."""
        result = []
        for e in self.evidence_chain:
            if evidence_id in e.links:
                result.append(e)
        return result

    def get_red_flag_summary(self) -> Dict[str, Any]:
        """Get summary of red-flag statistics."""
        return {
            "total_flagged": self.red_flag_stats["total_flagged"],
            "by_reason": dict(self.red_flag_stats["by_reason"]),
            "by_provider": dict(self.red_flag_stats["by_provider"]),
        }


@dataclass
class Metrics:
    """
    Computed metrics aligned with the paper.

    Paper metrics:
    - Overhead O% = (T_MAS - T_SAS) / T_SAS * 100
    - Efficiency Ec = Success / (T / T_SAS)
    - Error amplification Ae = E_MAS / E_SAS
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_elapsed_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # Per-provider stats
    provider_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Stage stats
    stage_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Selection stats
    selections_with_tests: int = 0
    selections_total: int = 0

    # Paper-aligned metrics
    coordination_overhead_percent: float = 0.0
    efficiency_score: float = 0.0
    consults_per_stage: float = 0.0

    @classmethod
    def compute(cls, trace_store: TraceStore, baseline_turns: int = 7) -> "Metrics":
        """
        Compute metrics from trace store.

        Args:
            trace_store: The trace store to compute from
            baseline_turns: Single-agent baseline turns (paper default: 7.2)
        """
        metrics = cls()

        # Provider call stats
        provider_stats = defaultdict(lambda: {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_ms": 0.0,
            "total_prompt_chars": 0,
            "total_response_chars": 0,
        })

        for call in trace_store.provider_calls:
            metrics.total_calls += 1
            metrics.total_elapsed_ms += call.elapsed_ms

            if call.success:
                metrics.successful_calls += 1
            else:
                metrics.failed_calls += 1

            ps = provider_stats[call.provider]
            ps["calls"] += 1
            ps["successes"] += 1 if call.success else 0
            ps["failures"] += 0 if call.success else 1
            ps["total_ms"] += call.elapsed_ms
            ps["total_prompt_chars"] += call.prompt_chars
            ps["total_response_chars"] += call.response_chars

        metrics.provider_stats = dict(provider_stats)

        # Average latency
        if metrics.total_calls > 0:
            metrics.avg_latency_ms = metrics.total_elapsed_ms / metrics.total_calls

        # Stage stats
        stage_stats = defaultdict(lambda: {
            "count": 0,
            "successes": 0,
            "total_consults": 0,
            "total_ms": 0.0,
        })

        for stage in trace_store.stage_traces:
            ss = stage_stats[stage.stage]
            ss["count"] += 1
            ss["successes"] += 1 if stage.success else 0
            ss["total_consults"] += stage.consults_used
            ss["total_ms"] += stage.elapsed_ms

        metrics.stage_stats = dict(stage_stats)

        # Selection stats
        for sel in trace_store.selection_traces:
            metrics.selections_total += 1
            if sel.had_test_signals:
                metrics.selections_with_tests += 1

        # Paper-aligned metrics
        total_turns = metrics.total_calls
        if total_turns > 0 and baseline_turns > 0:
            # Overhead O% = (T_MAS - T_SAS) / T_SAS * 100
            metrics.coordination_overhead_percent = (
                (total_turns - baseline_turns) / baseline_turns * 100
            )

            # Efficiency Ec = Success_rate / (T / T_SAS)
            success_rate = metrics.successful_calls / metrics.total_calls
            relative_turns = total_turns / baseline_turns
            metrics.efficiency_score = success_rate / relative_turns if relative_turns > 0 else 0

        # Consults per stage
        total_stages = len(trace_store.stage_traces)
        if total_stages > 0:
            metrics.consults_per_stage = metrics.total_calls / total_stages

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "total_elapsed_ms": self.total_elapsed_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "provider_stats": self.provider_stats,
            "stage_stats": self.stage_stats,
            "selections_total": self.selections_total,
            "selections_with_tests": self.selections_with_tests,
            "test_coverage_rate": self.selections_with_tests / self.selections_total if self.selections_total > 0 else 0,
            "coordination_overhead_percent": self.coordination_overhead_percent,
            "efficiency_score": self.efficiency_score,
            "consults_per_stage": self.consults_per_stage,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=== Maestro Skills Metrics ===",
            f"Total Provider Calls: {self.total_calls}",
            f"Success Rate: {self.successful_calls}/{self.total_calls} ({self.successful_calls/self.total_calls*100:.1f}%)" if self.total_calls > 0 else "No calls",
            f"Avg Latency: {self.avg_latency_ms:.0f}ms",
            "",
            "--- Paper-Aligned Metrics ---",
            f"Coordination Overhead (O%): {self.coordination_overhead_percent:.1f}%",
            f"Efficiency Score (Ec): {self.efficiency_score:.3f}",
            f"Consults per Stage: {self.consults_per_stage:.2f}",
            "",
            "--- Selection Quality ---",
            f"Selections with Tests: {self.selections_with_tests}/{self.selections_total}",
        ]

        if self.provider_stats:
            lines.append("")
            lines.append("--- Per-Provider Stats ---")
            for provider, stats in self.provider_stats.items():
                sr = stats['successes'] / stats['calls'] * 100 if stats['calls'] > 0 else 0
                lines.append(f"{provider}: {stats['calls']} calls, {sr:.0f}% success")

        return "\n".join(lines)
