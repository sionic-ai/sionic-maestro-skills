"""
Human-in-the-Loop (HITL) Module for Maestro Workflow

Each stage completion requires explicit human approval before proceeding.
This module provides:
1. Detailed stage reports with comprehensive information
2. Specific questions for human review at each stage
3. Approval/rejection flow with feedback handling
4. Stage-specific review criteria

Based on the principle: "매 stage마다 사용자의 의견을 매번 자세하게 꼼꼼히 물어보기"
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Deque
import json

logger = logging.getLogger("maestro.human_loop")

# Default max history size to prevent memory leaks
DEFAULT_MAX_HISTORY = 100


class ApprovalStatus(Enum):
    """Status of a stage approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"
    SKIPPED = "skipped"


class ReviewPriority(Enum):
    """Priority level for review items."""
    CRITICAL = "critical"  # Must be reviewed before proceeding
    HIGH = "high"          # Strongly recommended to review
    MEDIUM = "medium"      # Should review if time permits
    LOW = "low"            # Optional review


@dataclass
class ReviewQuestion:
    """A specific question for the human reviewer."""
    id: str
    question: str
    question_ko: str  # Korean translation
    context: str
    priority: ReviewPriority
    options: List[str] = field(default_factory=list)  # If multiple choice
    requires_text_response: bool = False
    response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "question_ko": self.question_ko,
            "context": self.context,
            "priority": self.priority.value,
            "options": self.options,
            "requires_text_response": self.requires_text_response,
            "response": self.response,
        }


@dataclass
class StageReport:
    """Comprehensive report for a completed stage."""
    stage: str
    stage_display_name: str
    stage_display_name_ko: str
    timestamp: str
    duration_ms: float

    # Summary
    summary: str
    summary_ko: str

    # Detailed outputs
    outputs: Dict[str, Any]

    # Key findings/artifacts
    key_findings: List[str]
    key_findings_ko: List[str]

    # Potential issues/risks
    risks: List[str]
    risks_ko: List[str]

    # Review questions
    questions: List[ReviewQuestion]

    # Metrics
    metrics: Dict[str, Any]

    # Context for next stage
    next_stage: Optional[str]
    next_stage_preview: str
    next_stage_preview_ko: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_display_name": self.stage_display_name,
            "stage_display_name_ko": self.stage_display_name_ko,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "summary": self.summary,
            "summary_ko": self.summary_ko,
            "outputs": self.outputs,
            "key_findings": self.key_findings,
            "key_findings_ko": self.key_findings_ko,
            "risks": self.risks,
            "risks_ko": self.risks_ko,
            "questions": [q.to_dict() for q in self.questions],
            "metrics": self.metrics,
            "next_stage": self.next_stage,
            "next_stage_preview": self.next_stage_preview,
            "next_stage_preview_ko": self.next_stage_preview_ko,
        }


@dataclass
class ApprovalRequest:
    """A request for human approval of a stage."""
    request_id: str
    stage: str
    report: StageReport
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    responded_at: Optional[str] = None

    # Human responses
    approval_decision: Optional[bool] = None
    feedback: Optional[str] = None
    question_responses: Dict[str, str] = field(default_factory=dict)
    revision_instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "stage": self.stage,
            "report": self.report.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "responded_at": self.responded_at,
            "approval_decision": self.approval_decision,
            "feedback": self.feedback,
            "question_responses": self.question_responses,
            "revision_instructions": self.revision_instructions,
        }


# Stage-specific review questions
STAGE_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "analyze": [
        {
            "id": "analyze_completeness",
            "question": "Are all relevant observations captured? Is anything missing?",
            "question_ko": "모든 관련 관찰 사항이 포함되었나요? 빠진 것이 있나요?",
            "priority": ReviewPriority.CRITICAL,
            "requires_text_response": True,
        },
        {
            "id": "analyze_accuracy",
            "question": "Are the identified facts accurate based on your domain knowledge?",
            "question_ko": "식별된 사실들이 귀하의 도메인 지식에 따라 정확한가요?",
            "priority": ReviewPriority.CRITICAL,
            "options": ["Yes, all accurate", "Some inaccuracies", "Major errors"],
        },
        {
            "id": "analyze_files",
            "question": "Are the affected files/modules correctly identified?",
            "question_ko": "영향을 받는 파일/모듈이 정확히 식별되었나요?",
            "priority": ReviewPriority.HIGH,
            "options": ["Yes", "Partially", "No, missing important files"],
        },
        {
            "id": "analyze_repro",
            "question": "Can you confirm the reproduction steps are correct?",
            "question_ko": "재현 단계가 올바른지 확인해 주시겠어요?",
            "priority": ReviewPriority.HIGH,
            "requires_text_response": True,
        },
        {
            "id": "analyze_invariants",
            "question": "Are there any invariants (things that must not break) that were missed?",
            "question_ko": "누락된 불변 조건(깨지면 안 되는 것들)이 있나요?",
            "priority": ReviewPriority.MEDIUM,
            "requires_text_response": True,
        },
    ],
    "hypothesize": [
        {
            "id": "hypo_plausibility",
            "question": "Which hypothesis seems most plausible based on your experience?",
            "question_ko": "귀하의 경험에 기반하여 어떤 가설이 가장 그럴듯해 보이나요?",
            "priority": ReviewPriority.CRITICAL,
            "requires_text_response": True,
        },
        {
            "id": "hypo_missing",
            "question": "Are there any potential causes that weren't considered?",
            "question_ko": "고려되지 않은 잠재적 원인이 있나요?",
            "priority": ReviewPriority.CRITICAL,
            "requires_text_response": True,
        },
        {
            "id": "hypo_testability",
            "question": "Are the proposed verification methods practical and feasible?",
            "question_ko": "제안된 검증 방법이 실용적이고 실현 가능한가요?",
            "priority": ReviewPriority.HIGH,
            "options": ["Yes, all feasible", "Some need adjustment", "Major issues with testability"],
        },
        {
            "id": "hypo_priority",
            "question": "Do you agree with the recommended testing order?",
            "question_ko": "권장된 테스트 순서에 동의하시나요?",
            "priority": ReviewPriority.MEDIUM,
            "options": ["Yes", "No, suggest different order"],
        },
        {
            "id": "hypo_risk",
            "question": "Which hypothesis would be most dangerous if wrong?",
            "question_ko": "어떤 가설이 틀릴 경우 가장 위험할까요?",
            "priority": ReviewPriority.MEDIUM,
            "requires_text_response": True,
        },
    ],
    "implement": [
        {
            "id": "impl_approach",
            "question": "Is this the right approach to fix the issue?",
            "question_ko": "이것이 문제를 해결하는 올바른 접근 방식인가요?",
            "priority": ReviewPriority.CRITICAL,
            "options": ["Yes", "Partially, needs adjustment", "No, different approach needed"],
        },
        {
            "id": "impl_scope",
            "question": "Is the change scope minimal and focused?",
            "question_ko": "변경 범위가 최소화되고 집중되어 있나요?",
            "priority": ReviewPriority.CRITICAL,
            "options": ["Yes, minimal", "Could be smaller", "Too broad, needs refocusing"],
        },
        {
            "id": "impl_side_effects",
            "question": "Are there any potential side effects or breaking changes?",
            "question_ko": "잠재적인 부작용이나 호환성 문제가 있나요?",
            "priority": ReviewPriority.CRITICAL,
            "requires_text_response": True,
        },
        {
            "id": "impl_tests",
            "question": "Is the verification command appropriate?",
            "question_ko": "검증 명령어가 적절한가요?",
            "priority": ReviewPriority.HIGH,
            "options": ["Yes", "Need additional tests", "Wrong test target"],
        },
        {
            "id": "impl_review",
            "question": "Please review the diff - any concerns with the code quality?",
            "question_ko": "diff를 검토해 주세요 - 코드 품질에 대한 우려 사항이 있나요?",
            "priority": ReviewPriority.HIGH,
            "requires_text_response": True,
        },
        {
            "id": "impl_rollback",
            "question": "If this fails, do you want to automatically rollback?",
            "question_ko": "실패할 경우 자동으로 롤백할까요?",
            "priority": ReviewPriority.MEDIUM,
            "options": ["Yes, auto-rollback", "No, keep changes for debugging"],
        },
    ],
    "debug": [
        {
            "id": "debug_analysis",
            "question": "Is the error analysis correct?",
            "question_ko": "오류 분석이 정확한가요?",
            "priority": ReviewPriority.CRITICAL,
            "options": ["Yes", "Partially", "No, missed the real issue"],
        },
        {
            "id": "debug_hypothesis",
            "question": "Should we continue with the current hypothesis or try alternative?",
            "question_ko": "현재 가설을 계속 추구할까요, 아니면 대안을 시도할까요?",
            "priority": ReviewPriority.CRITICAL,
            "options": ["Continue current", "Try alternative hypothesis", "Need more information"],
        },
        {
            "id": "debug_action",
            "question": "Is the proposed next action appropriate?",
            "question_ko": "제안된 다음 조치가 적절한가요?",
            "priority": ReviewPriority.HIGH,
            "options": ["Yes, proceed", "Modify action", "Different action needed"],
        },
        {
            "id": "debug_iteration",
            "question": "How many more debug iterations should we allow?",
            "question_ko": "몇 번의 디버그 반복을 더 허용할까요?",
            "priority": ReviewPriority.MEDIUM,
            "options": ["1 more", "2-3 more", "5 more", "Escalate to human now"],
        },
        {
            "id": "debug_insight",
            "question": "Do you have any insights that could help with debugging?",
            "question_ko": "디버깅에 도움이 될 만한 인사이트가 있으신가요?",
            "priority": ReviewPriority.HIGH,
            "requires_text_response": True,
        },
    ],
    "improve": [
        {
            "id": "improve_necessary",
            "question": "Are the suggested improvements necessary?",
            "question_ko": "제안된 개선 사항이 필요한가요?",
            "priority": ReviewPriority.CRITICAL,
            "options": ["Yes, all needed", "Some needed", "Over-engineering, skip most"],
        },
        {
            "id": "improve_priority",
            "question": "Which improvements should be prioritized?",
            "question_ko": "어떤 개선 사항을 우선시해야 하나요?",
            "priority": ReviewPriority.HIGH,
            "requires_text_response": True,
        },
        {
            "id": "improve_risk",
            "question": "Are any improvements too risky to implement now?",
            "question_ko": "지금 구현하기에 너무 위험한 개선 사항이 있나요?",
            "priority": ReviewPriority.HIGH,
            "requires_text_response": True,
        },
        {
            "id": "improve_tests",
            "question": "Are the suggested regression tests sufficient?",
            "question_ko": "제안된 회귀 테스트가 충분한가요?",
            "priority": ReviewPriority.MEDIUM,
            "options": ["Yes", "Need more tests", "Tests too extensive"],
        },
        {
            "id": "improve_merge",
            "question": "Is the code ready for merge?",
            "question_ko": "코드가 머지 준비가 되었나요?",
            "priority": ReviewPriority.CRITICAL,
            "options": ["Yes, ready to merge", "Need minor changes", "Not ready, major issues"],
        },
    ],
}


# Stage display names in Korean
STAGE_NAMES_KO: Dict[str, str] = {
    "analyze": "예제 분석 (Example Analysis)",
    "hypothesize": "가설 수립 (Hypothesis Formulation)",
    "implement": "코드 구현 (Code Implementation)",
    "debug": "반복 디버깅 (Iterative Debugging)",
    "improve": "재귀적 개선 (Recursive Improvement)",
}

# Next stage preview descriptions
NEXT_STAGE_PREVIEW: Dict[str, Dict[str, str]] = {
    "analyze": {
        "en": "Next: Generate competing hypotheses about root cause based on the analyzed facts.",
        "ko": "다음: 분석된 사실을 기반으로 근본 원인에 대한 경쟁 가설을 생성합니다.",
    },
    "hypothesize": {
        "en": "Next: Implement the minimal change to verify/fix the selected hypothesis.",
        "ko": "다음: 선택된 가설을 검증/수정하기 위한 최소한의 변경을 구현합니다.",
    },
    "implement": {
        "en": "Next: Run verification tests. If failed, enter debug loop.",
        "ko": "다음: 검증 테스트를 실행합니다. 실패하면 디버그 루프에 진입합니다.",
    },
    "debug": {
        "en": "Next: Continue debugging or move to improvement phase if tests pass.",
        "ko": "다음: 디버깅을 계속하거나 테스트가 통과하면 개선 단계로 이동합니다.",
    },
    "improve": {
        "en": "Workflow complete. Ready for final review and merge.",
        "ko": "워크플로우 완료. 최종 검토 및 머지 준비가 되었습니다.",
    },
}


class HumanLoopManager:
    """
    Manages the human-in-the-loop approval process.

    Responsibilities:
    1. Generate comprehensive stage reports
    2. Create approval requests with detailed questions
    3. Track approval status and responses
    4. Provide guidance for next steps based on feedback

    Thread Safety:
    All state modifications are protected by a lock for safe concurrent access.

    Memory Management:
    completed_requests uses a deque with maxlen to prevent unbounded growth.
    """

    def __init__(self, max_history: int = DEFAULT_MAX_HISTORY):
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        # Use deque with maxlen to prevent memory leak
        self.completed_requests: Deque[ApprovalRequest] = deque(maxlen=max_history)
        self._request_counter = 0
        # Thread safety lock
        self._lock = threading.Lock()

    def _generate_request_id(self, stage: str) -> str:
        """Generate unique request ID. Must be called within lock."""
        self._request_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"approval_{stage}_{timestamp}_{self._request_counter}"

    def _build_questions(self, stage: str, outputs: Dict[str, Any]) -> List[ReviewQuestion]:
        """Build review questions for a stage."""
        stage_q = STAGE_QUESTIONS.get(stage, [])
        questions = []

        for q_def in stage_q:
            # Add context based on outputs
            context = self._extract_question_context(q_def["id"], outputs)

            questions.append(ReviewQuestion(
                id=q_def["id"],
                question=q_def["question"],
                question_ko=q_def["question_ko"],
                context=context,
                priority=q_def["priority"],
                options=q_def.get("options", []),
                requires_text_response=q_def.get("requires_text_response", False),
            ))

        return questions

    def _extract_question_context(self, question_id: str, outputs: Dict[str, Any]) -> str:
        """Extract relevant context for a specific question."""
        # Context varies by question type
        if "completeness" in question_id:
            if "observations" in outputs:
                return f"Identified {len(outputs['observations'])} observations"
            return "See outputs section for details"

        if "hypothesis" in question_id or "hypo" in question_id:
            if "hypotheses" in outputs:
                return f"{len(outputs['hypotheses'])} hypotheses generated"
            return "See hypotheses section for details"

        if "impl" in question_id:
            if "diff" in outputs:
                return f"Patch affects {len(outputs.get('files_to_change', []))} files"
            return "See implementation details"

        if "debug" in question_id:
            if "error_analysis" in outputs:
                return outputs["error_analysis"][:200]
            return "See error analysis"

        return "Refer to the detailed outputs above"

    def _extract_key_findings(self, stage: str, outputs: Dict[str, Any]) -> tuple:
        """Extract key findings from stage outputs."""
        findings_en = []
        findings_ko = []

        if stage == "analyze":
            if "observations" in outputs:
                findings_en.append(f"Identified {len(outputs['observations'])} key observations")
                findings_ko.append(f"{len(outputs['observations'])}개의 주요 관찰 사항 식별")
            if "affected_modules" in outputs:
                findings_en.append(f"Affected modules: {', '.join(outputs['affected_modules'][:3])}")
                findings_ko.append(f"영향받는 모듈: {', '.join(outputs['affected_modules'][:3])}")

        elif stage == "hypothesize":
            if "hypotheses" in outputs:
                findings_en.append(f"Generated {len(outputs['hypotheses'])} competing hypotheses")
                findings_ko.append(f"{len(outputs['hypotheses'])}개의 경쟁 가설 생성")
            if "recommended_order" in outputs:
                findings_en.append(f"Recommended testing order: {outputs['recommended_order']}")
                findings_ko.append(f"권장 테스트 순서: {outputs['recommended_order']}")

        elif stage == "implement":
            if "files_to_change" in outputs:
                findings_en.append(f"Files to modify: {', '.join(outputs['files_to_change'])}")
                findings_ko.append(f"수정할 파일: {', '.join(outputs['files_to_change'])}")
            if "patch_plan" in outputs:
                findings_en.append(f"Plan: {outputs['patch_plan']}")
                findings_ko.append(f"계획: {outputs['patch_plan']}")

        elif stage == "debug":
            if "error_analysis" in outputs:
                findings_en.append(f"Error analysis: {outputs['error_analysis'][:100]}...")
                findings_ko.append(f"오류 분석: {outputs['error_analysis'][:100]}...")
            if "next_action" in outputs:
                action = outputs["next_action"]
                findings_en.append(f"Next action: {action.get('type', 'unknown')}")
                findings_ko.append(f"다음 조치: {action.get('type', 'unknown')}")

        elif stage == "improve":
            if "improvements" in outputs:
                findings_en.append(f"Suggested {len(outputs['improvements'])} improvements")
                findings_ko.append(f"{len(outputs['improvements'])}개의 개선 사항 제안")
            if "ready_for_merge" in outputs:
                status = "Ready" if outputs["ready_for_merge"] else "Not ready"
                status_ko = "준비됨" if outputs["ready_for_merge"] else "준비되지 않음"
                findings_en.append(f"Merge status: {status}")
                findings_ko.append(f"머지 상태: {status_ko}")

        return findings_en, findings_ko

    def _extract_risks(self, stage: str, outputs: Dict[str, Any]) -> tuple:
        """Extract potential risks from stage outputs."""
        risks_en = []
        risks_ko = []

        if "risks" in outputs:
            for risk in outputs["risks"][:5]:
                risks_en.append(risk)
                risks_ko.append(risk)  # Would need translation in production

        # Stage-specific risk extraction
        if stage == "implement":
            if not outputs.get("verification_command"):
                risks_en.append("No verification command specified")
                risks_ko.append("검증 명령어가 지정되지 않음")

        elif stage == "debug":
            attempts = outputs.get("attempts", 0)
            if attempts >= 3:
                risks_en.append(f"High number of debug iterations ({attempts})")
                risks_ko.append(f"높은 디버그 반복 횟수 ({attempts})")

        elif stage == "hypothesize":
            if len(outputs.get("hypotheses", [])) < 2:
                risks_en.append("Only one hypothesis - may miss alternative causes")
                risks_ko.append("가설이 하나뿐 - 대안적 원인을 놓칠 수 있음")

        return risks_en, risks_ko

    def create_stage_report(
        self,
        stage: str,
        outputs: Dict[str, Any],
        duration_ms: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> StageReport:
        """Create a comprehensive stage report."""
        from .workflow import Stage

        # Get stage enum for display names
        try:
            stage_enum = Stage(stage)
            stage_display = stage_enum.display_name
        except ValueError:
            stage_display = stage.replace("_", " ").title()

        # Extract findings and risks
        findings_en, findings_ko = self._extract_key_findings(stage, outputs)
        risks_en, risks_ko = self._extract_risks(stage, outputs)

        # Build questions
        questions = self._build_questions(stage, outputs)

        # Get next stage info
        try:
            next_stage = Stage(stage).next_stage
            next_stage_name = next_stage.value if next_stage else None
        except ValueError:
            next_stage_name = None

        preview = NEXT_STAGE_PREVIEW.get(stage, {"en": "", "ko": ""})

        # Generate summary
        summary_en = self._generate_summary(stage, outputs, findings_en)
        summary_ko = self._generate_summary_ko(stage, outputs, findings_ko)

        return StageReport(
            stage=stage,
            stage_display_name=stage_display,
            stage_display_name_ko=STAGE_NAMES_KO.get(stage, stage_display),
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
            summary=summary_en,
            summary_ko=summary_ko,
            outputs=outputs,
            key_findings=findings_en,
            key_findings_ko=findings_ko,
            risks=risks_en,
            risks_ko=risks_ko,
            questions=questions,
            metrics=metrics or {},
            next_stage=next_stage_name,
            next_stage_preview=preview["en"],
            next_stage_preview_ko=preview["ko"],
        )

    def _generate_summary(self, stage: str, outputs: Dict[str, Any], findings: List[str]) -> str:
        """Generate English summary for stage."""
        if stage == "analyze":
            return f"Analysis complete. {'. '.join(findings[:2]) if findings else 'See outputs for details.'}"
        elif stage == "hypothesize":
            return f"Hypothesis generation complete. {'. '.join(findings[:2]) if findings else 'See outputs for details.'}"
        elif stage == "implement":
            return f"Implementation ready. {'. '.join(findings[:2]) if findings else 'See outputs for details.'}"
        elif stage == "debug":
            return f"Debug iteration complete. {'. '.join(findings[:2]) if findings else 'See outputs for details.'}"
        elif stage == "improve":
            return f"Improvement analysis complete. {'. '.join(findings[:2]) if findings else 'See outputs for details.'}"
        return f"Stage {stage} complete."

    def _generate_summary_ko(self, stage: str, outputs: Dict[str, Any], findings: List[str]) -> str:
        """Generate Korean summary for stage."""
        if stage == "analyze":
            return f"분석 완료. {'. '.join(findings[:2]) if findings else '자세한 내용은 출력을 참조하세요.'}"
        elif stage == "hypothesize":
            return f"가설 생성 완료. {'. '.join(findings[:2]) if findings else '자세한 내용은 출력을 참조하세요.'}"
        elif stage == "implement":
            return f"구현 준비 완료. {'. '.join(findings[:2]) if findings else '자세한 내용은 출력을 참조하세요.'}"
        elif stage == "debug":
            return f"디버그 반복 완료. {'. '.join(findings[:2]) if findings else '자세한 내용은 출력을 참조하세요.'}"
        elif stage == "improve":
            return f"개선 분석 완료. {'. '.join(findings[:2]) if findings else '자세한 내용은 출력을 참조하세요.'}"
        return f"스테이지 {stage} 완료."

    def request_approval(
        self,
        stage: str,
        outputs: Dict[str, Any],
        duration_ms: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Create and register an approval request.

        Returns the ApprovalRequest that must be approved before proceeding.
        Thread-safe: Uses internal lock for state modification.
        """
        # Create report outside lock (no shared state access)
        report = self.create_stage_report(stage, outputs, duration_ms, metrics)

        with self._lock:
            request_id = self._generate_request_id(stage)

            request = ApprovalRequest(
                request_id=request_id,
                stage=stage,
                report=report,
            )

            self.pending_requests[request_id] = request

        logger.info(f"Created approval request {request_id} for stage {stage}")
        return request

    def submit_approval(
        self,
        request_id: str,
        approved: bool,
        feedback: Optional[str] = None,
        question_responses: Optional[Dict[str, str]] = None,
        revision_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit approval decision for a pending request.

        Args:
            request_id: The approval request ID
            approved: True to approve, False to reject
            feedback: General feedback from reviewer
            question_responses: Responses to specific questions
            revision_instructions: Instructions if revision is requested

        Returns:
            Dict with status and next steps

        Thread-safe: Uses internal lock for state modification.
        """
        with self._lock:
            if request_id not in self.pending_requests:
                return {
                    "ok": False,
                    "error": f"Request {request_id} not found or already processed",
                }

            request = self.pending_requests[request_id]
            request.responded_at = datetime.now().isoformat()
            request.approval_decision = approved
            request.feedback = feedback
            request.question_responses = question_responses or {}
            request.revision_instructions = revision_instructions

            if approved:
                request.status = ApprovalStatus.APPROVED
                next_action = "proceed_to_next_stage"
                message = f"Stage {request.stage} approved. Proceeding to next stage."
                message_ko = f"스테이지 {request.stage} 승인됨. 다음 스테이지로 진행합니다."
            elif revision_instructions:
                request.status = ApprovalStatus.REVISION_REQUESTED
                next_action = "revise_current_stage"
                message = f"Revision requested for stage {request.stage}."
                message_ko = f"스테이지 {request.stage}에 대한 수정이 요청되었습니다."
            else:
                request.status = ApprovalStatus.REJECTED
                next_action = "stop_workflow"
                message = f"Stage {request.stage} rejected. Workflow stopped."
                message_ko = f"스테이지 {request.stage} 거부됨. 워크플로우가 중단되었습니다."

            # Move to completed
            del self.pending_requests[request_id]
            self.completed_requests.append(request)

        logger.info(f"Approval request {request_id}: {request.status.value}")

        return {
            "ok": True,
            "request_id": request_id,
            "status": request.status.value,
            "next_action": next_action,
            "message": message,
            "message_ko": message_ko,
            "next_stage": request.report.next_stage if approved else None,
            "feedback_summary": self._summarize_feedback(request),
        }

    def _summarize_feedback(self, request: ApprovalRequest) -> Dict[str, Any]:
        """Summarize all feedback from a request."""
        summary = {
            "general_feedback": request.feedback,
            "question_responses": request.question_responses,
            "revision_instructions": request.revision_instructions,
            "critical_concerns": [],
        }

        # Extract critical concerns from question responses
        for q in request.report.questions:
            if q.priority == ReviewPriority.CRITICAL:
                response = request.question_responses.get(q.id)
                if response and ("no" in response.lower() or "error" in response.lower() or "issue" in response.lower()):
                    summary["critical_concerns"].append({
                        "question": q.question,
                        "response": response,
                    })

        return summary

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests. Thread-safe."""
        with self._lock:
            return [req.to_dict() for req in self.pending_requests.values()]

    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific request by ID. Thread-safe."""
        with self._lock:
            if request_id in self.pending_requests:
                return self.pending_requests[request_id].to_dict()

            for req in self.completed_requests:
                if req.request_id == request_id:
                    return req.to_dict()

            return None

    def get_approval_history(self) -> List[Dict[str, Any]]:
        """Get history of all approval decisions. Thread-safe."""
        with self._lock:
            return [req.to_dict() for req in self.completed_requests]

    def skip_approval(self, request_id: str, reason: str) -> Dict[str, Any]:
        """
        Skip approval for a request (for automated workflows).

        This should only be used in trusted automated environments.
        Thread-safe: Uses internal lock for state modification.
        """
        with self._lock:
            if request_id not in self.pending_requests:
                return {
                    "ok": False,
                    "error": f"Request {request_id} not found",
                }

            request = self.pending_requests[request_id]
            request.status = ApprovalStatus.SKIPPED
            request.feedback = f"Skipped: {reason}"
            request.responded_at = datetime.now().isoformat()

            del self.pending_requests[request_id]
            self.completed_requests.append(request)

        logger.warning(f"Approval skipped for {request_id}: {reason}")

        return {
            "ok": True,
            "request_id": request_id,
            "status": "skipped",
            "warning": "Approval was skipped - proceeding without human review",
        }


def format_approval_request_for_display(request: ApprovalRequest) -> str:
    """
    Format an approval request for display in CLI/chat.

    Returns a formatted string with all relevant information.
    """
    report = request.report
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"🔍 STAGE APPROVAL REQUEST: {report.stage_display_name}")
    lines.append(f"   {report.stage_display_name_ko}")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append("📋 SUMMARY / 요약")
    lines.append("-" * 40)
    lines.append(f"EN: {report.summary}")
    lines.append(f"KO: {report.summary_ko}")
    lines.append("")

    # Key Findings
    lines.append("🔑 KEY FINDINGS / 주요 발견사항")
    lines.append("-" * 40)
    for i, (en, ko) in enumerate(zip(report.key_findings, report.key_findings_ko), 1):
        lines.append(f"{i}. {en}")
        lines.append(f"   {ko}")
    lines.append("")

    # Risks
    if report.risks:
        lines.append("⚠️ RISKS / 위험 요소")
        lines.append("-" * 40)
        for i, (en, ko) in enumerate(zip(report.risks, report.risks_ko), 1):
            lines.append(f"{i}. {en}")
            lines.append(f"   {ko}")
        lines.append("")

    # Questions
    lines.append("❓ REVIEW QUESTIONS / 검토 질문")
    lines.append("-" * 40)

    for q in report.questions:
        priority_emoji = {
            ReviewPriority.CRITICAL: "🔴",
            ReviewPriority.HIGH: "🟠",
            ReviewPriority.MEDIUM: "🟡",
            ReviewPriority.LOW: "🟢",
        }.get(q.priority, "⚪")

        lines.append(f"{priority_emoji} [{q.priority.value.upper()}] {q.question}")
        lines.append(f"   {q.question_ko}")

        if q.options:
            lines.append(f"   Options: {' | '.join(q.options)}")

        if q.requires_text_response:
            lines.append("   (Requires text response / 텍스트 응답 필요)")

        lines.append("")

    # Next Stage Preview
    lines.append("➡️ NEXT STAGE PREVIEW / 다음 스테이지 미리보기")
    lines.append("-" * 40)
    lines.append(f"EN: {report.next_stage_preview}")
    lines.append(f"KO: {report.next_stage_preview_ko}")
    lines.append("")

    # Metrics
    lines.append("📊 METRICS / 메트릭")
    lines.append("-" * 40)
    lines.append(f"Duration: {report.duration_ms:.2f}ms")
    lines.append(f"Request ID: {request.request_id}")
    lines.append("")

    # Action required
    lines.append("=" * 60)
    lines.append("🎯 ACTION REQUIRED / 필요한 조치")
    lines.append("Please review and approve/reject this stage.")
    lines.append("이 스테이지를 검토하고 승인/거부해 주세요.")
    lines.append("=" * 60)

    return "\n".join(lines)
