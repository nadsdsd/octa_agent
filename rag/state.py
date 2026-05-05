from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict

AgentName = Literal[
    "intent_node",
    "vision_node",
    "diagnosis_node",
    "physical_node",
    "pathology_node",
    "rag_node",
    "final_node",
]

AgentRunStatus = Literal["pending", "success", "failed", "skipped", "degraded"]


class AgentStatus(TypedDict, total=False):
    status: AgentRunStatus
    reason: str
    attempts: int
    output_valid: bool
    last_error: Optional[str]
    skipped_by_ablation: bool


class SessionContext(TypedDict, total=False):
    rv_mask_base64: Optional[str]
    faz_mask_base64: Optional[str]
    metrics: Optional[Dict[str, Any]]
    scan_type: Optional[str]
    diagnosis_result: Optional[Dict[str, Any]]
    last_disease_label: Optional[str]
    physical_metrics: Optional[Dict[str, Any]]
    image_diagnosis_label: Optional[str]
    image_confidence: Optional[float]
    physical_diagnosis_label: Optional[str]
    pathology_diagnosis_result: Optional[Dict[str, Any]]
    physical_confidence: Optional[float]
    symptoms_requested: bool
    pending_report_after_pathology: bool


class AgentState(TypedDict, total=False):
    session_id: str
    user_text: str
    image_path: Optional[str]
    has_new_image: bool
    history: List[Dict[str, str]]
    context: SessionContext

    user_intent: Literal["segment_only", "diagnosis", "consult", "general_chat", "provide_symptoms"]
    needs_vision: bool
    needs_diagnosis: bool
    needs_rag: bool
    needs_physical: bool
    needs_pathology: bool
    route_reason: str
    # physical_confidence: Optional[float]
    enabled_agents: Dict[str, bool]
    agent_status: Dict[str, AgentStatus]
    warnings: List[str]
    last_node: Optional[str]

    vision_result: Optional[Dict[str, Any]]
    diagnosis_result: Optional[Dict[str, Any]]
    physical_result: Optional[Dict[str, Any]]
    pathology_result: Optional[Dict[str, Any]]
    rag_result: Optional[str]
    rag_docs: Optional[List[str]]

    blocked: bool
    error: Optional[str]
    final_text: str
    progress_event: Optional[Dict[str, Any]]
    thinking_details: List[Dict[str, Any]]
