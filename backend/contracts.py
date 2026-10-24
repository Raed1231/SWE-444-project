"""Bounded request and model-output contracts. No model-generated URLs."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from kb_routing import KNOWN_EVIDENCE_IDS

MAX_MESSAGE = 6000
MAX_HISTORY = 10
MAX_HISTORY_TEXT = 16000
MAX_HISTORY_TOTAL = 32000
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Message(StrictModel):
    role: Literal['user', 'assistant']
    text: Annotated[Text, Field(max_length=MAX_HISTORY_TEXT)]


class ComplaintRequest(StrictModel):
    message: Annotated[Text, Field(max_length=MAX_MESSAGE)]
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY)

    @field_validator('history')
    @classmethod
    def bounded_context(cls, history):
        if sum(len(item.text) for item in history) > MAX_HISTORY_TOTAL:
            raise ValueError('History exceeds the context limit')
        if history and (history[0].role != 'user' or history[-1].role != 'assistant'):
            raise ValueError('History must start with user and end with assistant')
        return history


class RoutingDecision(StrictModel):
    status: Literal['MATCHED', 'NEEDS_CLARIFICATION', 'NOT_SUPPORTED']
    service_id: str | None = Field(default=None, max_length=128)
    question: str = Field(default='', max_length=400)
    reason: str = Field(default='', max_length=800)
    # Constrain Gemini generation to source scenario/rule IDs. The independent
    # server guard still rejects unknown IDs and evidence for a different service.
    evidence_ids: list[Annotated[str, Field(max_length=128, json_schema_extra={
        'enum': sorted(KNOWN_EVIDENCE_IDS),
        'description': 'Documented complaint-scenario or routing-rule ID only; service IDs are not evidence.',
    })]] = Field(default_factory=list, max_length=8)

    @model_validator(mode='after')
    def consistent_decision(self):
        if self.status == 'MATCHED' and (not self.service_id or not self.reason.strip() or not self.evidence_ids):
            raise ValueError('A match requires a service, reason and KB evidence')
        if self.status != 'MATCHED' and self.service_id:
            raise ValueError('Only a match can contain a service ID')
        if self.status == 'NEEDS_CLARIFICATION' and not self.question.strip():
            raise ValueError('A clarification needs a question')
        if self.status == 'NOT_SUPPORTED' and not self.reason.strip():
            raise ValueError('An unsupported decision needs a reason')
        return self


class AttachmentAnalysis(StrictModel):
    summary: str = Field(max_length=800)
    extracted_facts: list[Annotated[str, Field(max_length=300)]] = Field(max_length=10)
    missing_information: list[Annotated[str, Field(max_length=250)]] = Field(max_length=6)
    unreadable_content: str = Field(max_length=400, description='Describe unreadable content, or an empty string when readable.')


class AttachmentDecision(StrictModel):
    analysis: AttachmentAnalysis
    routing: RoutingDecision
