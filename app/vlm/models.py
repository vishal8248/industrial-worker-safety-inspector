from pydantic import BaseModel


class SafetyObservations(BaseModel):
    phone_usage: bool
    ppe_violation: bool
    machine_interaction: bool
    unsafe_position: bool
    evidence: list[str]


class IncidentAnalysis(BaseModel):
    worker_id: int
    timestamp: float
    observations: SafetyObservations