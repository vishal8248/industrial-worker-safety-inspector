from dataclasses import dataclass
from typing import Optional


@dataclass
class IncidentCandidate:
    worker_id: int
    incident_type: str
    start_time: float
    violation_time: float
    end_time: Optional[float] = None
    status: str = "open"

    def close(self, end_time: float):
        self.end_time = end_time
        self.status = "closed"

