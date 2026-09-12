from typing import Dict, List

from app.incidents.models import IncidentCandidate


class IncidentManager:
    def __init__(self, max_missing_frames: int = 15):
        self.active_incidents: Dict[int, IncidentCandidate] = {}
        self.completed_incidents: List[IncidentCandidate] = []
        self.missing_frames: Dict[int, int] = {}
        self.max_missing_frames = max_missing_frames

    def update(self, status: dict):
        worker_id = status["worker_id"]
        inside_zone = status["inside_zone"]
        violation = status["violation"]
        timestamp = status["timestamp"]

        self.missing_frames[worker_id] = 0

        incident = self.active_incidents.get(worker_id)

        if inside_zone and incident is None:
            incident = IncidentCandidate(
                worker_id=worker_id,
                incident_type="restricted_zone_dwell",
                start_time=timestamp,
                violation_time=0.0,
            )

            self.active_incidents[worker_id] = incident

        if violation and incident is not None:
            if incident.violation_time == 0.0:
                incident.violation_time = timestamp

        if not inside_zone and incident is not None:
            incident.close(timestamp)

            self.completed_incidents.append(incident)
            del self.active_incidents[worker_id]

            return incident

        return None

    def mark_missing(self, worker_id: int):
        self.missing_frames[worker_id] = (
            self.missing_frames.get(worker_id, 0) + 1
        )

        if self.missing_frames[worker_id] <= self.max_missing_frames:
            return None

        incident = self.active_incidents.get(worker_id)

        if incident is not None:
            incident.close(incident.start_time)

            self.completed_incidents.append(incident)
            del self.active_incidents[worker_id]

            return incident

        return None

    def get_active_incident(self, worker_id: int):
        return self.active_incidents.get(worker_id)

    def get_completed_incidents(self):
        return self.completed_incidents