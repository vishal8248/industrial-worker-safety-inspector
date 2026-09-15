from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from app.notifications.email import EmailNotifier
from app.rag.sop_retriever import SOPRetriever


load_dotenv()


class IncidentState(TypedDict, total=False):
    incident_type: str
    worker_id: int
    timestamp: float
    camera_id: str
    machine: str
    evidence: list[str]

    sop_guidance: list[str]
    incident_report: str
    notification_sent: bool


retriever = SOPRetriever()


def retrieve_sop(state: IncidentState):
    results = retriever.retrieve(
        incident_type=state["incident_type"],
        top_k=2,
    )

    return {
        "sop_guidance": results,
    }


def generate_report(state: IncidentState):
    incident_type = state["incident_type"]
    worker_id = state["worker_id"]
    timestamp = state["timestamp"]
    camera_id = state["camera_id"]

    machine = state.get(
        "machine",
        "Unknown",
    )

    evidence = state.get(
        "evidence",
        [],
    )

    sop_guidance = state.get(
        "sop_guidance",
        [],
    )

    report_lines = [
        "INDUSTRIAL SAFETY INCIDENT REPORT",
        "",
        f"Incident Type: {incident_type}",
        f"Worker ID: {worker_id}",
        f"Timestamp: {timestamp:.2f}s",
        f"Camera: {camera_id}",
        f"Machine / Area: {machine}",
        "",
        "Observed Evidence:",
    ]

    for item in evidence:
        report_lines.append(
            f"- {item}"
        )

    report_lines.extend(
        [
            "",
            "Applicable SOP Guidance:",
        ]
    )

    for item in sop_guidance:
        report_lines.append(item)

    return {
        "incident_report": "\n".join(
            report_lines
        )
    }


def notify_supervisor(state: IncidentState):
    notifier = EmailNotifier()

    incident_type = state["incident_type"]
    worker_id = state["worker_id"]

    subject = (
        "Industrial Safety Incident - "
        f"{incident_type} - Worker {worker_id}"
    )

    notifier.send_incident_alert(
        subject=subject,
        report=state["incident_report"],
    )

    return {
        "notification_sent": True,
    }


def build_incident_graph():
    graph = StateGraph(
        IncidentState
    )

    graph.add_node(
        "retrieve_sop",
        retrieve_sop,
    )

    graph.add_node(
        "generate_report",
        generate_report,
    )

    graph.add_node(
        "notify_supervisor",
        notify_supervisor,
    )

    graph.add_edge(
        START,
        "retrieve_sop",
    )

    graph.add_edge(
        "retrieve_sop",
        "generate_report",
    )

    graph.add_edge(
        "generate_report",
        "notify_supervisor",
    )

    graph.add_edge(
        "notify_supervisor",
        END,
    )

    return graph.compile()