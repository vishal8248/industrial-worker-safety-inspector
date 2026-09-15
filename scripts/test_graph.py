from app.workflow.incident_graph import (
    build_incident_graph,
)


def main():
    graph = build_incident_graph()

    state = {
        "incident_type": "phone_usage",
        "worker_id": 2,
        "timestamp": 10.07,
        "camera_id": "camera_01",
        "machine": "hydraulic_press",
        "evidence": [
            "Worker is holding a mobile phone "
            "and looking at the screen."
        ],
    }

    result = graph.invoke(state)

    print()
    print("LANGGRAPH INCIDENT WORKFLOW")
    print("===========================")
    print()
    print(result["incident_report"])


if __name__ == "__main__":
    main()