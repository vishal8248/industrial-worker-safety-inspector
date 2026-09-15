from app.rag.sop_retriever import SOPRetriever


def main():
    retriever = SOPRetriever()

    incident_type = "phone_usage"

    results = retriever.retrieve(
        incident_type=incident_type,
        top_k=2,
    )

    print()
    print("SOP RETRIEVAL TEST")
    print("------------------")
    print(
        f"Incident type: {incident_type}"
    )
    print()

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"RESULT {index}"
        )
        print(result)
        print()
        print("-" * 60)


if __name__ == "__main__":
    main()