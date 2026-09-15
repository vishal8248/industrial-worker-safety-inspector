from pathlib import Path
import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class SOPRetriever:
    def __init__(
        self,
        sop_path: str = "data/sops/safety_sop.txt",
    ):
        self.sop_path = Path(sop_path)

        if not self.sop_path.exists():
            raise FileNotFoundError(
                f"SOP file not found: {self.sop_path}"
            )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )

        self.vector_store = self._build_vector_store()

    def _build_vector_store(self):
        text = self.sop_path.read_text(
            encoding="utf-8"
        )

        sections = self._split_sections(text)

        if not sections:
            raise RuntimeError(
                "No SOP sections were found. "
                "Check the headings in safety_sop.txt."
            )

        return FAISS.from_texts(
            texts=[
                section["content"]
                for section in sections
            ],
            embedding=self.embeddings,
            metadatas=[
                {
                    "section": section["section"],
                    "incident_type": section[
                        "incident_type"
                    ],
                }
                for section in sections
            ],
        )

    def _split_sections(
        self,
        text: str,
    ) -> list[dict]:
        section_map = {
            "MOBILE PHONE USAGE": "phone_usage",
            "PERSONAL PROTECTIVE EQUIPMENT": (
                "ppe_non_compliance"
            ),
            "UNSAFE MACHINE INTERACTION": (
                "unsafe_machine_interaction"
            ),
            "UNSAFE POSITION": "unsafe_position",
            "INCIDENT REPORTING": "incident_reporting",
        }

        sections = []

        current_title = None
        current_lines = []

        for line in text.splitlines():
            stripped = line.strip()

            normalized = re.sub(
                r"^\d+\.\s*",
                "",
                stripped,
            ).strip().upper()

            matched_title = None

            for title in section_map:
                if normalized == title:
                    matched_title = title
                    break

            if matched_title:
                if current_title is not None:
                    sections.append(
                        {
                            "section": current_title,
                            "incident_type": section_map[
                                current_title
                            ],
                            "content": "\n".join(
                                current_lines
                            ).strip(),
                        }
                    )

                current_title = matched_title

                current_lines = [
                    stripped
                ]

            elif current_title is not None:
                current_lines.append(line)

        if current_title is not None:
            sections.append(
                {
                    "section": current_title,
                    "incident_type": section_map[
                        current_title
                    ],
                    "content": "\n".join(
                        current_lines
                    ).strip(),
                }
            )

        return sections

    def retrieve(
        self,
        incident_type: str,
        top_k: int = 1,
    ) -> list[str]:
        results = self.vector_store.similarity_search(
            query=incident_type,
            k=top_k,
            filter={
                "incident_type": incident_type
            },
        )

        return [
            document.page_content
            for document in results
        ]