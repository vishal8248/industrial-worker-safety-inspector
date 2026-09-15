from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        loader = TextLoader(
            str(self.sop_path),
            encoding="utf-8",
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(
            documents
        )

        return FAISS.from_documents(
            chunks,
            self.embeddings,
        )

    def retrieve(
        self,
        incident_type: str,
        top_k: int = 2,
    ) -> list[str]:
        query = (
            f"Industrial safety SOP guidance "
            f"for incident type: {incident_type}"
        )

        documents = (
            self.vector_store.similarity_search(
                query,
                k=top_k,
            )
        )

        return [
            document.page_content
            for document in documents
        ]