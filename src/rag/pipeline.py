"""RAG Pipeline: Chunk → Embed → Retrieve → Generate"""

import os
from typing import List, Optional, Dict, Any, Union

from .chunker import chunk_by_semantic_boundaries, chunk_by_tokens, chunk_by_tabs


class RAGPipeline:
    """
    Simple RAG pipeline for long document QA.
    Uses sentence-transformers for embeddings and ChromaDB for vector store.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        llm_model: str = "gpt-4o-mini",
        use_semantic_chunking: bool = True,
        tab_contents: Optional[Dict[str, str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model
        self.use_semantic_chunking = use_semantic_chunking
        self.tab_contents = tab_contents  # When set, use tab-level chunking (fair)

        # Lazy load heavy dependencies
        self._embedding_model = None
        self._chroma_client = None
        self._embedding_model_name = embedding_model

    def _get_embeddings(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _chunk(self, text: str) -> List[str]:
        if self.tab_contents:
            return chunk_by_tabs(self.tab_contents)
        if self.use_semantic_chunking:
            return chunk_by_semantic_boundaries(
                text, self.chunk_size, self.chunk_overlap
            )
        return chunk_by_tokens(text, self.chunk_size, self.chunk_overlap)

    @property
    def collection_overlap(self):
        return self.chunk_overlap

    def index(self, text: str, collection_name: str = "documents") -> None:
        """Index document for retrieval."""
        import chromadb
        from chromadb.config import Settings

        chunks = self._chunk(text)
        if not chunks:
            raise ValueError("No chunks produced from document")

        embeddings_model = self._get_embeddings()
        embeddings = embeddings_model.encode(chunks).tolist()

        self._chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        try:
            self._chroma_client.delete_collection(name=collection_name)
        except:
            pass
        self._collection = self._chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self._collection.add(ids=ids, embeddings=embeddings, documents=chunks)
        self._chunks = chunks

    def retrieve(self, query: str, n_results: Optional[int] = None) -> List[str]:
        """Retrieve top-n chunks (default: top_k)."""
        if not hasattr(self, "_collection"):
            raise ValueError("Call index() before retrieve()")

        n = n_results if n_results is not None else self.top_k
        n = min(n, len(self._chunks))

        embeddings_model = self._get_embeddings()
        query_embedding = embeddings_model.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n,
            include=["documents"]
        )
        return results["documents"][0] if results["documents"] else []

    def generate(self, query: str, context: List[str]) -> str:
        """Generate response using LLM with retrieved context."""
        from ..llm_client import chat_completion

        context_str = "\n\n---\n\n".join(context)
        prompt = f"""Use the following context to answer the question. If the context doesn't contain the answer, say so.
If the question asks for a count or number, respond with only that number (no explanation, no \\boxed{{}}).

Context:
{context_str}

Question: {query}

Answer:"""

        return chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            model=self.llm_model,
            max_tokens=2048,
        )

    def query(
        self,
        text: str,
        query: str,
        collection_name: str = "documents",
        tab_contents: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Full RAG pipeline: index document, retrieve relevant chunks, generate answer.
        Pass tab_contents for tab-level chunking (one chunk per tab, fairer comparison).
        """
        if tab_contents:
            self.tab_contents = tab_contents
        self.index(text, collection_name)
        context = self.retrieve(query)
        return self.generate(query, context)

    def query_pre_indexed(self, query: str) -> str:
        """Query when document is already indexed."""
        context = self.retrieve(query)
        return self.generate(query, context)
