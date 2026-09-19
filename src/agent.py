from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question or not self.store or self.store.get_collection_size() == 0:
            return "I could not find relevant information in the knowledge base."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "I could not find relevant information in the knowledge base."

        context_lines = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id") or result.get("id")
            context_lines.append(f"[{index}] Source: {source}\n{result['content']}")

        context = "\n\n".join(context_lines)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say that the information is not available. "
            "Cite relevant source numbers in your answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        answer = self.llm_fn(prompt)
        return str(answer) if answer is not None else "I could not generate an answer from the provided context."
