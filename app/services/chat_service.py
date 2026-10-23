"""RAG orchestration: retrieve -> select evidence -> generate -> validate citations."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.citation_service import parse_llm_output, process_answer
from app.services.evidence_service import select_evidence
from app.services.prompts import build_messages
from app.services.retrieval_service import RetrievalService
from app.services.types import ChatModel, ChatResult, SearchFilters

log = logging.getLogger(__name__)

NO_CORPUS = "No papers are indexed yet, so I cannot answer from the research corpus."
NO_EVIDENCE = "The indexed papers do not contain sufficiently relevant evidence to answer this question."


@dataclass(frozen=True)
class RagConfig:
    threshold: float
    default_top_k: int
    evidence_budget_chars: int
    overfetch_factor: int = 3


class ChatService:
    def __init__(self, retrieval: RetrievalService, llm: ChatModel, config: RagConfig):
        self.retrieval, self.llm, self.cfg = retrieval, llm, config

    def _insufficient(self, text: str, retrieved: int = 0, warnings: list[str] | None = None) -> ChatResult:
        return ChatResult(text, [], retrieved, self.llm.model, True, warnings or [])

    async def answer(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None) -> ChatResult:
        filters = filters or SearchFilters()
        top_k = top_k or self.cfg.default_top_k
        candidates = await self.retrieval.retrieve(question, top_k * self.cfg.overfetch_factor, filters)
        if not candidates and not await self.retrieval.corpus_has_chunks():
            return self._insufficient(NO_CORPUS)

        evidence, stats = select_evidence(
            candidates, threshold=self.cfg.threshold, max_items=top_k, budget_chars=self.cfg.evidence_budget_chars
        )
        if not evidence:
            return self._insufficient(NO_EVIDENCE, retrieved=len(candidates))

        warnings = ["evidence_truncated_to_fit_context"] if stats.truncated else []
        raw = await self.llm.chat(build_messages(question, evidence))  # no automatic retry on timeout
        answer, model_says_insufficient = parse_llm_output(raw)
        if model_says_insufficient and not answer:
            return self._insufficient(NO_EVIDENCE, retrieved=len(evidence), warnings=warnings)
        processed = process_answer(answer, evidence)

        if processed.invalid_ids:
            log.warning("removed invalid citation ids from model output: %s", processed.invalid_ids)
            warnings.append("invalid_citations_removed")
        if model_says_insufficient:
            return ChatResult(processed.text, [], len(evidence), self.llm.model, True, warnings)
        if not processed.citations:
            warnings.append("no_valid_citations")
        return ChatResult(processed.text, processed.citations, len(evidence), self.llm.model, False, warnings)
