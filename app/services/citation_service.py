"""Application-controlled citation validation and resolution.

The model may only reference evidence ids the application supplied. Anything else is removed.
A valid id proves the cited chunk was supplied, NOT that it supports the claim.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.core.errors import InvalidLLMResponse
from app.services.types import Citation, Evidence

_GROUP = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]", re.IGNORECASE)
_ID = re.compile(r"E\d+", re.IGNORECASE)


@dataclass
class ProcessedAnswer:
    text: str
    citations: list[Citation]
    invalid_ids: list[str]


def parse_llm_output(raw: str) -> tuple[str, bool]:
    """Return (answer, insufficient_evidence) from the model's JSON reply."""
    raw = (raw or "").strip()
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise InvalidLLMResponse()
    flag = data.get("insufficient_evidence", False)
    flag = flag if isinstance(flag, bool) else False
    answer = data["answer"].strip()
    if flag and answer.lower() in ("insufficient_evidence", "insufficient evidence"):
        answer = ""  # the model used its flag as a placeholder; the app supplies a readable message
    if not answer and not flag:
        raise InvalidLLMResponse()
    return answer, flag


def process_answer(answer: str, evidence: list[Evidence]) -> ProcessedAnswer:
    by_id = {e.id.upper(): e for e in evidence}
    invalid: list[str] = []
    used: list[str] = []

    def rewrite(match: re.Match) -> str:
        ids = [i.upper() for i in _ID.findall(match.group(1))]
        good = [i for i in dict.fromkeys(ids) if i in by_id]
        invalid.extend(i for i in ids if i not in by_id)
        for i in good:
            if i not in used:
                used.append(i)
        return f"[{', '.join(good)}]" if good else ""

    cleaned = _GROUP.sub(rewrite, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned).strip()
    return ProcessedAnswer(cleaned, [resolve(by_id[i]) for i in used], list(dict.fromkeys(invalid)))


EXCERPT_CHARS = 600


def _excerpt(text: str) -> str | None:
    text = " ".join(text.split())
    if not text:
        return None
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS].rstrip() + "…"


def resolve(ev: Evidence) -> Citation:
    c, p = ev.chunk, ev.chunk.paper
    return Citation(
        id=ev.id, paper_id=p.id, title=p.title, authors=list(p.authors),
        page_number=c.page_number,  # None stays None; never guessed
        section=c.section, doi=p.doi, source_url=p.source_url, relevance_score=round(c.score, 4),
        excerpt=_excerpt(c.text),
    )
