import asyncio

import pytest

from rag_legal_assistant.config import settings
from rag_legal_assistant.graph import nodes
from rag_legal_assistant.graph.builder import decide_to_generate
from rag_legal_assistant.prompts import NO_CONTEXT_ANSWER


class ChainWasCalled(RuntimeError):
    pass


class ExplodingChain:
    """Stands in for the generator chain and blows up if anything invokes it."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, *args, **kwargs):
        self.calls += 1
        raise ChainWasCalled


def test_no_context_answer_is_usable_as_a_literal():
    assert isinstance(NO_CONTEXT_ANSWER, str)
    assert NO_CONTEXT_ANSWER.strip(), "the refusal text must not be empty"
    assert "{" not in NO_CONTEXT_ANSWER and "}" not in NO_CONTEXT_ANSWER, (
        "braces would be read as template placeholders"
    )


def test_empty_documents_refuse_without_calling_the_llm(monkeypatch):
    stub = ExplodingChain()
    monkeypatch.setattr(nodes, "chain", stub)

    state = {"query": "Jaki jest termin przedawnienia?", "documents": [], "retry_count": 3}
    result = asyncio.run(nodes.generate_answer_node(state))

    assert result == {"answer": NO_CONTEXT_ANSWER}
    assert stub.calls == 0, "the generator chain must not run on an empty context"


def test_stub_chain_is_really_wired_in(monkeypatch):
    """Control case: without it the test above would pass even if patching silently failed."""
    stub = ExplodingChain()
    monkeypatch.setattr(nodes, "chain", stub)

    state = {
        "query": "Jaki jest termin przedawnienia?",
        "documents": [{"source": "kodeks_cywilny.pdf", "text": "Art. 118 ..."}],
        "retry_count": 0,
    }

    with pytest.raises(ChainWasCalled):
        asyncio.run(nodes.generate_answer_node(state))

    assert stub.calls == 1


def test_rewrite_limit_comes_from_settings():
    limit = settings.MAX_QUERY_REWRITES

    assert decide_to_generate({"documents": [], "retry_count": limit - 1}) == "rewrite_query"
    assert decide_to_generate({"documents": [], "retry_count": limit}) == "generate_answer"
