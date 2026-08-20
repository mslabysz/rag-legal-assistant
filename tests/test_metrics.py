from rag_legal_assistant.eval.metrics import hit_rank


def doc(source: str, text: str) -> dict:
    return {"source": source, "text": text}


def test_returns_position_of_first_matching_chunk():
    docs = [
        doc("kc.pdf", "Art. 5. Nie można czynić ze swego prawa użytku..."),
        doc("kc.pdf", "Art. 118. Termin przedawnienia wynosi sześć lat."),
    ]

    assert hit_rank(docs, "kc.pdf", "118") == 2


def test_returns_none_when_article_absent():
    docs = [doc("kc.pdf", "Art. 5. Nie można czynić ze swego prawa użytku...")]

    assert hit_rank(docs, "kc.pdf", "118") is None


def test_source_must_match():
    docs = [doc("kk.pdf", "Art. 118. Kto, w celu wyniszczenia grupy...")]

    assert hit_rank(docs, "kc.pdf", "118") is None


def test_does_not_match_longer_article_number():
    docs = [doc("kc.pdf", "Art. 1181. Zupełnie inny przepis.")]

    assert hit_rank(docs, "kc.pdf", "118") is None


def test_does_not_match_lettered_variant():
    docs = [doc("kc.pdf", "Art. 118a. Zupełnie inny przepis.")]

    assert hit_rank(docs, "kc.pdf", "118") is None


def test_continuation_chunk_is_not_a_hit():
    docs = [doc("kc.pdf", "...ciąg dalszy przepisu, por. Art. 118 powyżej.")]

    assert hit_rank(docs, "kc.pdf", "118") is None