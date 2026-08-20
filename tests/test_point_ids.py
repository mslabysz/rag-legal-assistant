import uuid

from rag_legal_assistant.vectordb.ids import point_id


def test_is_deterministic():
    assert point_id("kodeks_cywilny.pdf", 7) == point_id("kodeks_cywilny.pdf", 7)


def test_differs_per_chunk_index():
    assert point_id("kodeks_cywilny.pdf", 7) != point_id("kodeks_cywilny.pdf", 8)


def test_differs_per_source():
    assert point_id("kodeks_cywilny.pdf", 7) != point_id("kodeks_karny.pdf", 7)


def test_is_a_uuid_qdrant_will_accept():
    value = point_id("kodeks_cywilny.pdf", 7)

    assert str(uuid.UUID(value)) == value