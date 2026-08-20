from rag_legal_assistant.chunking.chunker import chunk_document


def test_chunk_metadata_is_sequential():
    chunks = chunk_document("a" * 2000, "kodeks.pdf", chunk_size=300, chunk_overlap=0)

    assert len(chunks) > 1
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c["source"] for c in chunks} == {"kodeks.pdf"}


def test_chunks_do_not_exceed_chunk_size():
    chunks = chunk_document("a" * 5000, "kodeks.pdf", chunk_size=512, chunk_overlap=50)

    assert all(len(c["text"]) <= 512 for c in chunks)


def test_split_prefers_article_boundaries():
    text = "Art. 1. " + "a" * 300 + "\nArt. 2. " + "b" * 300

    chunks = chunk_document(text, "kodeks.pdf", chunk_size=400, chunk_overlap=50)

    assert len(chunks) == 2
    assert chunks[0]["text"].startswith("Art. 1.")
    assert chunks[1]["text"].startswith("Art. 2.")


def test_short_text_yields_single_chunk():
    chunks = chunk_document("Art. 1. Krótki przepis.", "kodeks.pdf", chunk_size=512, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0