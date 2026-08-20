from rag_legal_assistant.eval.corpus import split_articles

LAW = (
    "Art. 1. " + "Przepis pierwszy. " * 20
    + "\nArt. 2a. " + "Przepis drugi. " * 20
    + "\nArt. 3. Za krótki."
)


def test_splits_on_article_headers():
    articles = split_articles(LAW, min_chars=10)

    assert [a["article"] for a in articles] == ["1", "2a", "3"]


def test_article_body_starts_at_its_own_header():
    articles = split_articles(LAW, min_chars=10)

    assert articles[0]["text"].startswith("Art. 1.")
    assert "Art. 2a." not in articles[0]["text"]


def test_min_chars_filters_out_stubs():
    articles = split_articles(LAW, min_chars=250)

    assert [a["article"] for a in articles] == ["1", "2a"]


def test_article_numbers_are_not_confused_with_longer_ones():
    law = "Art. 1. " + "x" * 300 + "\nArt. 11. " + "y" * 300

    assert [a["article"] for a in split_articles(law, min_chars=10)] == ["1", "11"]


def test_plain_text_yields_no_articles():
    assert split_articles("Zwykły tekst bez przepisów. " * 20) == []