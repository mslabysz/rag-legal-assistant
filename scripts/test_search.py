from rag_legal_assistant.retrieval.retriever import search

queries = [
    "Jaki jest termin przedawnienia roszczeń?",
    "Kto wydaje pozwolenie na budowę?",
    "Czym jest księga wieczysta?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    results = search(q, top_k=3)
    for r in results:
        print(f"\n  Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"  Text: {r['text'][:150]}...")