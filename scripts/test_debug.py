from rag_legal_assistant.retrieval.retriever import search

results = search("Jaki jest termin przedawnienia roszczeń?", top_k=20)
print("looking for chunk in top 20:")
for i, r in enumerate(results):
    has_keyword = "przedawni" in r["text"].lower()
    marker = " ◄◄◄ found!" if has_keyword else ""
    print(f"  {i+1}. Score: {r['score']:.4f} | {r['source'][:30]} | {'...'+r['text'][:80]+'...'}{marker}")