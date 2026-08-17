from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)

info = client.get_collection("legal_docs")
print(f"Punktów w kolekcji: {info.points_count}")

from qdrant_client.models import Filter, FieldCondition, MatchValue
for source in ["kodeks_cywilny.pdf", "prawo_budowlane.pdf", "gospodarka_nieruchomościami.pdf"]:
    count = client.count(
        collection_name="legal_docs",
        count_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )
    print(f"  {source}: {count.count} chunks")