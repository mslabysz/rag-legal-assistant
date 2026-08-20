import uuid

def point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}"))