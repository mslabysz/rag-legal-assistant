import asyncio
from rag_legal_assistant.graph.builder import app

async def main():
    print("Starting stream...")
    async for event in app.astream_events({"query": "termin przedawnienia"}, version="v2"):
        kind = event["event"]
        name = event["name"]
        print(f"EVENT: {kind} | NAME: {name}")
        
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(f"  CHUNK: {chunk.content}")

if __name__ == "__main__":
    asyncio.run(main())
