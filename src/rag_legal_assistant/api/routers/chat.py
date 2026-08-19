import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from rag_legal_assistant.api.schemas import ChatRequest, ChatResponse
from rag_legal_assistant.graph.builder import app as agent_app

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    logger.info(f"Received query: {request.query}")

    try:
        final_state = await agent_app.ainvoke({"query": request.query, "retry_count": 0})
        return ChatResponse(answer=final_state["answer"], retries=final_state.get("retry_count", 0))
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def event_generator():
        try:
            is_generating = False
            retry_count = 0
            # Słownik (mapa) zdarzeń do komunikatów, by uniknąć powtarzania yieldów
            status_messages = {
                "retrieve": "Szukam dokumentów w bazie...",
                "grade_documents": "Sędzia ocenia przydatność dokumentów...",
                "rewrite_query": "Odrzucono dokumenty. Przepisuję zapytanie i ponawiam próbę...",
                "generate_answer": "Piszę odpowiedź..."
            }

            async for event in agent_app.astream_events(
                    {"query": request.query,
                     "retry_count": 0,
                     "filter_document": request.filter_document
                     },
                    version="v2"
            ):
                kind = event["event"]
                name = event["name"]

                match (kind, name):
                    case ("on_chain_start", n) if n in status_messages:
                        if n == "rewrite_query":
                            retry_count += 1
                        
                        yield f"data: {json.dumps({'type': 'status', 'message': status_messages[n]})}\n\n"

                        if n == "generate_answer":
                            is_generating = True
                            state_input = event.get("data", {}).get("input", {})
                            if isinstance(state_input, dict) and "documents" in state_input:
                                yield f"data: {json.dumps({'type': 'sources', 'documents': state_input['documents']})}\n\n"

                    case ("on_chain_end", "generate_answer"):
                        is_generating = False

                    case ("on_chat_model_stream", _) if is_generating:
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.content})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'retries': retry_count})}\n\n"

        except Exception as e:
            logger.error(f"Error during agent stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")