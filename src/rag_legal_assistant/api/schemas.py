from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's legal question or prompt", examples=["Jaki jest termin przedawnienia?"])


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated response from the LLM")
    retries: int = Field(0, description="The number of rewrite iterations performed by the Agent")