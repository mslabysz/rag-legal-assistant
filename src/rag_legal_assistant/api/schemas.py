from pydantic import BaseModel, ConfigDict, Field

class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's legal question or prompt",
        examples=["Jaki jest termin przedawnienia?"],
    )
    filter_document: str | None = Field(default=None, max_length=255)

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated response from the LLM")
    retries: int = Field(0, description="The number of rewrite iterations performed by the Agent")