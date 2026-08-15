from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_k: int = Field(default=40, ge=0, le=500)
    top_p: float = Field(default=0.95, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)


class ChatRequest(GenerateRequest):
    pass
