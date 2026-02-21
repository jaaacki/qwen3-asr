from pydantic import BaseModel, Field
from typing import Optional

class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the service")
    mode: Optional[str] = Field(None, description="Running mode (e.g., gateway, server)")
    model_loaded: bool = Field(..., description="Whether the model is currently loaded in memory")
    model_id: Optional[str] = Field(None, description="The HuggingFace model ID loaded")
    cuda: Optional[bool] = Field(None, description="Whether CUDA is available")
    gpu_name: Optional[str] = Field(None, description="Name of the GPU")
    gpu_allocated_mb: Optional[int] = Field(None, description="GPU memory currently allocated (MB)")
    gpu_reserved_mb: Optional[int] = Field(None, description="GPU memory currently reserved by PyTorch (MB)")
    worker_alive: Optional[bool] = Field(None, description="Whether the internal worker process is responsive")

class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="The transcribed text")
    language: str = Field(..., description="The language code detected or used for transcription")

class TranslationResponse(BaseModel):
    text: str = Field(..., description="The translated text")
    language: str = Field(..., description="The target language code used (e.g. 'en' or 'zh')")
