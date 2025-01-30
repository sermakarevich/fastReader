from typing import List
from pydantic import BaseModel, Field


class State(BaseModel):
    URL: List[str] = Field(None, description="The URL of the document")
    document_type: List[str] = Field(None, description="The type of the document")
    original_text: str = Field(None, description="The original text of the document")
    compressed_text: str = Field(None, description="The compressed text of the document")
    chunks: list = Field(None, description="The list of text chunks")
    compressed_chunk: list = Field(None, description="The list of compressed text chunks")
    short_summary: str = Field(None, description="The short summary of the document")
    medium_summary: str = Field(None, description="The medium summary of the document")
    extensive_summary: str = Field(None, description="The extensive summary of the document")
    formatted_summary: str = Field(None, description="The formatted summary of the document")
    extensive_instructions: str = Field(None, description="The extensive instructions of the document")

    model_class: str = Field("ollama", description="The class of the model used")
    model_name: str = Field("phi4", description="The name of the model used")

    chunk_size: int = Field(2000, description="The size of each chunk")
    chunk_size_overlap: int = Field(50, description="The overlap of each chunk")
    chunk_size_decay: float = Field(0.8, description="The decay rate of the chunk size")
    iteration: int = Field(0, description="The current iteration number")
    target_pre_summary_text_length: int = Field(15000, description="The target length of the pre-summary text")
