import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from app.utils import safe_invoke
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}


def llm_compress_text(state: BaseModel) -> dict:
    __start = time.time()
    logger.info(f"Text compression started")

    logger.info(f"Text compression: Iteration: {state.iteration}, Chunk size: {state.chunk_size}")

    llm = MODEL_CLASS_TO_CLASS_NAME[state.model_class](model=state.model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give 1 short sentence summary of the main idea in the `Content` in English."),
            ("human", "`Content`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        futures = [executor.submit(safe_invoke, chain, {"paragraph": paragraph.page_content}) for paragraph in state.chunks]
        out = [future.result() for future in futures]
        
    out_split = [i.content.split("</think>") for i in out]
    out = [i[1] if len(i) > 1 else i[0] for i in out_split]
    out = [" ".join(i.split(" ")) for i in out]
    compressed_text = "\n".join(out)

    iteration = state.iteration + 1
    chunk_size = int(state.chunk_size * state.chunk_size_decay)

    logger.info(
        f"Compression took {int(time.time() - __start)} seconds. Compressed text length: {len(compressed_text)}"
    )

    return {"compressed_text": compressed_text, "iteration": iteration, "chunk_size": chunk_size}
