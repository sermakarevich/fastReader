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
            # ("system", "Give 1 short sentence summary of the main idea in the `Content` in English."),
            ("system", """
You are a precision text compression specialist. Your task is to compress any text I provide into 1-2 concise sentences that capture the most essential information.

## Compression Guidelines:
1. Identify the central claim, conclusion, or main point of the text
2. Preserve key facts, figures, and relationships that support the main point
3. Maintain critical context necessary for understanding
4. Prioritize unique insights over general background information
5. Remove redundancies, examples, elaborations, and tangential points
6. Preserve causal relationships and important conditional statements
7. Keep specialized terminology only when essential to meaning

## Output Format:
- Provide exactly 1-2 complete sentences
- Use clear, direct language
- Maintain factual accuracy
- Preserve the original tone (formal/informal) when possible
- Do not use ellipses, bullet points, or other formatting

Always compress the text without adding interpretations, opinions, or information not contained in the original text.
"""),
            ("human", "`Content`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
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
