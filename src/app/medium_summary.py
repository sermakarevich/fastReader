import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.utils import safe_invoke

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}


def llm_medium_summary(state: BaseModel) -> dict:
    __start = time.time()
    logger.info("Medium summary started")

    llm = MODEL_CLASS_TO_CLASS_NAME[state.model_class](model=state.model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give short summary of the `Content` as a single paragraph. "),
            ("human", "`Content`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm

    out = safe_invoke(chain, {"paragraph": state.compressed_text})

    logger.info(f"Medium summary took {int(time.time() - __start)} seconds. Medium summary length: {len(out.content)}")

    return {"medium_summary": out.content}
