import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.utils import safe_invoke

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}
            

def llm_extensive_instruction(state: BaseModel) -> dict:
    __start = time.time()
    logger.info("Extensive instructions started")

    llm = MODEL_CLASS_TO_CLASS_NAME[state.model_class](model=state.model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a principal tutor. Your goal is to explain the `Content` in the most clear and logic way."),
            ("human", "`Content`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm

    out = safe_invoke(chain, {"paragraph": state.compressed_text})

    logger.info(
        f"Extensive instructions took {int(time.time() - __start)} seconds. Extensive instructions length: {len(out.content)}"
    )

    return {"extensive_instructions": out.content}
