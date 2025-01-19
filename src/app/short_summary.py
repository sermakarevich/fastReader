import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}


def llm_short_summary(state: BaseModel) -> dict:
    __start = time.time()

    llm = MODEL_CLASS_TO_CLASS_NAME[state.model_class](model=state.model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give 3 short sentences summary of the `Content` as a single paragraph. "),
            ("human", "`Content`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm

    out = chain.invoke({"paragraph": state.compressed_text})

    logger.info(f"Short summary took {int(time.time() - __start)} seconds. Short summary length: {len(out.content)}")

    return {"short_summary": out.content}
