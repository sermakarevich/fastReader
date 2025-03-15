import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.utils import safe_invoke

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}



def llm_summary_formatter(state: BaseModel) -> dict:
    __start = time.time()
    logger.info("Summary formatter started")

    llm = MODEL_CLASS_TO_CLASS_NAME[state.model_class_summary_formatter](model=state.model_name_summary_formatter)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
             You goal is to find the name of the article, given its first 5000 characters in the `Article`.
             Return ONLY the article name and nothing else.
             DO NOT include any explanation and empty line in your response.
             """),
            ("human", "`Article`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm

    out = safe_invoke(chain, {"paragraph": state.original_text[:5000]})

    article_name_split = out.content.split("</think>")
    if len(article_name_split) > 1:
        article_name = article_name_split[1]
    else:
        article_name = article_name_split[0]
    formatted_summary = f"- [**{article_name}**]({state.URL[0]})"

    logger.info(f"Summary formatter took {int(time.time() - __start)} seconds")

    return {"formatted_summary": formatted_summary}
