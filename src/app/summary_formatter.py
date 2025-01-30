import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.utils import safe_invoke

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}
MODEL_CLASS = "ollama"
MODEL_NAME = "llama3.1"


def llm_summary_formatter(state: BaseModel) -> dict:
    __start = time.time()
    logger.info("Summary formatter started")

    llm = MODEL_CLASS_TO_CLASS_NAME[MODEL_CLASS](model=MODEL_NAME)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You goal is to find the name of the article, given its first 5000 characters in the `Article`. Return ONLY the article name and nothing else."),
            ("human", "`Article`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm

    out = safe_invoke(chain, {"paragraph": state.original_text[:5000]})
    summary_split = state.short_summary.split("</think>")
    if len(summary_split) > 1:
        summary = summary_split[1]
    else:
        summary = summary_split[0]
        
    article_name_split = out.content.split("</think>")
    if len(article_name_split) > 1:
        article_name = article_name_split[1]
    else:
        article_name = article_name_split[0]
    formatted_summary = f"- [**{article_name}**]({state.URL[0]}) {' '.join(summary.split())}"

    logger.info(f"Summary formatter took {int(time.time() - __start)} seconds")

    return {"formatted_summary": formatted_summary}
