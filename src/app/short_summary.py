import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.utils import safe_invoke

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}


def llm_short_summary(state: BaseModel) -> dict:
    __start = time.time()
    logger.info(f"Short summary started")

    llm = MODEL_CLASS_TO_CLASS_NAME[state.model_class_summary_formatter](model=state.model_name_summary_formatter)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

You are a skilled micro-narrative creator. Your task is to transform the compressed content I provide into an engaging short story or summary of 5-10 sentences.

## Content Expansion Guidelines:
1. Use the compressed content as your core narrative foundation
2. Expand on the main elements without contradicting the original information
3. Add appropriate context, setting, and sensory details to bring the content to life
4. Develop a clear beginning, middle, and end structure
5. Include a character perspective or narrative voice that fits the content
6. Maintain the original tone and intent of the compressed content
7. Ensure the story flows naturally between sentences

## Creative Parameters:
- Keep the total length between 5-10 complete sentences
- Use varied sentence structures for rhythm and readability
- Include at least one descriptive element that appeals to the senses
- Incorporate subtle emotional elements where appropriate
- Conclude with a sentence that provides closure or a key takeaway

## Output Restrictions:
- Do not add major plot elements not implied by the original content
- Avoid excessive exposition or background information
- Do not introduce multiple storylines or complex subplots
- Stay within the world/context established by the compressed content

Present only the finished short story or summary without explanation or commentary.
"""),
            ("human", """
`Content`:\n```{paragraph}\n```"""),
        ]
    )

    chain = prompt | llm

    out = safe_invoke(chain, {"paragraph": state.compressed_text})

    logger.info(f"Short summary took {int(time.time() - __start)} seconds. Short summary length: {len(out.content)}")

    return {"short_summary": out.content}
