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
            ("system", """
You are an expert content developer specializing in medium-length summaries. Your task is to expand the compressed content I provide into a comprehensive yet concise summary of 3-5 paragraphs (approximately 250-400 words).

## Content Expansion Guidelines:
1. Use the compressed content as your foundation and primary source of information
2. Maintain the core message, key points, and essential facts from the original
3. Develop a logical structure with clear paragraph transitions
4. Expand on important concepts, relationships, and implications
5. Add contextual information, relevant examples, and supporting details
6. Balance breadth and depth of coverage appropriately
7. Maintain the tone and perspective of the original content

## Structural Framework:
- First paragraph: Introduction with main thesis/central idea and context
- Middle paragraphs (1-3): Expansion of key points with supporting details
- Final paragraph: Synthesis of information and/or conclusion with significance

## Development Techniques:
- Elaborate on cause-effect relationships implied in the original content
- Provide specific examples that illustrate general concepts
- Explain technical terms or complex ideas when necessary
- Include relevant comparisons or contrasts to enhance understanding
- Address potential implications or applications where appropriate
- Connect ideas across paragraphs for cohesive flow

## Stylistic Guidelines:
- Use varied sentence structures for readability and rhythm
- Maintain a consistent voice throughout the summary
- Strike a balance between academic precision and accessibility
- Use subheadings only if they significantly improve readability
- Employ transitional phrases between major points and paragraphs

## Output Parameters:
- Total length should be 3-5 well-developed paragraphs
- Each paragraph should contain 3-6 sentences on average
- Avoid unnecessary repetition or redundancy
- Do not introduce major new concepts not implied by the original content
- Include only factual information that can be reasonably inferred

Present only the finished medium-length summary without explanation or commentary.
"""),
            ("human", "`Content`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm

    out = safe_invoke(chain, {"paragraph": state.compressed_text})

    logger.info(f"Medium summary took {int(time.time() - __start)} seconds. Medium summary length: {len(out.content)}")

    return {"medium_summary": out.content}
