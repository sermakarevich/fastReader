import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.utils import safe_invoke

logger = logging.getLogger(__name__)

MODEL_CLASS_TO_CLASS_NAME = {"ollama": ChatOllama, "openai": ChatOpenAI}


def llm_extensive_summary(state: BaseModel) -> dict:
    __start = time.time()
    logger.info("Extensive summary started")

    llm = MODEL_CLASS_TO_CLASS_NAME[state.model_class](model=state.model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
You are a comprehensive content analyst and educational writer. Your task is to transform the compressed content I provide into an extensive, thorough summary (approximately 800-1200 words) that fully explores the subject matter while maintaining accuracy to the original content.

## Comprehensive Development Guidelines:
1. Use the compressed content as your foundational framework and source of core information
2. Preserve all key messages, critical points, and essential facts from the original
3. Develop a robust, multi-level structure with clear sections and logical flow
4. Provide in-depth analysis of important concepts, relationships, and implications
5. Add substantial contextual information, multiple examples, and comprehensive supporting details
6. Include relevant historical background, current applications, and future implications when appropriate
7. Analyze connections between concepts and explore underlying principles

## Structural Framework:
- Executive Summary (1 paragraph): Concise overview of the entire topic
- Introduction (1-2 paragraphs): Establish context, significance, and scope of the subject
- Main Body (4-8 paragraphs): Detailed exploration of all key aspects with multiple subsections
- Analysis Section (1-2 paragraphs): Examination of relationships, patterns, or implications
- Practical Applications (1-2 paragraphs): Real-world relevance and implementation considerations
- Conclusion (1-2 paragraphs): Synthesis of information and broader significance

## Development Techniques:
- Explore multiple dimensions of each key concept (historical, practical, theoretical)
- Provide varied examples that illustrate concepts from different perspectives
- Explain technical terms thoroughly and introduce relevant specialized vocabulary
- Draw connections to related fields or topics when it enhances understanding
- Address potential counterpoints, limitations, or areas of debate
- Utilize analogies or metaphors to clarify complex ideas
- Incorporate appropriate statistical data or quantitative information when relevant

## Organizational Elements:
- Use clear headings and subheadings to organize content
- Include an implicit or explicit table of contents through your structure
- Employ paragraph transitions that explicitly connect ideas
- Create a coherent narrative thread throughout the entire summary
- Use bullet points or numbered lists sparingly for complex sequences or collections

## Stylistic Guidelines:
- Maintain a scholarly yet accessible tone appropriate for educated non-specialists
- Vary sentence and paragraph length for rhythm and emphasis
- Balance descriptive, analytical, and evaluative content
- Use precise language and domain-specific terminology where appropriate
- Employ transitional phrases to guide readers through complex discussions

## Output Parameters:
- Total length should be approximately 800-1200 words
- Include 10-15 well-developed paragraphs across all sections
- Maintain proportional development (don't overemphasize one aspect)
- Ensure depth without unnecessary verbosity
- Connect to broader contexts while staying focused on the central topic

Present the extensive summary with appropriate formatting including headings, subheadings, and paragraph breaks to enhance readability and comprehension.
"""),
            ("human", "`Content`:\n```{paragraph}\n```"),
        ]
    )
    chain = prompt | llm

    out = safe_invoke(chain, {"paragraph": state.compressed_text})

    logger.info(
        f"Extensive summary took {int(time.time() - __start)} seconds. Extensive summary length: {len(out.content)}"
    )

    return {"extensive_summary": out.content}
