from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy

from app.compress_text import llm_compress_text
from app.extensive_summary import llm_extensive_summary
from app.short_summary import llm_short_summary
from app.medium_summary import llm_medium_summary
from app.summary_formatter import llm_summary_formatter
from app.state import State
from app.text_extraction import text_extraction
from app.text_splitter import split_text
from app.extensive_instruction import llm_extensive_instruction


def compression_cycles(state: State):
    if len(state.compressed_text) > state.target_pre_summary_text_length:
        return "split_text"
    return ["llm_extensive_summary", "llm_short_summary", "llm_medium_summary", "llm_extensive_instruction"]


def get_workflow():
    retry = RetryPolicy(max_attempts=3)

    workflow = StateGraph(state_schema=State)

    workflow.add_node(text_extraction)
    workflow.add_node(split_text)
    workflow.add_node(llm_compress_text, retry=retry)
    workflow.add_node(llm_extensive_summary)
    workflow.add_node(llm_short_summary)
    workflow.add_node(llm_medium_summary)
    workflow.add_node(llm_summary_formatter)
    workflow.add_node(llm_extensive_instruction)

    workflow.add_edge(START, "text_extraction")
    workflow.add_conditional_edges(
        "text_extraction", compression_cycles, ["llm_extensive_summary", "llm_short_summary", "llm_medium_summary", "split_text", "llm_extensive_instruction"]
    )
    workflow.add_edge("split_text", "llm_compress_text")
    workflow.add_conditional_edges(
        "llm_compress_text", compression_cycles, ["llm_extensive_summary", "llm_short_summary", "llm_medium_summary", "split_text", "llm_extensive_instruction"]
    )
    workflow.add_edge("llm_extensive_summary", END)
    workflow.add_edge("llm_short_summary", "llm_summary_formatter")
    workflow.add_edge("llm_summary_formatter", END)
    workflow.add_edge("llm_medium_summary", END)
    workflow.add_edge("llm_extensive_instruction", END)

    return workflow.compile()
