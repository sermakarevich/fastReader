import logging

from app.workflow import get_workflow

logger = logging.getLogger(__name__)


def run(state):
    
    logger.info(f"Running summarization workflow with state: {state}")
    
    workflow = get_workflow()

    out = workflow.invoke(state)

    logger.info(out["short_summary"])
    logger.info(out["extensive_summary"])
