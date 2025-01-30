import logging

logger = logging.getLogger(__name__)

RETRIES = 3

def safe_invoke(chain, data):
    for _ in range(RETRIES):
        try:
            return chain.invoke(data)
        except Exception as e:
            logger.error(f"Error: {e}")