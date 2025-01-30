import logging
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def split_text(state: BaseModel) -> dict:
    __start = time.time()
    logger.info(f"Text splitting started")

    splitter = RecursiveCharacterTextSplitter(chunk_size=state.chunk_size, chunk_overlap=state.chunk_size_overlap)
    chunks = splitter.split_documents([Document(page_content=state.compressed_text)])

    logger.info(f"Text splitting took {int(time.time() - __start)} seconds. Number of chunks: {len(chunks)}")

    return {"chunks": chunks}
