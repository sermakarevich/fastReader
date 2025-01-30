import logging
import time

from langchain_community.document_loaders import PDFMinerLoader, WebBaseLoader
from langchain_core.documents import Document
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


class YouTubeVideoLoader:
    def __init__(self, URL: str):
        self.video_id = URL.split("v=")[1]

    def load(self) -> Document:
        transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
        text = " ".join([line["text"] for line in transcript])
        return [Document(text)]
    

class LocalDocumentLoader:
    def __init__(self, URL: str):
        self.URL = URL

    def load(self) -> Document:
        with open(self.URL, "r") as f:
            text = f.read()
        return [Document(text)]


class DOCUMENT_TYPES:
    PDF = "pdf"
    TEXT = "text"
    YOUTUBE = "youtube"
    LOCAL = "local"


DOCUMENT_TO_LOADER = {
    DOCUMENT_TYPES.PDF: PDFMinerLoader,
    DOCUMENT_TYPES.TEXT: WebBaseLoader,
    DOCUMENT_TYPES.YOUTUBE: YouTubeVideoLoader,
    DOCUMENT_TYPES.LOCAL: LocalDocumentLoader,
}


def text_extraction(state: BaseModel) -> dict:
    __start = time.time()
    logger.info("Text extraction started")

    text = ""
    
    for document_type, URL in zip(state.document_type, state.URL):

        loader = DOCUMENT_TO_LOADER[document_type]

        documents = loader(URL).load()
        page_content = documents[0].page_content
        if len(text) > 0:
            text += "\n\n"
        text += "\n".join([line for line in page_content.split("\n") if line.strip() != ""])
        
        logger.info(f"Text extraction for URL {URL} took {int(time.time() - __start)} seconds. Text length: {len(text)}")

    logger.info(f"Text extraction took {int(time.time() - __start)} seconds. Text length: {len(text)}")

    return {"original_text": text, "compressed_text": text}
