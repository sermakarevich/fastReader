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


class DOCUMENT_TYPES:
    PDF = "pdf"
    TEXT = "text"
    YOUTUBE = "youtube"


DOCUMENT_TO_LOADER = {
    DOCUMENT_TYPES.PDF: PDFMinerLoader,
    DOCUMENT_TYPES.TEXT: WebBaseLoader,
    DOCUMENT_TYPES.YOUTUBE: YouTubeVideoLoader,
}


def text_extraction(state: BaseModel) -> dict:
    __start = time.time()

    loader = DOCUMENT_TO_LOADER[state.document_type]

    documents = loader(state.URL).load()
    text = documents[0].page_content
    text = "\n".join([line for line in text.split("\n") if line.strip() != ""])

    logger.info(f"Text extraction took {int(time.time() - __start)} seconds. Text length: {len(text)}")

    return {"original_text": text, "compressed_text": text}
