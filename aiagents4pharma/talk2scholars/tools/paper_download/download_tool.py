# tools/download_tool.py
"""
Tool to download paper from Pubmed or Arxiv
"""
import logging
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
import re

from .retrievers.download_arxiv_input import ArxivRetriever
from .retrievers.download_pubmed_paper import PubMedRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_arxiv_id(paper_id: str) -> bool:
    modern_pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
    legacy_pattern = r"^[a-z\-]+\/\d{7}(v\d+)?$"
    return re.match(modern_pattern, paper_id) or re.match(legacy_pattern, paper_id)

class DownloadPaperInput(BaseModel):
    source: Union[Literal["arxiv", "pubmed"]] = Field(
        default=None,
        description="The source to retrieve the paper from. Options: 'pubmed' or 'arxiv'."
    )
    paper_id: str = Field(description="The ID of the paper (e.g., arXiv ID, PMC ID)")
    tool_call_id: Annotated[str, InjectedToolCallId]

@tool(args_schema=DownloadPaperInput)
def download_paper(paper_id: str, source: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Download a paper given its ID and source (e.g., arXiv or PubMed).
    """
    logger.info("Inside paper download tool")
    if is_arxiv_id(paper_id) or source=='arxiv':
        logger.info("Getting into Arxiv")
        retriever = ArxivRetriever()
    elif paper_id.lower().startswith("pmc") or paper_id.lower().startswith("pm") or source=='pubmed':
        logger.info("Getting into Pubmed")
        retriever = PubMedRetriever()

    return retriever.paper_retriever(paper_id=paper_id, tool_call_id=tool_call_id)
