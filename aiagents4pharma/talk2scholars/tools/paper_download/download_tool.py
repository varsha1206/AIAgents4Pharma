# tools/download_tool.py
"""
Tool to download paper from Arxiv,Biorxiv, Medrxiv
"""
import logging
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from typing import Annotated, Literal, List
from pydantic import BaseModel, Field
import re

from .download_arxiv_input import DownloadArxivPaperInput
from .download_biorxiv_input import DownloadBiorxivPaperInput
from .download_medrxiv_input import DownloadMedrxivPaperInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadSourceDetermination(BaseModel):
    """
    Structured output schema for the source of the paper to be downloaded
    - the source is "arxiv" for arxiv ids and biorxiv for DOIs 
    """
    source: Literal["arxiv", "biorxiv"]


class DownloadPaperInput(BaseModel):
    paper_id: List[str] = Field(description="List of paper ids (e.g., arXiv ID, DOIs)")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

@tool(args_schema=DownloadPaperInput, return_direct=True)
def download_paper(paper_id: List[str], tool_call_id: Annotated[str, InjectedToolCallId],state: Annotated[dict, InjectedState]):
    """
    Download a paper given its ID and source (e.g., arXiv or Biorxiv).
    """
    logger.info("Inside paper download tool")
    llm_model = state.get("llm_model")
    structured_llm = llm_model.with_structured_output(DownloadSourceDetermination)
    source_obj = structured_llm.invoke([f"The paper ids are {paper_id}. Identify the source (arxiv or biorxiv)."])
    logger.info("Source chosen by LLM: %s",source_obj.source)
    if source_obj.source =='arxiv':
        logger.info("Getting into arxiv")
        retriever = DownloadArxivPaperInput()
    elif source_obj.source=='biorxiv':
        logger.info("Getting into bioarxiv")
        retriever = DownloadBiorxivPaperInput()
    else:
        logger.info("Getting into medrxiv")
        retriever = DownloadMedrxivPaperInput()

    return retriever.paper_retriever(paper_ids=paper_id, tool_call_id=tool_call_id)