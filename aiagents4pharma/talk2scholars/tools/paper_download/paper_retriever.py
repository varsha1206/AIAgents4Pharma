#!/usr/bin/env python3
"""
Tool for downloading paper from available sources like arxiv or pubmed.
"""

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from .download_pubmed_paper import download_pubmedx_paper
from .download_arxiv_input import download_arxiv_paper

class PaperRetrieverInput(BaseModel):
    """Input schema for the paper retriever tool."""
    source: Union[Literal["pubmed", "arxiv"]] = Field(
        default=None,
        description="The source to retrieve the paper from. Options: 'pubmed' or 'arxiv'."
    )
    paper_id: str = Field(..., description="The unique ID of the paper (PMC ID or Arxiv ID)")
    tool_call_id: Annotated[str, InjectedToolCallId]

@tool(args_schema=PaperRetrieverInput, parse_docstring=True)
def paper_retriever(
    source: str,
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Use this tool to search and retrieve papers from given sources like arxiv or pubmed.

    Args:
        source (str): The source to retrieve the paper from arxiv, pubmed or None.
        paper_id (str): The unique id of the paper to be searched.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.

    """
    if source.lower() == "pubmed":
        return download_pubmedx_paper(pmc_id=paper_id, tool_call_id=tool_call_id)

    #elif source.lower() == "arxiv"
    return download_arxiv_paper(arxiv_id=paper_id, tool_call_id=tool_call_id)
