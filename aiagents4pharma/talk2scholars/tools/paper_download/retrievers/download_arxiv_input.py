#!/usr/bin/env python3
"""
Functionality for downloading arXiv paper metadata and retrieving the PDF URL.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Annotated, Any

import hydra
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from ..utils.arxiv_utils import fetch_arxiv_metadata, extract_arxiv_metadata
from .base_retriever import BasePaperRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivRetriever(BasePaperRetriever):
    """Schema to retrieve paper metadata from Arxiv"""
    def __init__(self):
        self.api_url = None
        self.request_timeout = None
        self.ns = {"atom": "http://www.w3.org/2005/Atom"}

    def load_hydra_configs(self):
        """Load Hydra Configurations"""
        logger.info("Loading Hydra configs for arXiv retriever")
        with hydra.initialize(version_base=None, config_path="../../../configs"):
            cfg = hydra.compose(
                config_name="config",
                overrides=["tools/download_arxiv_paper=default"],
            )
            self.api_url = cfg.tools.download_arxiv_paper.api_url
            self.request_timeout = cfg.tools.download_arxiv_paper.request_timeout

    def fetch_metadata(
        self,
        url: str,
        paper_id: str
    ) -> ET.Element:
        """Fetch and parse metadata from the API url"""
        # Using the helper util to fetch metadata XML
        logger.info(f"Fetching metadata XML for arXiv ID: {paper_id} from URL: {url}")
        return fetch_arxiv_metadata(url, paper_id, self.request_timeout)

    def extract_metadata(
        self,
        xml_root: ET.Element,
        paper_id: str
    ) -> dict:
        """Extract metadata from the XML entry."""
        # Delegate to helper util to extract metadata dict
        entry = xml_root.find("atom:entry", self.ns)
        if entry is None:
            raise ValueError(f"No metadata entry found for arXiv ID: {paper_id}")
        return extract_arxiv_metadata(entry, self.ns, paper_id)

    def paper_retriever(
        self,
        paper_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Any]:
        """
        Get metadata and PDF URL for a paper from arxiv using unique arxiv ID.
        """
        logger.info(f"Starting arXiv paper retrieval for ID: {paper_id}")
        
        # Load configs
        self.load_hydra_configs()

        # Fetch metadata XML from arXiv API
        xml_root = self.fetch_metadata(self.api_url, paper_id)

        # Extract metadata dictionary
        metadata = self.extract_metadata(xml_root, paper_id)

        article_data = {paper_id: metadata}
        content = f"Successfully retrieved metadata and PDF URL for arXiv ID {paper_id}"
        status = {paper_id: {"paper_download _status":"success","source":"arxiv"}}
        return Command(
            update={
                "paper_download_status": status,
                "article_data": article_data,
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )
