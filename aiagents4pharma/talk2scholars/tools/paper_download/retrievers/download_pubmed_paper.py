#!/usr/bin/env python3
"""
Functionality for downloading PubMedX paper metadata and retrieving the PDF URL.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Annotated, Any

import hydra
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from .base_retriever import BasePaperRetriever
from ..utils.pubmed_utils import map_ids, fetch_pubmed_metadata, extract_pubmed_metadata

logger = logging.getLogger(__name__)

class PubMedRetriever(BasePaperRetriever):
    """Schema to retrieve paper metadata from PubMed Central"""
    def __init__(self):
        self.metadata_url= None
        self.pdf_base_url= None
        self.map_url= None

    def load_hydra_configs(self):
        """Load Hydra Configurations"""
        logger.info("Loading Hydra Configs for PubMedRetriever")
        with hydra.initialize(version_base=None, config_path="../../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/download_pubmed_paper=default"]
            )
            self.metadata_url = cfg.tools.download_pubmed_paper.metadata_url
            self.pdf_base_url = cfg.tools.download_pubmed_paper.pdf_base_url
            self.map_url = cfg.tools.download_pubmed_paper.map_url

    def fetch_metadata(
        self,
        url: str,
        paper_id: str
    ) -> ET.Element:
        """Fetch and parse metadata from the API url"""
        logger.info("Fetching metadata for paper id %s",paper_id)
        return fetch_pubmed_metadata(url, paper_id)

    def extract_metadata(
        self,
        xml_root: ET.Element,
        paper_id: str
    ) -> dict:
        """Extract metadata from the XML entry."""
        logger.info("Extracting metadata for paper id %s",paper_id)
        return extract_pubmed_metadata(xml_root, paper_id, self.pdf_base_url)

    def paper_retriever(
        self,
        paper_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command[Any]:
        """
        Get metadata and PDF URL for an pubmed paper using its unique PMC ID.
        """
        self.load_hydra_configs()
        paper_id = map_ids(paper_id, self.map_url)

        xml_root = self.fetch_metadata(self.metadata_url, paper_id)
        metadata = self.extract_metadata(xml_root, paper_id)
        logger.info("Metadata successfully extracted for paper %s",paper_id)

        article_data = {paper_id: metadata}
        content = f"Successfully retrieved metadata and PDF URL for PMC ID {paper_id}"
        return Command(
            update={
                "article_data": article_data,
                "messages": [ToolMessage(content, tool_call_id=tool_call_id)],
            }
        )
    