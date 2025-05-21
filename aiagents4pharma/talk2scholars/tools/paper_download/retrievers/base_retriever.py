#!/usr/bin/env python3
"""
Base Abstract Class definition to be followed when retrieving paper metadata from different sources.
"""
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from typing import Annotated, Any

from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

class BasePaperRetriever(ABC):
    """
    Abstract base class for implementing paper retrieval tools 
    (e.g., PubMed, arXiv, bioRxiv, medRxiv). 
    Defines a uniform interface for fetching and extracting metadata.
    """

    @abstractmethod
    def load_hydra_configs(self):
        """
        Load required configurations (e.g., URLs) using Hydra.
        """
        pass

    @abstractmethod
    def fetch_metadata(self, url: str, paper_id: str) -> ET.Element:
        """
        Fetch metadata from the given URL using the paper ID.

        Args:
            url (str): The endpoint to fetch metadata from.
            paper_id (str): The unique identifier for the paper.

        Returns:
            ET.Element: Parsed XML metadata root.
        """
        pass

    @abstractmethod
    def extract_metadata(self, xml_root: ET.Element, paper_id: str) -> dict:
        """
        Extract the metadata fields from the XML root.

        Args:
            xml_root (ET.Element): The parsed XML element.
            paper_id (str): The ID of the paper.

        Returns:
            dict: Dictionary containing extracted metadata.
        """
        pass

    @abstractmethod
    def paper_retriever(self, paper_id: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[Any]:
        """
        Unified entry point for retrieving paper metadata and PDF.

        Args:
            paper_id (str): Paper ID (PMC ID, arXiv ID, etc.).
            tool_call_id (str): Injected tool call ID for LangGraph communication.

        Returns:
            Command[Any]: LangGraph-compatible output containing metadata and tool message.
        """
        pass
