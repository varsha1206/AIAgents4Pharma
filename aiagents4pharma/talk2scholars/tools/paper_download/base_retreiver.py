#!/usr/bin/env python3
"""
Base Abstract Class definition to be followed when retrieving paper metadata from different sources.
"""
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from typing import Annotated, Any, List

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
    def fetch_metadata(self,url: str, paper_id: str) -> dict:
        """
        Fetch metadata from the given URL using the paper ID.

        Args:
            url (str): The endpoint to fetch metadata from.
            paper_id (str): The unique identifier for the paper.

        Returns:
            dict: Dictionary containing the metadata.
        """
        pass

    @abstractmethod
    def extract_metadata(self,data: dict, paper_id: str) -> dict:
        """
        Extract the metadata fields from the XML root.

        Args:
            data (dict): The data from fetch_metadata.
            paper_id (str): The ID of the paper.

        Returns:
            dict: Dictionary containing extracted metadata.
        """
        pass

    @abstractmethod
    def _get_snippet(self,abstract: str) -> str:
        """Extract the first one or two sentences from an abstract.
        
        Args:
            abstract (str): The Abstract of the paper extracted

        Returns:
            str: Returns the first one or two lines of the abstract
        """
        pass

    @abstractmethod
    def _build_summary(self,article_data: dict[str, Any]) -> str:
        """
        Build a summary for upto three papers by adding their snippets

        Args:
            article_data (dict): The metadata of the articles which have been extracted

        Returns:
            str: Summary snippet for each of the top 3 papers
        """
        pass

    @abstractmethod
    def paper_retriever(self, paper_ids: List[str], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[Any]:
        """
        Unified entry point for retrieving paper metadata and PDF.

        Args:
            paper_id List(str): Paper IDs (arXiv ID, DOIs, etc.).
            tool_call_id (str): Injected tool call ID for LangGraph communication.

        Returns:
            Command[Any]: LangGraph-compatible output containing metadata and tool message.
        """
        pass