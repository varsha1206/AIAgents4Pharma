#!/usr/bin/env python3
"""
Tool for downloading bioRxiv paper metadata and retrieving the PDF URL.
"""
 
import logging
from typing import Annotated, Any, List
 
import hydra
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from .base_retreiver import BasePaperRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
class DownloadBiorxivPaperInput(BasePaperRetriever):
    """Input schema for the bioRxiv paper download tool."""
 
    # doi: List[str] = Field(description=
    # """List of bioRxiv DOIs, from search_helper or multi_helper or single_helper,
    # used to retrieve paper details and PDF URLs"""
    # )
    # tool_call_id: Annotated[str, InjectedToolCallId]
 
    def __init__(self):
        self.request_timeout=None

    # Helper to load bioRxiv download configuration
    def load_hydra_configs(self) -> Any:
        """Load bioRxiv download configuration."""
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/download_biorxiv_paper=default"]
            )
        return cfg.tools.download_biorxiv_paper
    
    def fetch_metadata(
            self, url: str, paper_id: str
            ) -> dict:
        """
        Fetch metadata for a bioRxiv paper using its DOI and extract relevant fields.
        
        Parameters:
            doi (str): List of DOIs of the bioRxiv paper.
        
        Returns:
            dict: A dictionary containing the title, authors, abstract, publication date, and URLs.
        """
        # Strip any version suffix (e.g., v1) since bioRxiv's API is version-sensitive
        clean_doi = paper_id.split("v")[0]
    
        api_url = f"{url}{clean_doi}"
        logger.info("Fetching metadata from api url: %s", api_url)
        response = requests.get(api_url, timeout=self.request_timeout)
        response.raise_for_status()
    
        data = response.json()
        if not data.get("collection"):
            raise ValueError(f"No metadata found for DOI: {paper_id}")
    
        data = response.json()
    
        return data["collection"][0]
    
    def extract_metadata(self,paper: dict, doi: str) -> dict:
        """
        Extract relevant metadata fields from a bioRxiv paper entry.
        """
        title = paper.get("title", "")
        authors = paper.get("authors", "")
        abstract = paper.get("abstract", "")
        pub_date = paper.get("date", "")
        doi_suffix = paper.get("doi", "").split("10.1101/")[-1]
        pdf_url = f"https://www.biorxiv.org/content/10.1101/{doi_suffix}.full.pdf"
        logger.info("PDF URL: %s", pdf_url)
        if not pdf_url:
            raise ValueError(f"No PDF URL found for DOI: {doi}")
        return {
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Publication Date": pub_date,
            "URL": pdf_url,
            "pdf_url": pdf_url,
            "filename": f"{doi_suffix}.pdf",
            "source": "biorxiv",
            "biorxiv_id": doi
        }
    
    def _get_snippet(self,abstract: str) -> str:
        """Extract the first one or two sentences from an abstract."""
        if not abstract or abstract == "N/A":
            return ""
        sentences = abstract.split(". ")
        snippet_sentences = sentences[:2]
        snippet = ". ".join(snippet_sentences)
        if not snippet.endswith("."):
            snippet += "."
        return snippet
    
    def _build_summary(self,article_data: dict[str, Any]) -> str:
        """Build a summary string for up to three papers with snippets."""
        top = list(article_data.values())[:3]
        lines: list[str] = []
        for idx, paper in enumerate(top):
            title = paper.get("Title", "N/A")
            pub_date = paper.get("Publication Date", "N/A")
            url = paper.get("URL", "")
            snippet = self._get_snippet(paper.get("Abstract", ""))
            line = f"{idx+1}. {title} ({pub_date})"
            if url:
                line += f"\n   View PDF: {url}"
            if snippet:
                line += f"\n   Abstract snippet: {snippet}"
            lines.append(line)
        summary = "\n".join(lines)
        return (
            "Download was successful. Papers metadata are attached as an artifact. "
            "Here is a summary of the results:\n"
            f"Number of papers found: {len(article_data)}\n"
            "Top 3 papers:\n" + summary
        )
    
    # @tool(
    #     args_schema=DownloadBiorxivPaperInput,
    #     parse_docstring=True,
    # )
    def paper_retriever(
        self,
        paper_ids: List[str],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Any]:
        """
        Get metadata and PDF URLs for one or more bioRxiv papers using their unique dois.
        """
        logger.info("Fetching metadata from arXiv for paper IDs: %s", paper_ids)
    
        # Load configuration
        config = self.load_hydra_configs()
        api_url = config.api_url
        self.request_timeout = config.request_timeout
    
        # Aggregate results
        article_data: dict[str, Any] = {}
        for doi in paper_ids:
            logger.info("Processing DOI: %s", doi)
            # Fetch metadata
            entry = self.fetch_metadata(doi, api_url)
            if entry is None:
                logger.warning("No entry found for bioRxiv ID %s", doi)
                continue
            # Extract relevant metadata
            article_data = self.extract_metadata(entry, doi)
    
        # Build and return summary
        content = self._build_summary(article_data)
        return Command(
            update={
                "article_data": article_data,
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        artifact=article_data,
                    )
                ],
            }
        )
 