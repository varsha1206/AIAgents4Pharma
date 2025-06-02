#!/usr/bin/env python3
"""
Tool for downloading PubMed paper metadata and retrieving the PDF URL.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Annotated, Any

import hydra
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pubmed_config():
    """Fetching Hydra configurations"""
    logger.info("Loading Hydra Configs:")
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_pubmed_paper=default"]
        )
        return (
            cfg.tools.download_pubmed_paper.metadata_url,
            cfg.tools.download_pubmed_paper.pdf_base_url,
            cfg.tools.download_pubmed_paper.map_url
        )
class DownloadPubMedInput(BaseModel):
    """Input schema for the pubmed paper download tool with it's PM or PMC ID."""

    paper_id: str = Field(
        description="The unique ID used to retrieve the paper details and PDF URL."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

def map_ids(
        input_id: str,
        map_url: str,
        ) -> str:
    """Fetch the PMC ID of paper given other IDs for Pubmed retrieval"""
    response = requests.get(f"{map_url}?ids={input_id}", timeout=10)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    if((root.find('record') is not None) and (root.find('record').attrib.get('pmcid'))):
        logger.info("Retrieved PMC ID for the given id %s",input_id)
        return root.find('record').attrib.get('pmcid')
    raise RuntimeError(f"PMC prefix missing for {input_id}")

def fetch_metadata(
    metadata_url: str, paper_id: str
) -> ET.Element:
    """Fetch and parse metadata from the pubmed API."""
    response = requests.get(
        metadata_url,
        params={
            "db": "pmc",
            "id": paper_id,
            "retmode": "xml"
            },
        timeout=10
        )
    response.raise_for_status()
    return ET.fromstring(response.text)

def extract_metadata(root: ET.Element, paper_id: str, pdf_download_url: str) -> dict:
    """Extract metadata for the PMC ID."""
    title_elem = root.find('.//article-title')
    title = title_elem.text if title_elem is not None else ""

    # Abstract
    abstract_elem = root.find('.//abstract')
    abstract = ''.join(abstract_elem.itertext()).strip() if abstract_elem is not None else ""

    # Authors
    authors = ', '.join(
        f"{name.findtext('given-names', default='')} {name.findtext('surname', default='')}".strip()
        for contrib in root.findall('.//contrib[@contrib-type="author"]')
        if (name := contrib.find('name')) is not None
    ) or ""

    # Publication Data
    pub_date_elem = root.find('.//published')
    pub_date = pub_date_elem.text.strip() if pub_date_elem is not None else ""

    pdf_url = f"{pdf_download_url}{paper_id}?pdf=render"
    if requests.get(pdf_url,timeout=10).status_code != 200:
        raise RuntimeError(f"No PDF found or access denied at {pdf_url}")

    logger.info("Metadata from PubMed for paper PMC ID: %s", paper_id)
    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{paper_id}.pdf",
        "source": "pubmed",
        "paper_id": paper_id,
    }


@tool(args_schema=DownloadPubMedInput, parse_docstring=True)
def download_pubmed_paper(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URL for an PubMed paper using PMC or PM ID.
    """
    logger.info("Fetching metadata from PubMed for paper PMC ID: %s", paper_id)

    metadata_url,pdf_download_url,map_url = get_pubmed_config()
    given_id=paper_id
    #Mapping given id to paper_id
    if given_id.lower().startswith("pmc"):
        pass
    else:
        given_id = map_ids(paper_id,map_url)
    # Extract metadata
    raw_data = fetch_metadata(metadata_url,given_id)
    metadata = extract_metadata(raw_data,given_id,pdf_download_url)

    # Create article_data entry with the paper ID as the key
    article_data = {paper_id: metadata}
    content = f"Successfully retrieved metadata for paper ID {paper_id}"
    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
