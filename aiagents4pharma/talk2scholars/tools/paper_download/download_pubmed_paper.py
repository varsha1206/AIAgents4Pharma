#!/usr/bin/env python3
"""
Tool for downloading PubMedX paper metadata and retrieving the PDF URL.
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


class DownloadPubMedXInput(BaseModel):
    """Input schema for the pubmedx paper download tool."""

    doi_id: str = Field(
        description="The DOI ID used to retrieve the paper details and PDF URL."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


def fetch_pubmedx_ids(
    doi_converter_url: str, doi_id: str, request_timeout: int
) -> ET.Element:
    """Fetch and parse metadata from the PubMedX API."""
    query_url = f"{doi_converter_url}{doi_id}"
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return ET.fromstring(response.text)


def extract_metadata(metadata_url: str, pmc_id: str, pdf_download_url: str) -> dict:
    """Extract metadata for the PMC ID."""
    params = {
    "db": "pmc",
    "id": pmc_id,
    "retmode": "xml"
    }
    response = requests.get(metadata_url, params=params)
    root = ET.fromstring(response.content)
    article_title = root.find('.//article-title')
    title = article_title.text.strip() if article_title is not None else "N/A"

    # Title
    title_elem = root.find('.//article-title')
    title = title_elem.text if title_elem is not None else "N/A"

    # Abstract
    abstract_elem = root.find('.//abstract')
    abstract = ''.join(abstract_elem.itertext()).strip() if abstract_elem is not None else "N/A"

    # Authors
    authors = []
    for contrib in root.findall('.//contrib[@contrib-type="author"]'):
        name = contrib.find('name')
        if name is not None:
            given = name.findtext('given-names', default='')
            surname = name.findtext('surname', default='')
            full_name = f"{given} {surname}".strip()
            authors.append(full_name)
    authors = ', '.join(authors) if authors else "N/A"


    # Publication Data
    pub_date_elem = root.find('.//published')
    pub_date = pub_date_elem.text.strip() if pub_date_elem is not None else "N/A"

    pdf_url = f"{pdf_download_url}{pmc_id}/pdf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            }
    response = requests.get(pdf_url,timeout=10, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(f"No PDF found or access denied at {pdf_url}")

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{pmc_id}.pdf",
        "source": "pubmed",
        "pmc_id": pmc_id,
    }


@tool(args_schema=DownloadPubMedXInput, parse_docstring=True)
def download_pubmedx_paper(
    doi_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URL for an PubMedX paper using its unique DOI.
    """
    logger.info("Fetching metadata from PubMedx for paper PMC ID: %s", doi_id)

    # Load configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_pubmed_paper=default"]
        )
        api_url = cfg.tools.download_pubmed_paper.doi_converter_api
        request_timeout = cfg.tools.download_pubmed_paper.request_timeout
        metadata_url = cfg.tools.download_pubmed_paper.metadata_url
        pdf_download_url = cfg.download_pubmed_paper.pdf_base_url

    # Fetch and parse metadata
    root = fetch_pubmedx_ids(api_url, doi_id, request_timeout)
    record = root.find('record')
    if record is None:
        raise ValueError(f"No entry found for DOI ID {doi_id}")
    
    try:
        pmc_id = record.attrib.get('pmcid')
    except:
        raise ValueError(f"PMC ID does not exist for the DOI id {doi_id}")

    # Extract metadata
    metadata = extract_metadata(metadata_url,pmc_id,pdf_download_url)

    # Create article_data entry with the paper ID as the key
    article_data = {pmc_id: metadata}

    content = f"Successfully retrieved metadata and PDF URL for PMC ID {pmc_id}"

    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
