# utils/arxiv_utils.py
"""
Arxiv Utility functions to fetch and extract metadata.
"""
import requests
import re
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

def fetch_arxiv_metadata(api_url: str, arxiv_id: str, request_timeout: int) -> ET.Element:
    """Fetch and parse metadata from the arXiv API."""
    query_url = f"{api_url}?search_query=id:{arxiv_id}&start=0&max_results=1"
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return ET.fromstring(response.text)


def extract_arxiv_metadata(entry: ET.Element, ns: dict, arxiv_id: str) -> dict:
    """Extract metadata from the XML entry."""
    title_elem = entry.find("atom:title", ns)
    title = title_elem.text.strip() if title_elem is not None else "N/A"

    authors = [
        author_elem.find("atom:name", ns).text.strip()
        for author_elem in entry.findall("atom:author", ns)
        if author_elem.find("atom:name", ns) is not None
    ]

    summary_elem = entry.find("atom:summary", ns)
    abstract = summary_elem.text.strip() if summary_elem is not None else "N/A"

    published_elem = entry.find("atom:published", ns)
    pub_date = published_elem.text.strip() if published_elem is not None else "N/A"

    pdf_url = next(
        (
            link.attrib.get("href")
            for link in entry.findall("atom:link", ns)
            if link.attrib.get("title") == "pdf"
        ),
        None,
    )

    if not pdf_url:
        raise RuntimeError(f"Could not find PDF URL for arXiv ID {arxiv_id}")

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{arxiv_id}.pdf",
        "source": "arxiv",
        "arxiv_id": arxiv_id,
    }

def is_arxiv_id(paper_id: str) -> bool:
    modern_pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
    legacy_pattern = r"^[a-z\-]+\/\d{7}(v\d+)?$"
    return re.match(modern_pattern, paper_id) or re.match(legacy_pattern, paper_id)
