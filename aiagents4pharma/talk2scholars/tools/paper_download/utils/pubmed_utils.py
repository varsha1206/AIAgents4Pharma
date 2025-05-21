# utils/pubmed_utils.py
"""
Pubmed Utility functions to map unique ids, fetch and extract metadata.
"""
import requests
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

def map_ids(input_id: str, map_url: str) -> str:
    """Mapping the PM ID or DOI of paper in order to get the PMC ID"""
    response = requests.get(f"{map_url}?ids={input_id}", timeout=10)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    if ((record := root.find("record")) is not None and record.attrib.get("pmcid")):
        logger.info("Retrieved PMC ID for the given id %s", input_id)
        return record.attrib["pmcid"]
    #raise RuntimeError(f"PMC id not found for {input_id}")
    return "PMC"+input_id
def fetch_pubmed_metadata(url: str, paper_id: str) -> ET.Element:
    """Fetch and parse metadata from the API url"""
    response = requests.get(
        url,
        params={"db": "pmc", "id": paper_id, "retmode": "xml"},
        timeout=10,
    )
    response.raise_for_status()
    return ET.fromstring(response.text)

def extract_pubmed_metadata(xml_root: ET.Element, paper_id: str, pdf_base_url: str) -> dict:
    """Extract metadata from the XML entry."""
    title_elem = xml_root.find(".//article-title")
    title = title_elem.text if title_elem is not None else "N/A"

    abstract_elem = xml_root.find(".//abstract")
    abstract = "".join(abstract_elem.itertext()).strip() if abstract_elem is not None else "N/A"

    authors = ", ".join(
        f"{name.findtext('given-names', default='')} {name.findtext('surname', default='')}".strip()
        for contrib in xml_root.findall('.//contrib[@contrib-type="author"]')
        if (name := contrib.find("name")) is not None
    ) or "N/A"

    pub_date_elem = xml_root.find(".//published")
    pub_date = pub_date_elem.text.strip() if pub_date_elem is not None else "N/A"

    pdf_url = f"{pdf_base_url}{paper_id}?pdf=render"
    if requests.get(pdf_url, timeout=10).status_code != 200:
        raise RuntimeError(f"No PDF found or access denied at {pdf_url}")

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{paper_id}.pdf",
        "source": "pubmed",
        "pmc_id": paper_id,
    }
