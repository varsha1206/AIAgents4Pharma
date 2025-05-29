#!/usr/bin/env python3
"""
This package provides modules for fetching and downloading academic papers from arXiv,
pubmed, biorxiv and medrxiv.
"""

# Import modules
from . import (
    download_arxiv_input,
    download_pubmed_paper,
    download_biorxiv_input,
    download_medrxiv_input,
)

__all__ = [
    "download_arxiv_input",
    "download_pubmed_paper",
    "download_biorxiv_input",
    "download_medrxiv_input",
]
