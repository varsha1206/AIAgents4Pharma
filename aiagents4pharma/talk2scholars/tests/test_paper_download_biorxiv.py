"""
Unit tests for bioRxiv paper downloading functionality, including:
- download_bioRxiv_paper tool function.
"""

import unittest
from unittest.mock import MagicMock, patch

from aiagents4pharma.talk2scholars.tools.paper_download.download_biorxiv_input import (
    DownloadBiorxivPaperInput)

PATH = "aiagents4pharma.talk2scholars.tools.paper_download.download_biorxiv_input"
class TestDownloadBiorxivPaper(unittest.TestCase):
    """Tests for the download_bioRxiv_paper tool."""
    @patch(
    f"{PATH}.hydra.initialize"
    )
    @patch(
    f"{PATH}.hydra.compose"
    )
    def test_load_hydra_configs_runs_and_sets_attributes(self,mock_compose, mock_initialize):
        """
        Ensures:
        - logger.info runs
        - hydra.initialize runs
        - hydra.compose runs
        - self.metadata_url etc. get set
        """
        retriever = DownloadBiorxivPaperInput()

        # Fake config structure
        mock_cfg = MagicMock()
        mock_cfg.tools.download_biorxiv_paper.request_timeout = 10
        mock_compose.return_value = mock_cfg

        retriever.load_hydra_configs()

        mock_initialize.assert_called_once_with(version_base=None, config_path="../../configs")
        mock_compose.assert_called_once_with(
            config_name="config",
            overrides=["tools/download_biorxiv_paper=default"]
        )

        assert retriever.request_timeout == 10
    @patch(
        f"{PATH}.DownloadBiorxivPaperInput.load_hydra_configs"
    )
    @patch("requests.get")
    def test_download_biorxiv_paper_success(self, mock_get, mock_load_hydra):
        """Test successful metadata and PDF URL retrieval."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_biorxiv_paper.api_url = "http://dummy.biorxiv.org/api"
        dummy_cfg.tools.download_biorxiv_paper.request_timeout = 10
        mock_load_hydra.return_value = dummy_cfg

        doi = "10.1101/2025.05.13.653102"

        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.raise_for_status = MagicMock()
        dummy_response.json.return_value = {
            "collection": [
                {
                    "title": "Sample BioRxiv Paper",
                    "authors": "Author One; Author Two",
                    "abstract": "This is a bioRxiv abstract.",
                    "date": "2025-04-25",
                    "doi": doi,
                    "link": f"https://www.biorxiv.org/content/{doi}.full.pdf"
                }
            ]
        }
        mock_get.return_value = dummy_response

        tool_input = "doi:"+doi
        downloader = DownloadBiorxivPaperInput()
        update = downloader.paper_retriever([tool_input])

        self.assertIn("article_data", update)
        self.assertIn(doi, update["article_data"])
        metadata = update["article_data"][doi]
        self.assertEqual(metadata["Title"], "Sample BioRxiv Paper")
        self.assertEqual(metadata["Authors"], "Author One; Author Two")
        self.assertEqual(metadata["Abstract"], "This is a bioRxiv abstract.")
        self.assertEqual(metadata["Publication Date"], "2025-04-25")
        self.assertEqual(metadata["URL"], f"https://www.biorxiv.org/content/{doi}.full.pdf")
        self.assertEqual(metadata["pdf_url"], f"https://www.biorxiv.org/content/{doi}.full.pdf")
        self.assertEqual(metadata["filename"], f"{doi.rsplit('/', maxsplit=1)[-1]}.pdf")
        self.assertEqual(metadata["source"], "biorxiv")
        self.assertEqual(metadata["biorxiv_id"], doi)

    @patch(
        f"{PATH}.DownloadBiorxivPaperInput.load_hydra_configs"
    )
    @patch("requests.get")
    def test_no_entry_found(self, mock_get, mock_load_hydra):
        """Test behavior when no 'entry' is in response."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_biorxiv_paper.api_url = "http://dummy.biorxiv.org/api"
        dummy_cfg.tools.download_biorxiv_paper.request_timeout = 10
        mock_load_hydra.return_value = dummy_cfg

        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.raise_for_status = MagicMock()
        dummy_response.json.return_value = {}  # No collection
        mock_get.return_value = dummy_response

        doi = "10.1101/2025.05.13.653102"
        tool_input = "doi:"+doi

        downloader = DownloadBiorxivPaperInput()
        with self.assertRaises(ValueError) as context:
            downloader.paper_retriever([tool_input])

        self.assertEqual(str(context.exception), f"No metadata found for DOI: {doi}")

    @patch(
        f"{PATH}.DownloadBiorxivPaperInput.load_hydra_configs"
    )
    @patch("requests.get")
    def test_no_pdf_url_found(self, mock_get, mock_load_hydra):
        """Test fallback to DOI-based PDF URL construction when 'link' is missing."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_biorxiv_paper.api_url = "http://dummy.biorxiv.org/api"
        dummy_cfg.tools.download_biorxiv_paper.request_timeout = 10
        mock_load_hydra.return_value = dummy_cfg

        doi = "10.1101/2025.05.13.653102"

        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.raise_for_status = MagicMock()
        dummy_response.json.return_value = {
            "collection": [
                {
                    "title": "Sample BioRxiv Paper",
                    "authors": "Author One; Author Two",
                    "abstract": "This is a bioRxiv abstract.",
                    "date": "2025-04-25",
                    "doi": doi
                    # 'link' is intentionally omitted
                }
            ]
        }
        mock_get.return_value = dummy_response

        tool_input = "doi:"+doi
        downloader = DownloadBiorxivPaperInput()
        update = downloader.paper_retriever([tool_input])
        metadata = update["article_data"][doi]

        expected_suffix = doi.rsplit('/', maxsplit=1)[-1]
        expected_url = f"https://www.biorxiv.org/content/10.1101/{expected_suffix}.full.pdf"

        self.assertEqual(metadata["pdf_url"], expected_url)
        self.assertEqual(metadata["URL"], expected_url)

    @patch(
        f"{PATH}.DownloadBiorxivPaperInput.load_hydra_configs"
    )
    @patch("requests.get")
    def test_extract_metadata_pdf_not_found(self, mock_get, mock_load_hydra):
        """Test when PDF link returns non-200 status code (extract_metadata fallback)."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_biorxiv_paper.api_url = "http://dummy.biorxiv.org/api"
        dummy_cfg.tools.download_biorxiv_paper.request_timeout = 10
        mock_load_hydra.return_value = dummy_cfg

        # Mock initial metadata fetch (metadata found)
        metadata_response = MagicMock()
        metadata_response.status_code = 200
        metadata_response.raise_for_status = MagicMock()
        metadata_response.json.return_value = {
            "collection": [
                {
                    "title": "Test Title",
                    "authors": "Test Author",
                    "abstract": "Test Abstract",
                    "date": "2025-04-25",
                    "doi": "10.1101/2025.05.13.653102"
                }
            ]
        }

        # First GET returns metadata, second GET (PDF link) returns 404
        def side_effect(url, timeout):
            timeout+=1
            if "api" in url:
                return metadata_response
            pdf_response = MagicMock()
            pdf_response.status_code = 404  # Simulate PDF link not found
            return pdf_response

        mock_get.side_effect = side_effect

        doi = "10.1101/2025.05.13.653102"
        tool_input = "doi:" + doi
        downloader = DownloadBiorxivPaperInput()
        update = downloader.paper_retriever([tool_input])

        # Should be empty because PDF was not accessible
        self.assertNotIn(doi, update["article_data"])

    @patch(
        f"{PATH}.DownloadBiorxivPaperInput.load_hydra_configs"
    )
    @patch(f"{PATH}.DownloadBiorxivPaperInput.extract_metadata")
    @patch("requests.get")
    def test_paper_retriever_else_branch(self, mock_get, mock_extract_metadata, mock_load_hydra):
        """Test paper_retriever hits 'else' branch when extract_metadata returns empty dict."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_biorxiv_paper.api_url = "http://dummy.biorxiv.org/api"
        dummy_cfg.tools.download_biorxiv_paper.request_timeout = 10
        mock_load_hydra.return_value = dummy_cfg

        # Mock metadata fetch with valid collection
        metadata_response = MagicMock()
        metadata_response.status_code = 200
        metadata_response.raise_for_status = MagicMock()
        metadata_response.json.return_value = {
            "collection": [
                {
                    "title": "Test Title",
                    "authors": "Test Author",
                    "abstract": "Test Abstract",
                    "date": "2025-04-25",
                    "doi": "10.1101/2025.05.13.653102"
                }
            ]
        }
        mock_get.return_value = metadata_response

        # Force extract_metadata to return {}
        mock_extract_metadata.return_value = {}

        doi = "10.1101/2025.05.13.653102"
        tool_input = "doi:" + doi
        downloader = DownloadBiorxivPaperInput()
        update = downloader.paper_retriever([tool_input])

        # 'article_data' should not contain DOI because metadata is empty
        self.assertNotIn(doi, update["article_data"])
