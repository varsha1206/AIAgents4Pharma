"""
Unit tests for arXiv paper downloading functionality, including:
- download_arxiv_paper tool function.
"""
import unittest
from unittest.mock import MagicMock, patch

from aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input import (
    DownloadArxivPaperInput
)

PATH="""
aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.DownloadArxivPaperInput"""
class TestDownloadArxivPaper(unittest.TestCase):
    """tests for the download_arxiv_paper tool.""" 
    @patch("requests.get")
    @patch(f"{PATH}.load_hydra_configs"
    )
    def test_download_arxiv_paper_success(
        self, mock_load_hydra_configs,mock_get
    ):
        """test the download_arxiv_paper tool for successful retrieval of metadata and PDF URL."""
        # Set up a dummy Hydra config.
        dummy_download_cfg = MagicMock()
        dummy_download_cfg.api_url = "http://dummy.arxiv.org/api"
        dummy_download_cfg.request_timeout = 10
        mock_load_hydra_configs.return_value = dummy_download_cfg

        # Set up a dummy XML response with a valid entry including a pdf link.
        arxiv_id = "arxiv_id:1234.56789"
        actual_id = arxiv_id.split(":")[1]
        retriever = DownloadArxivPaperInput()
        dummy_response = MagicMock()
        dummy_response.text = (
            f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
        <feed xmlns=\"http://www.w3.org/2005/Atom\">"""
            f"            <entry>"
            f"<title>Sample Paper Title</title>"
            f"<author><name>Author One</name></author>"
            f"<author><name>Author Two</name></author>"
            f"<summary>This is a sample abstract.</summary>"
            f"<published>2020-01-01T00:00:00Z</published>"
            f'<link title="pdf" href="http://arxiv.org/pdf/{actual_id}v1"/>'
            f"</entry></feed>"
        )
        dummy_response.raise_for_status = MagicMock()
        mock_get.return_value = dummy_response

        update = retriever.paper_retriever([arxiv_id])

        # Check that article_data was correctly set.
        self.assertIn("article_data", update)
        self.assertIn(actual_id, update["article_data"])
        metadata = update["article_data"][actual_id]
        self.assertEqual(metadata["Title"], "Sample Paper Title")
        self.assertEqual(metadata["Authors"], ["Author One", "Author Two"])
        self.assertEqual(metadata["Abstract"], "This is a sample abstract.")
        self.assertEqual(metadata["Publication Date"], "2020-01-01T00:00:00Z")
        self.assertEqual(metadata["URL"], f"http://arxiv.org/pdf/{actual_id}v1")
        self.assertEqual(metadata["pdf_url"], f"http://arxiv.org/pdf/{actual_id}v1")
        self.assertEqual(metadata["filename"], f"{actual_id}.pdf")
        self.assertEqual(metadata["source"], "arxiv")
        self.assertEqual(metadata["arxiv_id"], actual_id)

    @patch("requests.get")
    @patch(
    f"{PATH}.load_hydra_configs"
    )
    def test_no_entry_found(self, mock_load_hydra_configs,mock_get):
        """test the download_arxiv_paper tool for no entry found in XML response."""
        # Set up a dummy Hydra config.
        dummy_download_cfg = MagicMock()
        dummy_download_cfg.api_url = "http://dummy.arxiv.org/api"
        dummy_download_cfg.request_timeout = 10
        mock_load_hydra_configs.return_value = dummy_download_cfg

        # Set up XML with no entry element.
        arxiv_id = "arxiv_id:1234.56789"
        dummy_xml = (
            """<?xml version="1.0" encoding="UTF-8"?>"""
            """<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        )
        dummy_response = MagicMock()
        dummy_response.text = dummy_xml
        dummy_response.raise_for_status = MagicMock()
        mock_get.return_value = dummy_response

        retriever = DownloadArxivPaperInput()
        # No entry found should result in empty article_data and header-only summary
        update = retriever.paper_retriever([arxiv_id])
        self.assertIn("article_data", update)
        self.assertEqual(update["article_data"], {})

    @patch("requests.get")
    @patch(
    f"{PATH}.load_hydra_configs"
    )
    def test_no_pdf_url_found(self, mock_load_hydra_configs, mock_get):
        """test the download_arxiv_paper tool for no PDF URL found in XML response."""
        # Dummy config.
        dummy_download_cfg = MagicMock()
        dummy_download_cfg.api_url = "http://dummy.arxiv.org/api"
        dummy_download_cfg.request_timeout = 10
        mock_load_hydra_configs.return_value = dummy_download_cfg

        # Set up XML with an entry that does not contain a pdf link.
        arxiv_id = "arxiv_id:1234.56789"
        dummy_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Sample Paper Title</title>
                <author>
                    <name>Author One</name>
                </author>
                <summary>This is a sample abstract.</summary>
                <published>2020-01-01T00:00:00Z</published>
                <!-- Missing pdf link -->
            </entry>
        </feed>
        """
        dummy_response = MagicMock()
        dummy_response.text = dummy_xml
        dummy_response.raise_for_status = MagicMock()
        mock_get.return_value = dummy_response

        retriever  = DownloadArxivPaperInput()
        with self.assertRaises(RuntimeError) as context:
            retriever.paper_retriever([arxiv_id])
        self.assertEqual(
            str(context.exception), f"Could not find PDF URL for arXiv ID {arxiv_id.split(":")[1]}"
        )
