"""
Unit tests for paper download toal
"""

from unittest.mock import patch, MagicMock
from aiagents4pharma.talk2scholars.tools.paper_download import download_tool


# Dummy state fixture (if you use pytest fixtures, otherwise inline)
dummy_state = {
    "llm_model": MagicMock()
}


@patch("aiagents4pharma.talk2scholars.tools.paper_download.utils.summary_builder.build_summary")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadArxivPaperInput")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadBiorxivPaperInput")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadPubmedPaperInput")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadMedrxivPaperInput")
def test_download_paper_tool(
    mock_medrxiv_cls,
    mock_pubmed_cls,
    mock_biorxiv_cls,
    mock_arxiv_cls,
    mock_build_summary
):
    """Testing the download paper tool"""
    # Create mocks for each retriever
    mock_arxiv = MagicMock()
    mock_biorxiv = MagicMock()
    mock_pubmed = MagicMock()
    mock_medrxiv = MagicMock()

    mock_arxiv.paper_retriever.return_value = {
        "article_data": {"1234.5678": {"Title": "Arxiv Paper"}}
    }
    mock_biorxiv.paper_retriever.return_value = {
        "article_data": {"10.1101/2025.06.22.660927": {"Title": "Biorxiv Paper"}}
    }
    mock_pubmed.paper_retriever.return_value = {
        "article_data": {"PMC123456": {"Title": "Pubmed Paper"}}
    }
    mock_medrxiv.paper_retriever.return_value = {
        "article_data": {"10.1101/2025.06.22.660927": {"Title": "Medrxiv Paper"}}
    }

    # Set the patched classes to return the mocks
    mock_arxiv_cls.return_value = mock_arxiv
    mock_biorxiv_cls.return_value = mock_biorxiv
    mock_pubmed_cls.return_value = mock_pubmed
    mock_medrxiv_cls.return_value = mock_medrxiv

    mock_build_summary.return_value = (
    "Download was successful. \n"
    "Papers metadata are attached as an artifact. Here is a summary of the results:\n"
    "Number of papers found: 3\n"
    "Top 3 papers:\n"
    "1. Biorxiv Paper (N/A)\n"
    "2. Pubmed Paper (N/A)\n"
    "3. Arxiv Paper (N/A)"
)
    expected = """Download was successful.
     Papers metadata are attached as an artifact. Here is a summary of the results:\n
    Number of papers found: 3\n
    Top 3 papers:\n
    1. Biorxiv Paper (N/A)\n
    2. Pubmed Paper (N/A)\n
    3. Arxiv Paper (N/A)"""
    tool_input = {
        "paper_id": ["arxiv_id:1234.5678", "doi:10.1101/2025.06.22.660927", "pubmed:PMC123456"],
        "tool_call_id": "test_call_id",
        "state": dummy_state
    }

    result = download_tool.download_paper.run(tool_input)

    # Assert each was called
    mock_arxiv.paper_retriever.assert_called_once()
    mock_biorxiv.paper_retriever.assert_called_once()
    mock_pubmed.paper_retriever.assert_called_once()
    # Medrxiv should not be called in the normal path
    mock_medrxiv.paper_retriever.assert_not_called()

    assert expected in result.update["messages"][0]


@patch("aiagents4pharma.talk2scholars.tools.paper_download.utils.summary_builder.build_summary")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadArxivPaperInput")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadBiorxivPaperInput")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadPubmedPaperInput")
@patch("aiagents4pharma.talk2scholars.tools.paper_download.download_tool.DownloadMedrxivPaperInput")
def test_download_paper_medrxiv_fallback(
    mock_medrxiv_cls,
    mock_pubmed_cls,
    mock_biorxiv_cls,
    mock_arxiv_cls,
    mock_build_summary
):
    """testing download when medrxiv id is given"""
    mock_arxiv = MagicMock()
    mock_biorxiv = MagicMock()
    mock_pubmed = MagicMock()
    mock_medrxiv = MagicMock()

    # BioRxiv fails -> fallback to MedRxiv
    mock_biorxiv.paper_retriever.side_effect = Exception("BioRxiv failed")
    mock_medrxiv.paper_retriever.return_value = {
        "article_data": {"10.1101/2025.06.22.660927": {"Title": "Medrxiv Paper"}}
    }

    mock_arxiv.paper_retriever.return_value = {
        "article_data": {"1234.5678": {}}
    }
    mock_pubmed.paper_retriever.return_value = {
        "article_data": {"PMC123456": {}}
    }

    mock_arxiv_cls.return_value = mock_arxiv
    mock_biorxiv_cls.return_value = mock_biorxiv
    mock_pubmed_cls.return_value = mock_pubmed
    mock_medrxiv_cls.return_value = mock_medrxiv

    mock_build_summary.return_value = (
    "Download was successful. \n"
    "Papers metadata are attached as an artifact. Here is a summary of the results:\n"
    "Number of papers found: 3\n"
    "Top 3 papers:\n"
    "1. Medrxiv Paper (N/A)\n"
    "2. N/A (N/A)\n"
    "3. N/A (N/A)"
)
    expected = """Download was successful.
     Papers metadata are attached as an artifact. Here is a summary of the results:\n
    Number of papers found: 3\n
    Top 3 papers:\n
    1. Medrxiv Paper (N/A)\n
    2. N/A (N/A)\n
    3. N/A (N/A)"""

    tool_input = {
        "paper_id": ["arxiv_id:1234.5678", "doi:10.1101/2025.06.22.660927", "pubmed:PMC123456"],
        "tool_call_id": "test_call_id",
        "state": dummy_state
    }

    result = download_tool.download_paper.run(tool_input)

    mock_biorxiv.paper_retriever.assert_called_once()
    mock_medrxiv.paper_retriever.assert_called_once()

    assert expected in result.update["messages"][0]
