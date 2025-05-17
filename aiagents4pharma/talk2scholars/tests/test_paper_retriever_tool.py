"""
Unit tests for paper retrieval router tool (paper_retriever), including:
- Delegation to arXiv and PubMed downloaders
- Validation for invalid or missing source
"""

import unittest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from langgraph.types import Command

from aiagents4pharma.talk2scholars.tools.paper_download.paper_retriever import paper_retriever


class TestPaperRetrieverTool(unittest.TestCase):
    """Tests for the paper_retriever tool."""

    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.paper_retriever.download_pubmedx_paper"
            )
    def test_retrieve_pubmed_paper(self, mock_pubmed_download):
        """Test paper retrieval when source is 'pubmed'."""
        mock_command = MagicMock(spec=Command)
        mock_pubmed_download.return_value = mock_command

        tool_input = {
            "source": "pubmed",
            "paper_id": "PMC123456",
            "tool_call_id": "test_tool_id"
        }

        result = paper_retriever.invoke(tool_input)

        mock_pubmed_download.assert_called_once_with(
            pmc_id="PMC123456", tool_call_id="test_tool_id"
        )
        self.assertEqual(result, mock_command)

    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.paper_retriever.download_arxiv_paper"
        )
    def test_retrieve_arxiv_paper(self, mock_arxiv_download):
        """Test paper retrieval when source is 'arxiv'."""
        mock_command = MagicMock(spec=Command)
        mock_arxiv_download.return_value = mock_command

        tool_input = {
            "source": "arxiv",
            "paper_id": "arXiv:9876.5432",
            "tool_call_id": "test_tool_id"
        }

        result = paper_retriever.invoke(tool_input)

        mock_arxiv_download.assert_called_once_with(
            arxiv_id="arXiv:9876.5432", tool_call_id="test_tool_id"
        )
        self.assertEqual(result, mock_command)

    def test_invalid_source(self):
        """Test ValidationError raised for unsupported source."""
        tool_input = {
            "source": "invalid_source",  # not in Literal
            "paper_id": "12345",
            "tool_call_id": "test_tool_id"
        }

        with self.assertRaises(ValidationError) as context:
            paper_retriever.invoke(tool_input)

        self.assertIn("Input should be 'pubmed' or 'arxiv'", str(context.exception))

    def test_missing_source(self):
        """Test ValidationError raised when source is None."""
        tool_input = {
            "source": None,
            "paper_id": "12345",
            "tool_call_id": "test_tool_id"
        }

        with self.assertRaises(ValidationError) as context:
            paper_retriever.invoke(tool_input)

        self.assertIn("Input should be 'pubmed' or 'arxiv'", str(context.exception))
