"""
Test cases for utils/embeddings/embedding_with_sentence_transformer.py
"""

import pytest
from ..utils.embeddings.embedding_with_sentence_transformer import EmbeddingWithSentenceTransformer

@pytest.fixture(name="embedding_model")
def embedding_model_fixture():
    """
    Fixture for creating an instance of EmbeddingWithSentenceTransformer.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v1"  # Small model for testing
    return EmbeddingWithSentenceTransformer(model_name=model_name)

def test_embed_documents(embedding_model):
    """
    Test the embed_documents method of EmbeddingWithSentenceTransformer class.
    """
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = embedding_model.embed_documents(texts)
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == 384

def test_embed_query(embedding_model):
    """
    Test the embed_query method of EmbeddingWithSentenceTransformer class.
    """
    text = "This is a test query."
    embedding = embedding_model.embed_query(text)
    assert len(embedding) > 0
    assert len(embedding) == 384

@pytest.fixture(name="embedding_model_half")
def embedding_model_half_fixture():
    """
    Fixture for creating an instance of EmbeddingWithSentenceTransformer with half precision.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v1"  # Small model for testing
    return EmbeddingWithSentenceTransformer(model_name=model_name, half_precision=True)

def test_half_prec_embed_documents(embedding_model_half):
    """
    Test the embed_documents method of EmbeddingWithSentenceTransformer class.
    """
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = embedding_model_half.embed_documents(texts)
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == 384

def test_half_prec_embed_query(embedding_model_half):
    """
    Test the embed_query method of EmbeddingWithSentenceTransformer class.
    """
    text = "This is a test query."
    embedding = embedding_model_half.embed_query(text)
    assert len(embedding) > 0
    assert len(embedding) == 384
