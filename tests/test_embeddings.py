import numpy as np
from unittest.mock import patch, MagicMock
import pytest

@patch("vocalid.embeddings.EmbeddingExtractor")
def test_embedding_extractor(mock_extractor):
    """
    Ensure that the EmbeddingExtractor returns a fixed-size embedding
    without requiring SpeechBrain to be installed.
    """
    # Mock the extractor instance
    mock_instance = MagicMock()
    mock_instance.extract.return_value = np.random.rand(192)  # fixed embedding size
    mock_extractor.return_value = mock_instance

    # Import inside the test to ensure mock patching works
    from vocalid.embeddings import EmbeddingExtractor

    extractor = EmbeddingExtractor()
    dummy_input = np.random.rand(16000).astype("float32")  # 1-second dummy audio at 16kHz

    emb = extractor.extract(dummy_input)

    # Check the embedding shape
    assert emb.shape[0] == 192
    mock_instance.extract.assert_called_once_with(dummy_input)
