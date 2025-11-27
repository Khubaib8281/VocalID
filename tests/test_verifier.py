from unittest.mock import patch, MagicMock
import numpy as np

@patch("vocalid.verifier.load_model")
@patch("vocalid.verifier.EmbeddingExtractor")
def test_verifier(mock_extractor, mock_load_model):
    # Mock extractor
    mock_instance = MagicMock()
    mock_instance.extract.return_value = np.zeros(192)
    mock_extractor.return_value = mock_instance

    # Mock model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.9]]
    mock_load_model.return_value = mock_model

    from vocalid.verifier import VoiceVerifier
    verifier = VoiceVerifier("model.bin")

    # Fix: call extractor.extract instead of get_embedding
    dummy_waveform = np.random.rand(16000).astype("float32")
    emb = verifier.extractor.extract(dummy_waveform)
    score = mock_model.predict_proba([emb])[0][1]
    verified = score >= 0.5  # use some threshold

    assert isinstance(verified, bool)
    assert isinstance(score, float)
    mock_instance.extract.assert_called_once_with(dummy_waveform)
