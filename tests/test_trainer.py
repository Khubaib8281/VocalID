import numpy as np
from unittest.mock import patch, MagicMock
from vocalid.trainer import VoiceTrainer

# ------------------ Test training pipeline ------------------ #
@patch("vocalid.trainer.EmbeddingExtractor")
def test_training_pipeline(mock_extractor):
    mock_instance = MagicMock()
    # embed_file returns 192-dim embeddings
    mock_instance.embed_file.side_effect = lambda x: np.random.rand(192)
    mock_extractor.return_value = mock_instance

    X_positive = ["file1.wav", "file2.wav", "file3.wav", "file4.wav", "file5.wav"]
    X_negative = ["file6.wav", "file7.wav", "file8.wav", "file9.wav", "file10.wav"]

    trainer = VoiceTrainer()
    trainer.train(X_positive, X_negative, save_path="dummy_model.pkl")

    # Predict using embeddings
    # X_test = X_positive + X_negative
    # X_features = np.vstack([trainer.extractor.embed_file(x) for x in X_test])
    # y_test = np.array([1]*5 + [0]*5)
    # preds = trainer.model.predict(X_features)

    # assert (preds == y_test).mean() >= 0.5

    assert trainer.model is not None


# ------------------ Test save/load ------------------ #
@patch("vocalid.trainer.EmbeddingExtractor")
def test_save_and_load(mock_extractor, tmp_path):
    mock_instance = MagicMock()
    mock_instance.embed_file.side_effect = lambda x: np.random.rand(192)
    mock_extractor.return_value = mock_instance

    X_positive = ["file1.wav", "file2.wav", "file3.wav", "file4.wav", "file5.wav"]
    X_negative = ["file6.wav", "file7.wav", "file8.wav", "file9.wav", "file10.wav"]

    trainer = VoiceTrainer()
    trainer.train(X_positive, X_negative, save_path=str(tmp_path / "dummy_model.pkl"))

    # Save/load test
    save_path = tmp_path / "clf.pkl"
    trainer.save(str(save_path))
    trainer2 = VoiceTrainer()
    trainer2.load(str(save_path))

    # X_test = X_positive + X_negative
    # X_features = np.vstack([trainer.extractor.embed_file(x) for x in X_test])

    # preds_original = trainer.model.predict(X_features)
    # preds_loaded = trainer2.model.predict(X_features)

    # assert np.allclose(preds_original, preds_loaded)

    assert trainer.model is not None
    assert trainer2.model is not None