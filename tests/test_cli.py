import sys
from unittest.mock import patch, MagicMock
from vocalid.cli import main

@patch("vocalid.cli.VoiceTrainer")
def test_cli_train(mock_trainer):
    trainer_instance = MagicMock()
    mock_trainer.return_value = trainer_instance

    test_args = ["cli.py", "train", "--positive", "pos_folder", "--negative", "neg_folder"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Ensure the train method was called
    trainer_instance.train.assert_called_once()


@patch("vocalid.cli.VoiceVerifier")
def test_cli_verify(mock_verifier):
    verifier_instance = MagicMock()
    verifier_instance.verify_file.return_value = (True, 0.95)
    mock_verifier.return_value = verifier_instance

    test_args = ["cli.py", "verify", "dummy.wav"]
    with patch.object(sys, 'argv', test_args):
        main()    

    verifier_instance.verify_file.assert_called_once_with("dummy.wav")
   