# VocalID: A Lightweight Voice Authentication Toolkit

VocalID is a compact and practical voice authentication library that
combines ECAPA-TDNN embeddings with a simple classifier to verify user
identity from audio recordings. It supports file-based verification and
real-time microphone input. The project is designed to be easy to train,
deploy, and# VocalID: A Lightweight Voice Authentication Toolkit

VocalID is a compact and practical voice authentication library that
combines ECAPA-TDNN embeddings with a simple classifier to verify user
identity from audio recordings. It supports file-based verification and
real-time microphone input. The project is designed to be easy to train,
deploy, and extend.

------------------------------------------------------------------------

## Features

- **ECAPA-TDNN embeddings** using `speechbrain/spkrec-ecapa-voxceleb`
- **Training** with positive (owner) and negative (impostor) audio
    samples
- **Evaluation** with accuracy and classification metrics
- **Verification** from audio files or live microphone input
- **CLI toolkit** for training, evaluating, and verifying
- **Modular design** with trainer, verifier, embeddings, config, and
    utilities
- **Simple model storage** using pickle-based persistence
- **Full test suite included**

------------------------------------------------------------------------

## How It Works

### 1. Audio Processing

Audio is loaded or recorded, resampled, and normalized.

### 2. Embedding Extraction

ECAPA-TDNN generates fixed-dimensional speaker embeddings. These embeddings represent unique speaker characteristics.

### 3. Feature Preparation

Positive and negative embeddings are labeled and fed into the trainer.

### 4. Classification Model

A simple Logistic Regression model is trained on the embeddings.

### 5. Verification

During verification:

1. Extract embeddings for the new audio.
2. Predict with trained classifier.
3. Return a confidence score.
4. Compare score with threshold from `config.py`.

------------------------------------------------------------------------

## Package Structure

    VocalID
        └── voice_verifier/
            │
            ├── trainer.py         # Training logic, evaluation, model save/load
            ├── verifier.py        # File and waveform verification
            ├── embeddings.py      # ECAPA-TDNN embedding extraction
            ├── audio_utils.py     # Audio loading and microphone recording
            ├── config.py          # Threshold + ECAPA model configuration
            ├── model_store.py     # Model checkpoint loader
            ├── cli.py             # Command-line interface
        └── tests/                 # Full pytest suite
        └── examples/
        └── requirements.txt
        └── api/
            ├── app.py
        └── README.md

------------------------------------------------------------------------

## Components

### VoiceTrainer

Functions: - `train()` - `evaluate()` - `prepare_features()` -
`save()` - `load()`

### VoiceVerifier

Methods: - `verify_file(path)` - `verify_array(audio_tensor)`

### EmbeddingExtractor

- `embed_file(path)`
- `embed_waveform(waveform, sr)`

### Audio Utilities

- `load_audio(path)`
- `record_audio(seconds)`

------------------------------------------------------------------------

## Installation

``` python
    pip install vocalid
```

------------------------------------------------------------------------

## Example usage script(Python)

### Directory Structure Example

Assume your dataset looks like this:

> Voice tip: Each voice sample of 5-6 seconds with different tone/ bg noise/ accent/ microphone

    └── dataset/
        └── my_voice/               <-- positive class (your voice)
            sample1.wav
            sample2.wav
            sample3.wav
            sample4.wav
            
        └── other_voices/           <-- negative class(other's voices)
            voice1.wav
            voice2.wav
            voice3.wav
            voice4.wav

### Full python script

``` python
from vocalid.trainer import VoiceTrainer
from vocalid.verifier import VoiceVerifier
from vocalid.audio_utils import load_audio
import glob

# 1. TRAINING THE MODEL

pos_files = glob.glob("dataset/my_voice/*.wav")
neg_files = glob.glob("dataset/other_voices/*.wav")

trainer = VoiceTrainer()
trainer.train(pos_files, neg_files, save_path="my_voice_model.pkl")

# (Optional) Check metrics printed by evaluate() in train()
print("Training complete. Model saved.")


# 2. EVALUATING THE TRAINED MODEL (Manually)

# This is useful if you want to evaluate after loading the model.
# Or you want to compute new metrics on a different test set.

# Example test data (can be same folders or separate ones)
test_pos = glob.glob("dataset/my_voice_test/*.wav")
test_neg = glob.glob("dataset/other_voices_test/*.wav")

metrics = trainer.evaluate(test_pos, test_neg)

print("Accuracy:", metrics["accuracy"])
print("Report:\n", metrics["report"])

# Example output:
# Classification report text
# Accuracy: 0.91


# 3. VERIFY A FILE

verifier = VoiceVerifier("my_voice_model.pkl")

to_verify = "verify_samples/unknown_voice.wav"
ok, score = verifier.verify_file(to_verify)

print(f"\nVerification result: {ok}, Score: {score:.3f}")
# ok = True means it matches your voice
# score is probability from the classifier


# 4. VERIFY LIVE MICROPHONE AUDIO (Windows supported)

# Record a short clip and verify
audio_tensor = record_audio(seconds=4)
ok, score = verifier.verify_array(audio_tensor)

print(f"Live verification: {ok}, Score: {score:.3f}")

```

### Example Evaluate-Only Script

If someone just wants to evaluate the model later:

``` python
from vocalid.trainer import VoiceTrainer
import glob

trainer = VoiceTrainer()
trainer.load("my_voice_model.pkl")

test_pos = glob.glob("dataset/my_voice_test/*.wav")
test_neg = glob.glob("dataset/other_voices_test/*.wav")

metrics = trainer.evaluate(test_pos, test_neg)

print("Accuracy:", metrics["accuracy"])
print("Report:\n", metrics["report"])

```

### CLI Commands   

``` python
    vocalid train --positive my_voice --negative others --output model.pkl
    vocalid evaluate --model model.pkl
    vocalid verify audio.wav --model model.pkl
    vocalid live --model model.pkl --seconds 4
```

------------------------------------------------------------------------

## Use Cases

- Personal voice unlock systems
- Lightweight identity verification
- Speaker recognition prototypes
- Research experiments in speaker embeddings
- Security analyses for spoof detection

------------------------------------------------------------------------

## Why It Matters

This toolkit allows developers and researchers to:

- Build practical speaker authentication systems quickly
- Learn how ECAPA embeddings work
- Train custom voiceprints without heavy dependencies
- Extend or plug into larger voice systems

------------------------------------------------------------------------

## Contributing

Pull requests are welcome.\
Tests can be run with:

``` python
    pytest -v
```

## Author

**Muhammad Khubaib Ahmad**\

AI/ML Engineer, Data Scientist and Voice Intelligence Researcher

### Portfolio and Links

- [**Portfolio:**](https://huggingface.co/spaces/Khubaib01/KhubaibAhmad_Portfolio)
- [**Gmail:**](khubaib0.1ai@gmail.com)
- [**GitHub:**](https://github.com/Khubaib8281)
- [**LinkedIn:**](https://www.linkedin.com/in/muhammad-khubaib-ahmad-)
- [**Kaggle:**](https://kaggle.com/muhammadkhubaibahmad)
- [**HuggingFace:**](https://huggingface.co/Khubaib01)

------------------------------------------------------------------------

## License

MIT License
