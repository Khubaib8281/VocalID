import torch
from .audio_utils import load_audio
import numpy as np


class EmbeddingExtractor:
    def __init__(self, model_path="speechbrain/spkrec-ecapa-voxceleb"):
        # lazy torchaudio patch
        try:
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["sox_io"]
        except Exception:
            pass

        try:
            from speechbrain.inference import EncoderClassifier
        except Exception as e:
            raise ImportError(
                "SpeechBrain could not load. Torchaudio is incompatible.\n"
                f"Original error: {e}"
            )

        self.model = EncoderClassifier.from_hparams(
            source=model_path,
            run_opts={"device": "cpu"},
            savedir="pretrained_models/ecapa",
        )

    def _prepare_waveform(self, wav):
        # Convert numpy → tensor
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav, dtype=torch.float32)

        # Ensure shape (1, T)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # If multichannel, average
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Pad waves shorter than 1 sec (ECAPA expects enough frames)
        min_len = 16000  # 1 s at 16kHz
        if wav.shape[1] < min_len:
            pad_len = min_len - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        return wav

    def _normalize(self, emb):
        emb = np.asarray(emb).squeeze()
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def embed_file(self, path):
        try:
            waveform = load_audio(path)
            waveform = self._prepare_waveform(waveform)
        except Exception:
            return None

        with torch.no_grad():
            emb = self.model.encode_batch(waveform)[0].cpu().numpy()

        return self._normalize(emb)

    def emb_waveform(self, waveform):
        waveform = self._prepare_waveform(waveform)

        with torch.no_grad():
            emb = self.model.encode_batch(waveform)[0].cpu().numpy()

        return self._normalize(emb)

    def extract(self, waveform):
        return self.emb_waveform(waveform)