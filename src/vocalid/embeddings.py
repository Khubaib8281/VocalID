import torch
from .config import EMBEDDING_MODEL
from .audio_utils import load_audio

class EmbeddingExtractor:
    def __init__(self, model_path="speechbrain/spkrec-ecapa-voxceleb"):
        # Lazy import, avoid SpeechBrain crashing on torchaudio
        try:
            import torchaudio
            # Patch missing API if needed
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["sox_io"]
        except Exception:
            pass

        try:
            from speechbrain.inference import EncoderClassifier
        except Exception as e:
            raise ImportError(
                "SpeechBrain failed to import. "
                "Your torchaudio version is incompatible.\n"
                "Install newer torchaudio or use CPU-only fallback.\n"
                f"Original error: {e}"
            )

        self.model = EncoderClassifier.from_hparams(
            source=model_path,
            run_opts={"device": "cpu"},
            savedir="pretrained_models/ecapa",
        )

    def embed_file(self, path):
        waveform = load_audio(path)
        with torch.no_grad():
            emb = self.model.encode_batch(waveform).squeeze().numpy()
        return emb

    def emb_waveform(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        with torch.no_grad():
            emb = self.model.encode_batch(waveform).squeeze().numpy()
        return emb

    # only for tests
    def extract(self, waveform):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform).unsqueeze(0)
        return self.emb_waveform(waveform)
