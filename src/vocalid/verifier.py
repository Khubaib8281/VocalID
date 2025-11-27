from .embeddings import EmbeddingExtractor
from .model_store import load_model
from .audio_utils import record_audio
from .config import THRESHOLD

class VoiceVerifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.extractor = EmbeddingExtractor()

    def verify_file(self, file_path, threshold=THRESHOLD):
        emb = self.extractor(file_path)
        score = self.model.predict_proba([emb])[0][1]
        verified = score >= threshold
        return verified, float(score)
    
    def verify_file(self, seconds = 4, threshold=THRESHOLD):
        waveform = record_audio(seconds)
        emb = self.extractor(waveform)
        score = self.model.predict_proba([emb])[0][1]
        verified = score >= threshold
        return verified, float(score)   
    
    def verify_array(self, audio_tensor):
        emb = self.get_embedding(audio_tensor)
        pred = self.clf.predict([emb])[0]
        proba = self.clf.predict_proba([emb])[0].max()

        return (pred == "positive"), proba
