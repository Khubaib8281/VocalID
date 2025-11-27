import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from .embeddings import EmbeddingExtractor
from .model_store import save_model, load_model

class VoiceTrainer:
    def __init__(self):
        self.extractor = EmbeddingExtractor()
        self.model = None

    def train(self, positive_paths, negative_paths, save_path="voice_auth.pkl"):
        X, y = [], []

        for p in positive_paths:   
            X.append(self.extractor.embed_file(p))
            y.append(1)

        for p in negative_paths:
            X.append(self.extractor.embed_file(p))
            y.append(0)
    
        X = np.stack(X)  # shape (n_samples, n_features)
        y = np.array(y)

        self.model = LogisticRegression(max_iter=1000)    
        self.model.fit(X, y)

        save_model(self.model, save_path)
        print(f"Model saved at: {save_path}")
        return save_path
    
    def prepare_features(self, pos_files, neg_files):
        X, y = [], []

        for f in pos_files:
            X.append(self.extractor.embed_file(f))
            y.append(1)

        for f in neg_files:
            X.append(self.extractor.embed_file(f))
            y.append(0)

        return np.array(X), np.array(y)
    
    def evaluate(self, positive_files, negative_files):
        if self.model is None:
            raise ValueError("Model is not trained. Train before evaluating.")

        # Extract embeddings for positives
        pos_embeds = [self.embedding_extractor.embed_file(f).numpy() for f in positive_files]

        # Extract embeddings for negatives
        neg_embeds = [self.embedding_extractor.embed_file(f).numpy() for f in negative_files]

        # Build dataset
        X = pos_embeds + neg_embeds
        y = [1] * len(pos_embeds) + [0] * len(neg_embeds)

        # Predict
        preds = self.model.predict(X)

        # Accuracy
        acc = accuracy_score(y, preds)

        # Classification report as string
        report = classification_report(y, preds)

        return {
            "accuracy": acc,
            "report": report
        }


    def save(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        save_model(self.model, path)

    def load(self, path):
        from .model_store import load_model
        self.model = load_model(path)
        if self.model is None:
            raise ValueError(f"Failed to load model from {path}")

