import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BERTEmotionClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.id2label = None
        self.emotion_labels = None

    def load_model(self, model_path='models/bert_emotion_model_final'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.emotion_labels = [self.id2label[i] for i in range(len(self.id2label))]

    def predict(self, text):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # FIX: previously class_weights = [1.2, 1.8, 0.6, 1.0, 1.4] was applied
        # to EVERY prediction unconditionally. Since 1.8 (Confident) was the
        # single highest weight and 0.6 (Confused) the lowest, this alone
        # biased almost every prediction toward Confident even when the raw
        # model output was flat/uncertain or leaning elsewhere. Now weights
        # start neutral (1.0 = no change) and only shift when an actual
        # keyword match in the text justifies it.
        class_weights = np.ones(len(self.id2label))

        text_lower = text.lower()
        confidence_keywords = ['comfortable', 'confident', 'easy', 'clear now',
                               'got it', 'makes sense', 'understand clearly']
        confusion_keywords = ['confused', 'confusing', 'unclear', 'lost',
                              "don't understand", 'dont understand',
                              "doesn't make sense", 'puzzled']

        confident_idx = self.emotion_labels.index('Confident')
        confused_idx = self.emotion_labels.index('Confused')

        # Confusion checked FIRST since these patterns are more specific
        # (they include negation phrases like "don't understand")
        if any(keyword in text_lower for keyword in confusion_keywords):
            class_weights[confused_idx] *= 2.0
            class_weights[confident_idx] *= 0.5
        elif any(keyword in text_lower for keyword in confidence_keywords):
            class_weights[confident_idx] *= 2.0
            class_weights[confused_idx] *= 0.5

        weighted_probs = probs * class_weights
        pred_id = int(np.argmax(weighted_probs))
        emotion = self.id2label[pred_id]

        return {
            "emotion": emotion,
            "confidence": float(weighted_probs[pred_id] / np.sum(weighted_probs)),
            "scores": {self.id2label[i]: float(weighted_probs[i] / np.sum(weighted_probs)) for i in range(len(self.id2label))},
            "cleaned_text": text.strip()
        }

    def predict_proba(self, text):
        """Raw probabilities without weighting, for combining with BiLSTM elsewhere"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        return probs