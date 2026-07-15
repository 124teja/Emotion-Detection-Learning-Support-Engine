import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 80

class BiLSTMModel:
    def __init__(self, model_dir="models/bltsm"):
        try:
            self.model = keras.models.load_model(
                f"{model_dir}/bilstm_student_adaptive.keras"
            )
        except Exception:
            self.model = keras.models.load_model(
                f"{model_dir}/bilstm_student_adaptive.keras", compile=False
            )

        with open(f"{model_dir}/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.classes = list(self.label_encoder.classes_)

    def predict(self, cleaned_text: str):
        if not cleaned_text.strip():
            return {
                'emotion': 'Confused',
                'confidence': 0.5,
                'scores': {cls: 1.0 / len(self.classes) for cls in self.classes},
            }

        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        if not sequence or not sequence[0]:
            return {
                'emotion': 'Confused',
                'confidence': 0.5,
                'scores': {cls: 1.0 / len(self.classes) for cls in self.classes},
            }

        padded = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

        probs = self.model.predict(padded, verbose=0)
        probs = np.array(probs).flatten()

        if len(probs) != len(self.classes):
            probs = np.resize(probs, len(self.classes))

        probs = np.exp(probs) / np.sum(np.exp(probs))

        top_idx = int(np.argmax(probs))
        return {
            'emotion': self.classes[top_idx],
            'confidence': float(probs[top_idx]),
            'scores': {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        }

    def predict_proba(self, cleaned_text: str):
        """
        Returns a normalized (softmax) probability distribution over self.classes.
        Used by app.py before keyword boosting, so it must already be a clean
        probability distribution -- not raw logits.
        """
        if not cleaned_text.strip():
            return np.ones(len(self.classes)) / len(self.classes)

        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        if not sequence or not sequence[0]:
            return np.ones(len(self.classes)) / len(self.classes)

        padded = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

        probs = self.model.predict(padded, verbose=0)
        probs = np.array(probs).flatten()

        if len(probs) != len(self.classes):
            probs = np.resize(probs, len(self.classes))

        # FIX: previously returned raw/unnormalized output. Now applies the
        # same softmax normalization used in predict(), so downstream
        # keyword boosting operates on a true probability distribution.
        probs = np.exp(probs) / np.sum(np.exp(probs))
        return probs


if __name__ == "__main__":
    print("Testing BiLSTM model loading...")
    bilstm = BiLSTMModel()
    print("✅ Model loaded successfully")
    print("Classes:", bilstm.classes)

    result = bilstm.predict("i am confused about this topic")
    print("Test prediction:", result)

    proba = bilstm.predict_proba("i am confused about this topic")
    print("predict_proba (should sum to 1.0):", proba, "sum =", proba.sum())