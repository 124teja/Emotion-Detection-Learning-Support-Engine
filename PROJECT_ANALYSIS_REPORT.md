# Project Analysis Report — AI Learning Assistant

## 1. Project Overview

This project implements an emotion-aware learning assistant that detects student emotional states (Bored, Confident, Confused, Curious, Frustrated) from free-text input and generates personalized, supportive responses. The system combines two independently trained deep learning models (BiLSTM and BERT) with Google Gemini for natural language response generation.

## 2. Data Pipeline

### 2.1 Dataset Construction
Since a pre-defined emotion-label mapping was not available, a custom dataset was constructed by combining and remapping several public emotion datasets:

| Source | Original Purpose | Rows Contributed | Mapped To |
|---|---|---|---|
| GoEmotions | Reddit comment emotion (27 classes) | ~30,000 | Confused, Curious, Confident, Frustrated |
| ISEAR | Emotion diary entries (7 classes) | ~5,000 | Confident, Frustrated |
| EmoContext | Conversational emotion (4 classes) | ~10,000 | Confident, Frustrated |
| Empathetic Dialogues | Dialogue emotion (32 classes) | ~5,000 | Confident, Frustrated, Curious |
| Template-generated | Synthetic | ~2,000 | Bored, Confused (top-up), Curious (top-up) |

**Total combined dataset: ~51,000 labeled examples**

### 2.2 Class Imbalance Handling
The "Bored" class had no natural representation in any source dataset. This was addressed through:
1. Template-based synthetic sentence generation (440 examples)
2. Class-weighted focal loss during BiLSTM training
3. Balanced sampling for BERT fine-tuning (max 3,000 examples per class)

### 2.3 Preprocessing
- Lowercasing, URL removal, non-alphabetic character stripping (while preserving emotion-relevant punctuation)
- NLTK tokenization with stopword filtering
- Keras `Tokenizer` (30,000-word vocabulary) with sequence padding to 80 tokens

## 3. Model 1: BiLSTM

### 3.1 Architecture
- Embedding layer (128-dim, 30,000 vocab)
- Bidirectional LSTM (128 units, dropout 0.2)
- Dense (128, ReLU) → Dropout (0.3) → Dense (5, Softmax)
- Total parameters: 4,136,709

### 3.2 Training
- Loss: Focal loss (gamma=2.0) to address class imbalance
- Optimizer: Adam (lr=1e-3, clipnorm=1.0)
- Callbacks: EarlyStopping (patience=3), ReduceLROnPlateau
- Result: 70.5% test accuracy, macro F1 0.63 (baseline)

### 3.3 Domain-Adaptive Fine-Tuning
The baseline model was further fine-tuned on a 10,000-example student-domain-specific synthetic dataset (2,000 per class), with the embedding layer frozen to preserve general language understanding while adapting to academic/learning-context vocabulary.
- Result: 99.1% validation accuracy on student domain data
- Saved as `bilstm_student_adaptive.keras`

## 4. Model 2: BERT (DistilBERT)

### 4.1 Architecture
`distilbert-base-uncased` fine-tuned for 5-class sequence classification.

### 4.2 Training
- Optimizer: AdamW (lr=2e-5)
- Epochs: 3
- Batch size: 32
- Training subset: balanced sample (~12,500 examples, max 3,000/class) for training efficiency

### 4.3 Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Bored | 1.00 | 1.00 | 1.00 |
| Confident | 0.75 | 0.80 | 0.78 |
| Confused | 0.64 | 0.66 | 0.65 |
| Curious | 0.64 | 0.63 | 0.64 |
| Frustrated | 0.82 | 0.75 | 0.78 |

**Overall accuracy: 72.1% | Macro F1: 0.77**

BERT significantly outperformed BiLSTM on the previously weak Confused and Curious classes, confirming the value of contextual transformer-based understanding over the sequential BiLSTM approach for this task.

## 5. Application Architecture

The Streamlit application (`app.py`) integrates both models with:
- **Keyword-based prediction boosting**: Explicit emotion keywords in user text directly reinforce model confidence, correcting cases where the raw model is uncertain or has learned biased patterns from templated training data
- **Mixed emotion detection**: Emotions scoring above a 15% threshold are reported together rather than forcing single-label output
- **Gemini integration**: Field- and emotion-aware prompts generate personalized responses, with automatic fallback to curated template responses on API failure or quota limits
- **Session analytics**: Real-time Plotly dashboards track emotion trends across a session

## 6. Known Limitations

1. **No official emotion-label mapping was provided** for this assignment; the mapping from source dataset labels to the 5 target classes was independently designed and documented above. This introduces some subjectivity, particularly for the "Bored" class, which has no natural equivalent in any public dataset used.
2. **Training data size**: Both models were trained on a substantially smaller dataset (~51,000 rows) than would be used in a production system, due to compute/quota constraints during development (Kaggle session instability, Gemini API free-tier limits).
3. **BERT training subset**: For training efficiency, BERT was fine-tuned on a balanced subset (~12,500 examples) rather than the full combined dataset.
4. **Raw model confidence gaps**: On certain complex or contrastive sentences (e.g., mixing positive and negative sentiment in one sentence), the underlying models can be overconfident in an incorrect direction; keyword boosting mitigates but does not fully eliminate this.

## 7. Future Improvements

- Expand the "Bored" class with additional real (non-synthetic) examples
- Train BERT on the full combined dataset with more epochs, given more stable compute access
- Implement a MySQL backend for persistent, multi-session analytics
- Add confidence calibration to prevent keyword boosting from producing unrealistic near-100% confidence scores
