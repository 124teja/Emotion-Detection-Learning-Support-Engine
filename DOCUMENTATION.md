# AI Learning Assistant — Complete Project Documentation

**Project:** Emotion-Aware Learning Assistant
**Author:** Tejaswini
**Program:** Smartbridge Internship

---

# Table of Contents

1. Project Overview
2. Features
3. Tech Stack
4. Project Structure
5. Setup & Installation
6. Data Pipeline
7. Model 1: BiLSTM
8. Model 2: BERT
9. Application Architecture
10. How to Use the App
11. Known Limitations
12. Future Improvements

---

# 1. Project Overview

The AI Learning Assistant is a Streamlit application that detects a student's emotional state from free-text input and provides personalized, empathetic learning support. It combines two independently trained deep learning models — a BiLSTM and a fine-tuned BERT transformer — with Google Gemini for generating natural, contextual responses.

The system classifies text into five learning-related emotional states: **Bored, Confident, Confused, Curious, Frustrated**.

# 2. Features

- **Dual Model Emotion Detection**: BiLSTM (with domain-adaptive fine-tuning on student-specific data) and DistilBERT working in parallel for robust classification.
- **Keyword-Enhanced Predictions**: Explicit emotion keywords in user text reinforce and sharpen model confidence.
- **Mixed Emotion Detection**: Reports multiple emotions together when more than one crosses a 15% confidence threshold, rather than forcing a single label.
- **Gemini-Powered Responses**: Personalized, field-specific encouragement and study tips, with automatic fallback to template responses if the API is unavailable.
- **Field-Specific Personalization**: 11 academic fields supported (Computer Science, Mathematics, Physics, Chemistry, Biology, Engineering, Business, Literature, History, Psychology, Other).
- **Session Analytics Dashboard**: Interactive Plotly charts — emotion distribution, emotional journey over time, and field-based breakdowns.
- **Continuous Learning Logs**: All interactions saved to CSV for future analysis and model improvement.

# 3. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Deep Learning | TensorFlow/Keras (BiLSTM), PyTorch + Hugging Face Transformers (DistilBERT) |
| NLP Preprocessing | NLTK |
| AI Response Generation | Google Gemini API (`google-genai`) |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly Express |

# 4. Project Structure

```
emotion detection/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                             # Environment variables (Gemini API key)
├── models/
│   ├── bltsm/                       # BiLSTM model, tokenizer, label encoder
│   └── bert_emotion_model_final/    # Fine-tuned BERT model files
├── src/
│   ├── preprocessing.py             # Text cleaning
│   ├── model.py                     # BiLSTM loading & inference
│   ├── bert_model.py                # BERT loading & inference
│   └── predict.py                   # Keyword-based prediction boosting
├── emotion_response_examples.csv    # Logged interactions
├── emotion_response_mapping.csv     # Emotion-response pair reference
├── README.md
├── PROJECT_ANALYSIS_REPORT.md
├── USER_GUIDE.md
└── DOCUMENTATION.md                 # This file
```

# 5. Setup & Installation

1. Clone/download the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your Gemini API key to `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   ```
5. Confirm trained model files exist under `models/bltsm/` and `models/bert_emotion_model_final/`.
6. Run the app:
   ```bash
   streamlit run app.py
   ```

# 6. Data Pipeline

### 6.1 Dataset Construction

No pre-defined emotion-label mapping was provided for this assignment, so a custom dataset was built by combining and remapping several public emotion datasets:

| Source | Original Purpose | Rows Contributed | Mapped To |
|---|---|---|---|
| GoEmotions | Reddit comment emotion (27 classes) | ~30,000 | Confused, Curious, Confident, Frustrated |
| ISEAR | Emotion diary entries (7 classes) | ~5,000 | Confident, Frustrated |
| EmoContext | Conversational emotion (4 classes) | ~10,000 | Confident, Frustrated |
| Empathetic Dialogues | Dialogue emotion (32 classes) | ~5,000 | Confident, Frustrated, Curious |
| Template-generated | Synthetic | ~2,000 | Bored, Confused (top-up), Curious (top-up) |

**Total combined dataset: ~51,000 labeled examples**

### 6.2 Class Imbalance Handling

The "Bored" class had no natural representation in any source dataset and was addressed through:
1. Template-based synthetic sentence generation (440 examples)
2. Class-weighted focal loss during BiLSTM training
3. Balanced sampling for BERT fine-tuning (max 3,000 examples per class)

### 6.3 Preprocessing

- Lowercasing, URL removal, non-alphabetic character stripping (preserving emotion-relevant punctuation)
- NLTK tokenization with stopword filtering
- Keras `Tokenizer` (30,000-word vocabulary) with sequence padding to 80 tokens

# 7. Model 1: BiLSTM

### 7.1 Architecture

- Embedding layer (128-dim, 30,000 vocab)
- Bidirectional LSTM (128 units, dropout 0.2)
- Dense (128, ReLU) → Dropout (0.3) → Dense (5, Softmax)
- Total parameters: 4,136,709

### 7.2 Training

- Loss: Focal loss (gamma=2.0) for class imbalance
- Optimizer: Adam (lr=1e-3, clipnorm=1.0)
- Callbacks: EarlyStopping (patience=3), ReduceLROnPlateau
- Result: 70.5% test accuracy, macro F1 0.63 (baseline)

### 7.3 Domain-Adaptive Fine-Tuning

The baseline model was further fine-tuned on a 10,000-example student-domain-specific synthetic dataset (2,000 per class), with the embedding layer frozen to preserve general language understanding while adapting to academic vocabulary.

- Result: 99.1% validation accuracy on student domain data
- Saved as `bilstm_student_adaptive.keras`

# 8. Model 2: BERT (DistilBERT)

### 8.1 Architecture

`distilbert-base-uncased` fine-tuned for 5-class sequence classification.

### 8.2 Training

- Optimizer: AdamW (lr=2e-5)
- Epochs: 3
- Batch size: 32
- Training subset: balanced sample (~12,500 examples, max 3,000/class)

### 8.3 Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Bored | 1.00 | 1.00 | 1.00 |
| Confident | 0.75 | 0.80 | 0.78 |
| Confused | 0.64 | 0.66 | 0.65 |
| Curious | 0.64 | 0.63 | 0.64 |
| Frustrated | 0.82 | 0.75 | 0.78 |

**Overall accuracy: 72.1% | Macro F1: 0.77**

BERT significantly outperformed BiLSTM on the previously weak Confused and Curious classes, confirming the value of contextual transformer-based understanding for this task.

# 9. Application Architecture

The Streamlit application (`app.py`) integrates both models with:

- **Keyword-based prediction boosting**: Explicit emotion keywords in user text reinforce model confidence, correcting cases where the raw model is uncertain.
- **Mixed emotion detection**: Emotions scoring above a 15% threshold are reported together.
- **Gemini integration**: Field- and emotion-aware prompts generate personalized responses, with automatic fallback to template responses.
- **Session analytics**: Real-time Plotly dashboards track emotion trends across a session.

# 10. How to Use the App

1. **Select your field of study** from the dropdown.
2. **Describe your problem** in the text box, or click a Quick Example button.
3. **Adjust settings** as needed:
   - *Use AI Response (Gemini)* — live personalized response vs. template fallback
   - *Save to CSV for learning* — logs the interaction
   - *Show analysis details* — reveals full BiLSTM vs BERT comparison
4. Click **"🔍 Get AI Learning Help"**.
5. Review the detected emotion, confidence score, AI response, and suggested strategy.
6. Scroll down to the **Learning Analytics** dashboard to see trends across your session (Emotions / Fields / Summary tabs).

# 11. Known Limitations

1. **No official emotion-label mapping was provided**; the mapping from source dataset labels to the 5 target classes was independently designed, introducing some subjectivity — particularly for "Bored," which has no natural equivalent in any public dataset used.
2. **Training data size**: ~51,000 rows, smaller than a production-scale dataset, due to compute/quota constraints (Kaggle session instability, Gemini API free-tier limits).
3. **BERT training subset**: fine-tuned on a balanced ~12,500-example subset rather than the full combined dataset, for training efficiency.
4. **Raw model confidence gaps**: on complex or contrastive sentences, the underlying models can be overconfident in an incorrect direction; keyword boosting mitigates but does not fully eliminate this.

# 12. Future Improvements

- Expand the "Bored" class with additional real (non-synthetic) examples
- Train BERT on the full combined dataset with more epochs
- Implement a MySQL backend for persistent, multi-session analytics
- Add confidence calibration to prevent keyword boosting from producing unrealistic near-100% confidence scores
