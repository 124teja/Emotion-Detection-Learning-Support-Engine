# 🎓 AI Learning Assistant — Emotion-Aware Tutoring System

An AI-powered Streamlit application that detects a student's emotional state from text input and provides personalized, empathetic learning support using dual emotion-detection models (BiLSTM and BERT) combined with Google Gemini for generating contextual responses.

## Demo

📹 See `demo/app_demo.mp4` for a full walkthrough of the app — field selection, emotion detection, AI-generated responses, and the analytics dashboard.

## Features

- **Dual Model Emotion Detection**: Combines a fine-tuned BiLSTM (with domain-adaptive tuning on student-specific data) and a fine-tuned DistilBERT transformer for robust emotion classification across 5 classes: Bored, Confident, Confused, Curious, Frustrated.
- **Keyword-Enhanced Predictions**: Boosts model confidence using explicit emotion keyword detection to sharpen predictions on clear emotional language.
- **Mixed Emotion Detection**: Identifies when multiple emotions are present above a confidence threshold, rather than forcing a single label.
- **Gemini-Powered Responses**: Generates personalized, field-specific encouragement and study tips based on detected emotion, with automatic fallback to template responses if the API is unavailable.
- **Field-Specific Personalization**: Supports 11 academic fields (Computer Science, Mathematics, Physics, Chemistry, Biology, Engineering, Business, Literature, History, Psychology, Other).
- **Session Analytics Dashboard**: Interactive Plotly visualizations showing emotion distribution, emotional journey over time, and breakdowns by study field.
- **Continuous Learning Logs**: All interactions saved to CSV for future model improvement and pattern analysis.

## Tech Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras (BiLSTM), PyTorch + Hugging Face Transformers (DistilBERT)
- **NLP**: NLTK for text preprocessing
- **AI Integration**: Google Gemini API (`google-genai`)
- **Data**: Pandas, NumPy
- **Visualization**: Plotly Express

## Project Structure

```
emotion detection/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                             # Environment variables (Gemini API key)
├── demo/
│   └── app_demo.mp4                 # Demo video walkthrough
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
└── PROJECT_ANALYSIS_REPORT.md       # Detailed technical writeup
```

## Setup & Installation

1. Clone/download this repository
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
5. Ensure trained model files are present under `models/bltsm/` and `models/bert_emotion_model_final/`
   > **Note:** `model.safetensors` (BERT weights, ~255MB) is excluded from this repository due to GitHub's file size limits. Download it separately from https://drive.google.com/file/d/1o4lsATDqNSnoKo4VEBCtkgJVK0mBtghc/view?usp=sharing and place it in `models/bert_emotion_model_final/`, or retrain using the notebook described in `PROJECT_ANALYSIS_REPORT.md`.
6. Run the app:
   ```bash
   streamlit run app.py
   ```

## Model Training

Both models were trained on Kaggle using GPU acceleration:
- **Datasets**: A combined dataset built from GoEmotions, ISEAR, EmoContext, Empathetic Dialogues, plus template-generated and synthetic examples for underrepresented classes (Bored, Confused, Curious)
- **BiLSTM**: Bidirectional LSTM with focal loss for class imbalance, further fine-tuned on a student-domain-specific dataset (~99% validation accuracy on fine-tuning set)
- **BERT**: Fine-tuned `distilbert-base-uncased` for 3 epochs, achieving ~72% test accuracy with strong macro F1-score (0.77) across all 5 classes

See `PROJECT_ANALYSIS_REPORT.md` for full technical details and performance metrics.

## Author

Tejaswini — Smartbridge Internship Project