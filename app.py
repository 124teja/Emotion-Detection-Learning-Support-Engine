from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import streamlit as st
from google import genai

from src.preprocessing import TextPreprocessor
from src.model import BiLSTMModel
from src.bert_model import BERTEmotionClassifier
from src.predict import EmotionPredictor

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=api_key) if api_key else None
print(f"DEBUG: api_key loaded = {api_key is not None}")
print(f"DEBUG: client created = {client is not None}")

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

EMOTION_RESPONSES = {
    'Confused': {'emoji': '😕', 'response': "I see you might be confused. Let me break this down step-by-step...", 'action': 'Show detailed explanation'},
    'Frustrated': {'emoji': '😤', 'response': "I understand this is frustrating! Let's try a simpler approach...", 'action': 'Suggest alternative learning path'},
    'Confident': {'emoji': '💪', 'response': "Great! You're making excellent progress! Ready for the next challenge?", 'action': 'Suggest advanced content'},
    'Bored': {'emoji': '😴', 'response': "Let's make this more engaging. Here are some interactive exercises...", 'action': 'Show interactive content'},
    'Curious': {'emoji': '🤔', 'response': "Excellent question! Here's more in-depth information...", 'action': 'Provide research papers & advanced materials'}
}

QUICK_EXAMPLES = [
    "I'm confused about recursion",
    "Debugging is frustrating",
    "I'm curious about machine learning"
]

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'problem_text' not in st.session_state:
    st.session_state.problem_text = ""


@st.cache_resource
def load_models():
    preprocessor = TextPreprocessor()
    bilstm = BiLSTMModel()
    bert = BERTEmotionClassifier()
    bert.load_model()
    return preprocessor, bilstm, bert


@st.cache_resource
def load_keyword_booster(_classes):
    return EmotionPredictor(_classes)


def get_gemini_response(field, problem, emotion, confidence):
    if not client:
        return None
    try:
        prompt = f"""
You are a helpful learning assistant. A student studying {field} is feeling {emotion} (confidence: {confidence:.1%}) about this problem:

"{problem}"

Provide a clear, supportive response with:
1. Brief acknowledgment of their feeling
2. One specific tip or strategy for {field}
3. One encouraging next step

Use simple, clear language. Keep each point to 1-2 sentences. No markdown formatting.
"""
        response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        return response.text.strip()
    except Exception:
        return None


def save_to_csv(field, problem, emotion, confidence, ai_response):
    """Save new interaction to CSV files."""
    try:
        new_example = {
            'text': problem,
            'emotion': emotion.lower(),
            'confidence': confidence,
            'response': ai_response,
            'field': field,
            'timestamp': datetime.now().isoformat()
        }

        if os.path.exists("emotion_response_examples.csv"):
            df = pd.read_csv("emotion_response_examples.csv")
            df = pd.concat([df, pd.DataFrame([new_example])], ignore_index=True)
        else:
            df = pd.DataFrame([new_example])
        df.to_csv("emotion_response_examples.csv", index=False)

        if os.path.exists("emotion_response_mapping.csv"):
            mapping_df = pd.read_csv("emotion_response_mapping.csv")
            if emotion not in mapping_df['emotion'].values:
                new_mapping = pd.DataFrame([{'emotion': emotion, 'response': ai_response}])
                mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)
                mapping_df.to_csv("emotion_response_mapping.csv", index=False)
        else:
            mapping_df = pd.DataFrame([{'emotion': emotion, 'response': ai_response}])
            mapping_df.to_csv("emotion_response_mapping.csv", index=False)

        return True
    except Exception as e:
        st.error(f"Failed to save to CSV: {e}")
        return False


def main():
    st.title("🤖 Emotion-Aware Learning Assistant")
    st.write("Get personalized help based on your field and emotional state")

    # Function to detect mixed sentiment
    def get_mixed_emotions(scores, threshold=0.15):
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_emotions[0]
        mixed = [primary]

        for emotion, score in sorted_emotions[1:]:
            if score >= threshold:
                mixed.append((emotion, score))

        return mixed if len(mixed) > 1 else [primary]

    def add_to_history(field, problem, emotion, confidence, ai_response, bilstm_scores, bert_result=None):
        mixed_emotions = get_mixed_emotions(bilstm_scores)
        emotion_label = " + ".join([em[0] for em in mixed_emotions]) if len(mixed_emotions) > 1 else emotion

        st.session_state.emotion_history.append({
            'timestamp': datetime.now(),
            'field': field,
            'problem': problem,
            'emotion': emotion_label,
            'confidence': confidence,
            'ai_response': ai_response,
            'all_scores': bilstm_scores,
            'model': 'BiLSTM'
        })

        if bert_result:
            bert_mixed = get_mixed_emotions(bert_result['scores'])
            bert_emotion_label = " + ".join([em[0] for em in bert_mixed]) if len(bert_mixed) > 1 else bert_result['emotion']

            st.session_state.emotion_history.append({
                'timestamp': datetime.now(),
                'field': field,
                'problem': problem,
                'emotion': bert_emotion_label,
                'confidence': bert_result['confidence'],
                'ai_response': ai_response,
                'all_scores': bert_result['scores'],
                'model': 'BERT'
            })

    preprocessor, bilstm_model, bert_model = load_models()
    keyword_booster = load_keyword_booster(bilstm_model.classes)
    status = "✅ Models loaded" if bilstm_model and bert_model else "⚠️ Partial"

    if os.path.exists("emotion_response_examples.csv"):
        examples_df = pd.read_csv("emotion_response_examples.csv")
    else:
        examples_df = pd.DataFrame()

    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard")
        st.write(f"Models: {status}")
        st.write(f"Total Interactions: {len(st.session_state.emotion_history)}")
        st.write(f"CSV Examples: {len(examples_df)}")

        if st.button("Clear History"):
            st.session_state.emotion_history = []
            st.rerun()

        if st.session_state.emotion_history:
            st.subheader("Recent Sessions")
            recent = st.session_state.emotion_history[-3:]
            for item in reversed(recent):
                st.write(f"• {item['field']}: {item['emotion']} ({item['confidence']:.1%})")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Tell us about your learning challenge")
        field = st.selectbox(
            "What field are you studying?",
            ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
             "Engineering", "Business", "Literature", "History", "Psychology", "Other"],
            help="Select your area of study for personalized responses"
        )

        problem = st.text_area(
            f"Describe your {field} problem or challenge:",
            value=st.session_state.problem_text,
            placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
            height=120,
            key="problem_input"
        )

        st.write("**Quick Examples:**")
        ex_cols = st.columns(len(QUICK_EXAMPLES))
        for i, example in enumerate(QUICK_EXAMPLES):
            if ex_cols[i].button(example, use_container_width=True):
                st.session_state.problem_text = example
                st.rerun()

    with col2:
        st.subheader("⚙️ Settings")
        use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
        save_data = st.checkbox("Save to CSV for learning", value=True)
        show_details = st.checkbox("Show analysis details", value=False)

        st.markdown("---")
        st.write("**📊 Predict from Saved Data**")
        use_csv_prediction = st.checkbox("Use CSV-based prediction", value=False)

        if use_csv_prediction and len(examples_df) > 0:
            st.info(f"Using {len(examples_df)} saved examples for prediction")

    if st.button("🔍 Get AI Learning Help", type="primary", use_container_width=True):
        if problem.strip():
            with st.spinner("Analyzing your learning state..."):
                cleaned = preprocessor.clean_text(problem)

                # BiLSTM with keyword boosting
                raw_probs = bilstm_model.predict_proba(cleaned)
                boosted_probs = keyword_booster.boost_with_keywords(cleaned, raw_probs)
                top_idx = int(boosted_probs.argmax())
                bilstm_result = {
                    'emotion': bilstm_model.classes[top_idx],
                    'confidence': float(boosted_probs[top_idx]),
                    'scores': {cls: float(boosted_probs[i]) for i, cls in enumerate(bilstm_model.classes)}
                }

                # BERT with its own built-in keyword weighting (already in bert_model.py)
                bert_result = bert_model.predict(cleaned) if bert_model else None

                emotion_result = bilstm_result
                emotion = emotion_result['emotion']
                confidence = emotion_result['confidence']

                if show_details:
                    st.subheader("📊 Model Predictions Comparison")

                    if bert_result:
                        detail_col1, detail_col2 = st.columns(2)
                    else:
                        detail_col1 = st.columns(1)[0]

                    with detail_col1:
                        st.write("**BiLSTM Student Adaptive**")
                        bilstm_mixed = get_mixed_emotions(bilstm_result['scores'])

                        if len(bilstm_mixed) > 1:
                            mixed_text = " + ".join([f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bilstm_mixed])
                            st.metric("Mixed Emotions", mixed_text, f"Primary: {bilstm_mixed[0][1]:.1%}")
                        else:
                            bilstm_emoji = EMOTION_RESPONSES[bilstm_result['emotion']]['emoji']
                            st.metric("Emotion", f"{bilstm_emoji} {bilstm_result['emotion']}", f"{bilstm_result['confidence']:.1%}")

                        for emotion_name, score in sorted(bilstm_result['scores'].items(), key=lambda x: x[1], reverse=True):
                            st.progress(score, text=f"{emotion_name}: {score:.1%}")

                    if bert_result:
                        with detail_col2:
                            st.write("**BERT Transformer**")
                            bert_mixed = get_mixed_emotions(bert_result['scores'])

                            if len(bert_mixed) > 1:
                                mixed_text = " + ".join([f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bert_mixed])
                                st.metric("Mixed Emotions", mixed_text, f"Primary: {bert_mixed[0][1]:.1%}")
                            else:
                                bert_emoji = EMOTION_RESPONSES[bert_result['emotion']]['emoji']
                                st.metric("Emotion", f"{bert_emoji} {bert_result['emotion']}", f"{bert_result['confidence']:.1%}")

                            for emotion_name, score in sorted(bert_result['scores'].items(), key=lambda x: x[1], reverse=True):
                                st.progress(score, text=f"{emotion_name}: {score:.1%}")

                if use_ai:
                    ai_response = get_gemini_response(field, problem, emotion, confidence)
                    ai_model_used = "Gemini"
                    if ai_response is None:
                        ai_response = EMOTION_RESPONSES[emotion]['response']
                        ai_model_used = "Template Fallback"
                else:
                    ai_response = EMOTION_RESPONSES[emotion]['response']
                    ai_model_used = "Template"

                st.markdown("---")
                st.header("🤖 AI Learning Assistant Response")
                st.info(f"💡 AI Response based on BiLSTM prediction: **{emotion}**")
                st.write(ai_response)

                st.subheader("📖 Additional Support")
                st.info(f"**Strategy:** {EMOTION_RESPONSES[emotion]['action']}")

                with st.expander("🔍 Analysis Details"):
                    st.write(f"**Original Problem:** {problem}")
                    st.write(f"**BiLSTM Processed:** {cleaned}")
                    st.write(f"**BiLSTM Confidence:** {confidence:.3f}")
                    st.write(f"**AI Model:** {ai_model_used}")
                    st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                if save_data:
                    save_success = save_to_csv(field, problem, emotion, confidence, ai_response)
                    if save_success:
                        st.success("💾 Interaction saved to improve future responses!")

                add_to_history(field, problem, emotion, confidence, ai_response, bilstm_result['scores'], bert_result)
        else:
            st.warning("Please describe your problem first.")

    # Analytics Dashboard
    if st.session_state.emotion_history:
        st.markdown("---")
        st.header("📈 Learning Analytics")

        df = pd.DataFrame(st.session_state.emotion_history)

        tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                emotion_counts = df['emotion'].value_counts()
                fig1 = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                              title="Emotion Distribution")
                st.plotly_chart(fig1, use_container_width=True)

            with col_b:
                df_copy = df.copy()
                df_copy['time'] = df_copy['timestamp'].dt.strftime('%H:%M:%S')
                fig2 = px.line(df_copy, x='time', y='confidence', color='emotion',
                                title="Emotional Journey", markers=True)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            if 'model' in df.columns:
                field_emotion = df.groupby(['field', 'emotion', 'model']).size().reset_index(name='count')
                fig3 = px.bar(field_emotion, x='field', y='count', color='emotion', facet_col='model',
                              title="Emotions by Study Field & Model")
            else:
                field_emotion = df.groupby(['field', 'emotion']).size().reset_index(name='count')
                fig3 = px.bar(field_emotion, x='field', y='count', color='emotion',
                              title="Emotions by Study Field")
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.subheader("Overall Statistics")
            col_x, col_y, col_z = st.columns(3)
            col_x.metric("Total Interactions", len(df))
            col_y.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
            col_z.metric("Most Common Emotion", df['emotion'].mode()[0] if len(df) > 0 else "N/A")

            st.write("**Model Breakdown**")
            st.write(df['model'].value_counts())


if __name__ == "__main__":
    main()