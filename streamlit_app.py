import streamlit as st
from nlp_backend import detect_language, preprocess_text, semantic_analysis, evaluate_essay, summarize_text

# 🔧 Setup Streamlit app
import streamlit as st
from nlp_backend import detect_language, preprocess_text, semantic_analysis, evaluate_essay, summarize_text

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Multilingual Essay Evaluator", layout="centered")

# 🔠 Title and intro
st.title("📝 Multilingual Essay Evaluator")
st.markdown("Evaluate essays written in Indian languages with feedback, sentiment, and summarization.")


# Form to input essay text
essay = st.text_area("Paste your essay here...", height=200)

if st.button("Evaluate Essay"):
    if essay:
        try:
            # Run NLP pipeline
            lang = detect_language(essay)
            sentences, words = preprocess_text(essay, lang)
            feedback = evaluate_essay(sentences, words)
            sentiment_msg, pos_count, neg_count = semantic_analysis(words, lang)
            summary = summarize_text(essay, lang)

            # Display results
            st.subheader("🌐 Detected Language")
            st.write(f"{lang} - {detect_language(essay)}")

            st.subheader("📋 Feedback")
            for point in feedback:
                st.write(f"- {point}")

            st.subheader("🧠 Sentiment")
            st.write(f"{sentiment_msg} (Positive: {pos_count}, Negative: {neg_count})")

            st.subheader("🔠 Word Counts")
            st.write(f"Positive Word Count: {pos_count}")
            st.write(f"Negative Word Count: {neg_count}")

            st.subheader("📝 Summary")
            st.write(summary)

        except Exception as e:
            st.error(f"Error processing essay: {e}")
    else:
        st.warning("Please enter an essay to evaluate.")
MAX_WORDS = 700
if len(essay.split()) > MAX_WORDS:
    st.warning(f"Essay too long! Only processing the first {MAX_WORDS} words for speed.")
    essay = " ".join(essay.split()[:MAX_WORDS])
