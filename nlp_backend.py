import nltk
from langdetect import detect
from indicnlp.tokenize.sentence_tokenize import sentence_split
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import warnings

warnings.filterwarnings("ignore")
nltk.download('punkt_tab')

# ✅ Supported Indian Languages
supported_languages = {
    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'ta': 'Tamil',
    'te': 'Telugu', 'mr': 'Marathi', 'pa': 'Punjabi',
    'sa': 'Sanskrit', 'ml': 'Malayalam'
}

# ✅ Sample Sentiment Word Lists
positive_words = {
    'hi': ['शानदार', 'अद्भुत', 'सकारात्मक', 'उत्कृष्ट'],
    'bn': ['চমৎকার', 'ভালো', 'সুন্দর'],
    'gu': ['સારા', 'શાનદાર'],
    'ta': ['நன்று', 'சிறந்தது'],
    'te': ['మంచి', 'అద్భుతం'],
    'mr': ['उत्तम', 'छान'],
    'pa': ['ਵਧੀਆ', 'ਚੰਗਾ'],
    'sa': ['शुभ', 'उत्तम'],
    'ml': ['നല്ലത്', 'ശ്രേഷ്ഠം']
}

negative_words = {
    'hi': ['खराब', 'बुरा', 'नकारात्मक'],
    'bn': ['খারাপ', 'বাজে'],
    'gu': ['ખરાબ', 'નકારાત્મક'],
    'ta': ['மோசமானது', 'தவறு'],
    'te': ['చెడు', 'నకిలీ'],
    'mr': ['वाईट', 'खराब'],
    'pa': ['ਮਾੜਾ', 'ਨਕਾਰਾਤਮਕ'],
    'sa': ['दुष्ट', 'अधम'],
    'ml': ['ചീത്ത', 'തകരാറായ']
}

import streamlit as st

@st.cache_resource
def load_summarization_model():
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return model, tokenizer

summarizer_model, summarizer_tokenizer = load_summarization_model()


# ✅ Language Mapping for Summarizer
lang_map = {
    'hi': 'hi_IN', 'bn': 'bn_IN', 'gu': 'gu_IN', 'ta': 'ta_IN',
    'te': 'te_IN', 'mr': 'mr_IN', 'pa': 'pa_IN', 'sa': 'sa_IN',
    'ml': 'ml_IN'
}

# 🔍 Detect Language
def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in supported_languages else 'en'
    except:
        return 'en'

# ✂️ Preprocess
def preprocess_text(text, lang):
    if lang in supported_languages:
        sentences = sentence_split(text.strip(), lang)
        words = [word for sent in sentences for word in trivial_tokenize(sent)]
    else:
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
    return sentences, words

# 😊 Semantic Sentiment Analysis
def semantic_analysis(words, lang):
    pos_list = positive_words.get(lang, [])
    neg_list = negative_words.get(lang, [])
    pos_count = sum(word in pos_list for word in words)
    neg_count = sum(word in neg_list for word in words)
    if pos_count > neg_count:
        sentiment = "Positive tone detected."
    elif neg_count > pos_count:
        sentiment = "Negative tone detected."
    else:
        sentiment = "Neutral tone detected."
    return sentiment, pos_count, neg_count

# 📝 Chunked Summarization for Long Essays
def summarize_text(text, lang):
    if lang not in lang_map:
        return "Summarization not supported for this language."

    summarizer_tokenizer.src_lang = lang_map[lang]
    max_chunk_len = 1024  # Max chunk length

    # Tokenize input and split into smaller chunks
    inputs = summarizer_tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    chunks = [input_ids[i:i+max_chunk_len] for i in range(0, len(input_ids), max_chunk_len)]

    summaries = []
    for chunk in chunks:
        chunk = chunk.unsqueeze(0)
        summary_ids = summarizer_model.generate(
            chunk,
            max_length=420,  # You can adjust this value
            num_beams=2,
            early_stopping=True
        )
        summary = summarizer_tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        summaries.append(summary)

    return " ".join(summaries)  # Return combined summary of all chunks

# ✅ Essay Feedback (adjusted for longer essays)
def evaluate_essay(sentences, words):
    feedback = []
    if len(words) < 100:
        feedback.append("Essay is too short. Please elaborate.")
    elif len(words) > 1500:
        feedback.append("Essay is extremely long. Try focusing on key ideas.")
    if len(set(words)) / len(words) < 0.4:
        feedback.append("Try using a more diverse vocabulary.")
    long_sentences = [s for s in sentences if len(s.split()) > 40]
    if long_sentences:
        feedback.append(f"{len(long_sentences)} sentence(s) are quite long. Consider splitting them.")
    if len(sentences) > 1 and sentences[0].strip() == sentences[-1].strip():
        feedback.append("Intro and conclusion are similar. Make the conclusion more unique.")
    return feedback

# 🚀 Main Pipeline
# 🚀 Main Pipeline with Translated Output
def grade_essay(text):
    lang = detect_language(text)
    lang_name = supported_languages.get(lang, 'English/Unsupported')

    sentences, words = preprocess_text(text, lang)
    feedback = evaluate_essay(sentences, words)
    sentiment_msg, pos_count, neg_count = semantic_analysis(words, lang)
    summary = summarize_text(text, lang)

    return {
        'language': f"{lang_name} ({lang})",
        'feedback': feedback,
        'sentiment': f"{sentiment_msg} (Positive: {pos_count}, Negative: {neg_count})",
        'summary': summary
    }

