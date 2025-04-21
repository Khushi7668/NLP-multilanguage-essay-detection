# 📝 Multilingual Essay Evaluation System

This is a **Streamlit-based web application** that evaluates essays written in multiple Indian languages. It performs **language detection**, **text preprocessing**, **sentiment analysis**, **summarization**, and provides **constructive feedback** for the input essay.

---

## 🚀 Features

- 🌐 Supports multiple Indian languages including:
  - Hindi, Bengali, Gujarati, Tamil, Telugu, Marathi, Punjabi, Sanskrit, and Malayalam
- 🔍 Automatic language detection
- ✨ Semantic sentiment analysis using HuggingFace models
- 🧠 Essay summarization using MBart
- 💬 Constructive feedback generation
- 📊 Vocabulary diversity and sentence structure analysis
- ⚡ Built with a clean and interactive Streamlit UI

---

## 📁 Project Structure

```plaintext
essay_evaluation_streamlit/
├── streamlit_app.py            # Streamlit frontend logic
├── nlp_backend.py              # NLP processing: detection, sentiment, summarization, feedback
├── templates/
│   └── index.html              # HTML template (not used in Streamlit but for reference)
├── static/
│   └── style.css               # Optional styling
├── file.txt                    # Example essay input (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
⚙️ Installation
Clone the repository

git clone https://github.com/your-username/essay_evaluation_streamlit.git
cd essay_evaluation_streamlit
Create a virtual environment (recommended)

python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On macOS/Linux
Install dependencies

pip install -r requirements.txt
Run the Streamlit app

streamlit run streamlit_app.py
🧠 Technology Stack
Python 3.10+

Streamlit – UI framework

HuggingFace Transformers – Sentiment & Summarization (MBart)

Langdetect – Language detection

IndicNLP – Preprocessing Indian language texts

NLTK – Tokenization and sentence analysis

✅ Use Case
This system is designed to:

Help teachers evaluate essays in Indian languages quickly

Support students with automated feedback

Enable educational NLP research for regional languages

📌 To Do / Future Enhancements
Add grammar/spelling check per language

Upload support for .txt or .docx essays

Export evaluation as PDF or report

Deploy on cloud (e.g., Streamlit Cloud or HuggingFace Spaces)

🙏 Acknowledgements
HuggingFace Transformers

Streamlit

Indic NLP Library

📄 License
This project is licensed under the MIT License.

✍️ Author
Km Khushi
📧 kk9648259@gmail.com
